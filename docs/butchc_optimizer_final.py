"""
BUTChC: Bayesian Update Tree-based Configuration/Hyperparameter Optimizer

A tree-structured probabilistic search algorithm that maintains a probability
tree over a hierarchical search space and updates it based on observed
objective values. Supports categorical parameters (with optional conditional
sub-spaces) and continuous parameters.

Continuous parameters are modelled using a weighted KDE reservoir rather than
a parametric distribution. This makes no assumptions about shape, naturally
captures multimodality, and requires no moments. The reservoir is a fixed-size
set of weighted sample points; sampling draws a point proportional to weights
then adds Silverman-bandwidth Gaussian jitter, clipped to [min, max].

Usage example:
    searchspace = {
        'optimizer': {
            'values': ['adam', 'sgd'],
            'next_level': {
                # Sub-parameters that only apply when 'adam' is chosen
                'adam': {'lr': {'min': 1e-4, 'max': 1e-2}},
                # Sub-parameters that only apply when 'sgd' is chosen
                'sgd': {
                    'lr':       {'min': 1e-3, 'max': 1e-1},
                    'momentum': {'min': 0.0,  'max': 0.99},
                },
            },
        },
        'batch_size': {'values': [16, 32, 64, 128]},
        'dropout':    {'min': 0.0, 'max': 0.5},
    }

    def objective(config):
        return -((config['lr'] - 0.005) ** 2)  # dummy

    results = BUTChC_optimize(searchspace, objective, budget=50,
                               lambda_=1.0, alpha=1.0, temp=1.0)
"""

import copy
import math
import random
from collections import defaultdict


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum number of sample points stored per continuous node reservoir.
# More points = better distribution fidelity but marginally more memory.
# 50 is a good balance — enough to capture multimodality, cheap to maintain.
# Can be increased for very complex continuous surfaces if needed.
KDE_RESERVOIR_SIZE = 50


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def softmax(logits, temp=1.0):
    """
    Compute a temperature-scaled softmax over a list of log-space scores.

    Args:
        logits: Unnormalized log-scores (NOT raw probabilities).
        temp:   Temperature > 1 flattens the distribution (more exploration);
                temp < 1 sharpens it (more exploitation).

    Returns:
        List of probabilities that sum to 1.
    """
    # Subtract the max for numerical stability before exponentiating
    scaled = [l / temp for l in logits]
    max_val = max(scaled)
    exp_vals = [math.exp(v - max_val) for v in scaled]
    s = sum(exp_vals)
    return [v / s for v in exp_vals]


def _silverman_bandwidth(reservoir, weights):
    """
    Compute Silverman's rule-of-thumb KDE bandwidth from a weighted reservoir.

    Silverman's rule: h = 1.06 * sigma * n^(-1/5)
    where sigma is the weighted standard deviation of the reservoir and n is
    the effective sample size (sum of weights squared normalisation).

    A larger bandwidth = smoother, more exploratory sampling.
    A smaller bandwidth = sharper, more exploitative sampling around peaks.
    As the reservoir fills with good observations, sigma naturally shrinks,
    focusing sampling without any explicit exploitation parameter.

    Args:
        reservoir: List of float sample values.
        weights:   Corresponding normalised weights (must sum to 1).

    Returns:
        Scalar bandwidth h > 0.
    """
    n = len(reservoir)
    if n < 2:
        # Too few points to compute meaningful bandwidth — return a wide default
        return 1.0

    # Weighted mean
    w_mean = sum(w * x for w, x in zip(weights, reservoir))

    # Weighted variance
    w_var = sum(w * (x - w_mean) ** 2 for w, x in zip(weights, reservoir))
    w_std = math.sqrt(max(w_var, 1e-12))   # guard against zero variance

    # Effective sample size — accounts for unequal weights
    # n_eff = 1 / sum(w^2), analogous to n in the unweighted case
    n_eff = 1.0 / max(sum(w ** 2 for w in weights), 1e-12)

    return 1.06 * w_std * (n_eff ** (-0.2))


# ---------------------------------------------------------------------------
# Tree initialisation
# ---------------------------------------------------------------------------

def _initialize_prob_tree(searchspace):
    """
    Recursively build a probability tree from a search-space definition.

    Each node takes one of two forms:
      • Categorical: {'counts': {...}, 'prob': {...}, ['next_level': {...}]}
      • Continuous:  {'min': float, 'max': float,
                      'reservoir': [float, ...], 'weights': [float, ...]}

    Continuous nodes are initialised with a uniform reservoir of
    KDE_RESERVOIR_SIZE evenly-spaced points across [min, max], each with
    equal weight. This gives a flat prior — no region is initially preferred.
    As trials update the reservoir, weights shift toward good regions and
    the effective bandwidth narrows, focusing future sampling.

    The 'next_level' of a categorical node is a dict keyed by choice value,
    where each entry is a probability sub-tree for parameters that are only
    relevant when that choice is made.

    Args:
        searchspace: Dict mapping parameter names to their definitions.

    Returns:
        Nested dict representing the initialised probability tree.
    """
    tree = {}
    for key, val in searchspace.items():

        node = {}

        if isinstance(val, dict) and "values" in val:
            # --- Categorical parameter ---
            # Start with uniform counts (Laplace smoothing seed = 1)
            node['counts'] = {k: 1 for k in val['values']}
            node['prob']   = {k: 1 / len(val['values']) for k in val['values']}

            if 'next_level' in val:
                node['next_level'] = {
                    choice: _initialize_prob_tree(sub_space)
                    for choice, sub_space in val['next_level'].items()
                }

        elif isinstance(val, dict) and 'min' in val and 'max' in val:
            # --- Continuous parameter (KDE reservoir) ---
            lo, hi = val['min'], val['max']
            node['min'] = lo
            node['max'] = hi

            # Initialise reservoir as evenly-spaced points across [min, max]
            # with uniform weights — a flat, uninformative prior.
            # Using KDE_RESERVOIR_SIZE points gives good initial coverage.
            n = KDE_RESERVOIR_SIZE
            if n > 1:
                step = (hi - lo) / (n - 1)
                reservoir = [lo + i * step for i in range(n)]
            else:
                reservoir = [(lo + hi) / 2]
            uniform_w = 1.0 / len(reservoir)

            node['reservoir'] = reservoir
            node['weights']   = [uniform_w] * len(reservoir)
            # Track total weight mass — used to compute normalised update deltas
            node['total_weight'] = 1.0

        tree[key] = node

    return tree


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _sample_node(node, temp):
    """
    Draw a single value from a probability-tree node.

    For categorical nodes the temperature controls exploration vs exploitation:
    high temp → near-uniform sampling; low temp → greedy towards best seen.

    For continuous nodes a KDE draw is used:
      1. Sample a reservoir point proportional to its weight
      2. Add Gaussian jitter with Silverman bandwidth
      3. Clip to [min, max] to stay within the defined range
    As good regions accumulate weight, the bandwidth naturally narrows and
    sampling concentrates — no explicit temperature needed for continuous nodes.

    Args:
        node: A single node dict from the probability tree.
        temp: Softmax temperature for categorical nodes (must be > 0).

    Returns:
        The sampled value (a key string for categorical, float for continuous).
    """
    if 'prob' in node:
        # --- Categorical ---
        keys  = list(node['prob'].keys())
        probs = [node['prob'][k] for k in keys]

        # Softmax expects log-space scores — convert raw probs to log-space
        # first to avoid distorting the temperature scaling
        log_probs = [math.log(max(p, 1e-12)) for p in probs]
        weights   = softmax(log_probs, temp)

        return random.choices(keys, weights=weights, k=1)[0]

    elif 'reservoir' in node:
        # --- Continuous (KDE draw) ---
        reservoir = node['reservoir']
        weights   = node['weights']

        # 1. Weighted draw of a centre point from the reservoir
        centre = random.choices(reservoir, weights=weights, k=1)[0]

        # 2. Silverman bandwidth jitter — adds smooth exploration around
        #    the chosen point proportional to local density spread
        h      = _silverman_bandwidth(reservoir, weights)
        jitter = random.gauss(0, h)

        # 3. Clip to valid range — jitter can push outside [min, max]
        return max(node['min'], min(node['max'], centre + jitter))

    else:
        raise ValueError(
            f"Malformed node — expected 'prob' (categorical) or "
            f"'reservoir' (continuous). Got keys: {list(node.keys())}"
        )


def _traverse_sample(tree, temp):
    """
    Recursively sample a full configuration from the probability tree.

    When a categorical parameter has an associated 'next_level', the sampled
    choice determines which conditional sub-tree is explored next, so only
    relevant sub-parameters are included in the returned config.

    Args:
        tree: A (sub-)tree dict as produced by _initialize_prob_tree.
        temp: Softmax temperature forwarded to _sample_node.

    Returns:
        Flat dict of {parameter_name: sampled_value} for the whole (sub-)tree.
    """
    config = {}
    for key, node in tree.items():
        val = _sample_node(node, temp)
        config[key] = val

        # If this categorical choice has a conditional sub-space, recurse into
        # the sub-tree for the chosen value and merge its params into config.
        if 'next_level' in node and val in node['next_level']:
            sub_config = _traverse_sample(node['next_level'][val], temp)
            config.update(sub_config)

    return config


# ---------------------------------------------------------------------------
# Tree updates
# ---------------------------------------------------------------------------

def _update_tree(tree, config, lambda_, alpha):
    """
    Update the probability tree in-place after observing a configuration.
    Returns a list of normalised per-node deltas for loss computation.

    Categorical: Laplace-smoothed count update.
        count[chosen] += lambda_
        prob[k] = (count[k] + alpha) / (sum(counts) + alpha * n_choices)

    Continuous (KDE reservoir):
        1. Add observed value as a new reservoir point with weight lambda_
        2. Renormalise all weights to sum to 1
        3. If reservoir exceeds KDE_RESERVOIR_SIZE, prune the lowest-weight
           point — old, low-signal observations fade out naturally over time

    Higher lambda_ gives more weight to recent observations (faster adaptation).
    Higher alpha keeps categorical probabilities closer to uniform (more smoothing).

    Args:
        tree:    Probability tree to update.
        config:  The sampled configuration that was just evaluated.
        lambda_: Weight applied to the new observation (> 0).
        alpha:   Laplace smoothing strength for categorical nodes (> 0).

    Returns:
        List of normalised scalar deltas, one per updated node, each in [0, 1].
    """
    deltas = []

    for key, node in tree.items():
        if key not in config:
            # This parameter wasn't part of the current config branch
            continue

        val = config[key]

        if 'prob' in node:
            # --- Snapshot probs before update for delta computation ---
            probs_before = dict(node['prob'])

            # --- Update categorical counts & probabilities ---
            node['counts'][val] += lambda_

            # Laplace-smoothed probability estimate
            total = sum(node['counts'].values()) + alpha * len(node['counts'])
            for k in node['counts']:
                node['prob'][k] = (node['counts'][k] + alpha) / total

            # Delta: mean absolute prob shift per choice, in [0, 1].
            # Dividing by n_choices normalises across nodes of different sizes.
            n_choices = len(node['prob'])
            delta = sum(
                abs(node['prob'][k] - probs_before[k])
                for k in node['prob']
            ) / n_choices
            deltas.append(delta)

            # Recurse into the conditional sub-tree for the chosen value
            if 'next_level' in node and val in node['next_level']:
                sub_deltas = _update_tree(node['next_level'][val], config, lambda_, alpha)
                deltas.extend(sub_deltas)

        elif 'reservoir' in node:
            # --- KDE reservoir update ---

            # Snapshot weighted mean before update for delta computation.
            # Weighted mean is the best single-number summary of the current
            # distribution — its shift captures how much the model changed.
            w_mean_before = sum(
                w * x for w, x in zip(node['weights'], node['reservoir'])
            )

            # 1. Add new observation with weight lambda_
            node['reservoir'].append(val)
            node['weights'].append(lambda_)

            # 2. Prune lowest-weight point if over reservoir budget.
            #    This keeps memory bounded while naturally expiring old,
            #    low-signal observations as new ones accumulate.
            if len(node['reservoir']) > KDE_RESERVOIR_SIZE:
                min_idx = node['weights'].index(min(node['weights']))
                node['reservoir'].pop(min_idx)
                node['weights'].pop(min_idx)

            # 3. Renormalise weights to sum to 1
            total_w = sum(node['weights'])
            node['weights'] = [w / total_w for w in node['weights']]

            # Delta: weighted mean shift normalised by node range, in [0, 1]
            w_mean_after = sum(
                w * x for w, x in zip(node['weights'], node['reservoir'])
            )
            node_range = node['max'] - node['min']
            delta = (
                abs(w_mean_after - w_mean_before) / node_range
                if node_range > 0 else 0.0
            )
            deltas.append(delta)

    return deltas


def _compute_trial_loss(deltas):
    """
    Aggregate per-node deltas from a single trial into a scalar loss value.

    Loss here is not the objective value — it measures how much the model
    was surprised by this trial, i.e. the mean magnitude of probability
    updates across all nodes touched. High loss = model was far from
    expecting this config's outcome. Low loss = model is converging.

    Interpretation:
        - High early loss is expected and healthy (model is learning fast)
        - Steadily declining loss indicates convergence
        - Loss plateau followed by rise may indicate overfitting to noise
        - Near-zero loss means the tree has effectively stopped updating

    Args:
        deltas: List of normalised per-node deltas returned by _update_tree.

    Returns:
        Scalar mean loss for this trial, or 0.0 if no updates were made.
    """
    if not deltas:
        return 0.0
    return sum(deltas) / len(deltas)


# ---------------------------------------------------------------------------
# Main optimisation loop
# ---------------------------------------------------------------------------

def BUTChC_optimize(
    searchspace,
    objective,
    budget,
    lambda_,
    alpha,
    temp=1.0,
    start_prob_tree=None,
    verbose=True,
    *args,
    **kwargs,
):
    """
    Run the BUTChC optimisation loop (maximisation).

    Samples a configuration from the probability tree, evaluates it, then
    updates the tree to nudge future samples toward promising regions.

    Args:
        searchspace:     Dict defining the parameter search space (see module
                         docstring for format).
        objective:       Callable(config, *args, **kwargs) -> float.
                         Must return a scalar; higher is better.
        budget:          Total number of objective evaluations to run.
        lambda_:         Update weight — higher values make the tree adapt
                         faster but can cause premature convergence.
        alpha:           Laplace smoothing for categorical nodes — higher
                         values keep probabilities closer to uniform longer.
        temp:            Softmax temperature for sampling — higher values
                         encourage exploration, lower values exploitation.
        start_prob_tree: Optional pre-built probability tree (e.g. from a
                         previous run's 'prob_tree' output) to warm-start.
        verbose:         If True, print per-trial progress to stdout.
        *args, **kwargs: Extra arguments forwarded to `objective`.

    Returns:
        Dict with keys:
            'best_params'  — config dict that achieved the highest objective.
            'best_value'   — the corresponding objective value.
            'prob_tree'    — the final (updated) probability tree.
            'history'      — list of {'params': ..., 'objective': ...,
                             'loss': ..., 'rolling_loss': ...} dicts,
                             one per trial, in evaluation order.
            'loss_history' — per-trial loss values (convenience copy).
            'rolling_loss' — per-trial rolling mean loss (convenience copy).
    """
    # Deep-copy a warm-start tree, or build a fresh one from the search space
    tree = (copy.deepcopy(start_prob_tree)
            if start_prob_tree is not None
            else _initialize_prob_tree(searchspace))

    best_value   = -float('inf')   # Tracks the best objective seen so far
    best_params  = None
    history      = []
    loss_history = []              # Per-trial model surprise/update magnitude

    # Rolling loss window scales with budget so it's always ~5% of run length.
    # Provides a smoothed convergence signal without needing a fixed window size.
    rolling_window = max(10, budget // 20)

    # Decay time constant for both schedules. Set to `budget` so that by the
    # final trial: lambda has decayed to ~37% of its initial value, and update
    # probability has likewise reached ~37% — a gentle, principled anneal.
    # If finer control is needed in future, expose as args with a multiplier
    # in the range 0.75–1.5x budget (smaller = faster decay, larger = slower).
    tau = budget

    for t in range(1, budget + 1):
        # 1. Sample a configuration from the current probability tree
        config = _traverse_sample(tree, temp)

        # 2. Evaluate the objective function on the sampled config
        obj_val = objective(config, *args, **kwargs)

        # 3. Track the best result seen across all trials
        if obj_val > best_value:
            best_value  = obj_val
            best_params = config

        # 4. Decay schedules — both use the same tau for simplicity.
        #    lambda_t: reduces update weight over time, preventing late-stage
        #              observations from overwriting well-founded estimates.
        #    beta_t:   decaying update probability acts like a shrinking batch
        #              size — dense early updates stabilise the tree quickly,
        #              sparse later updates let good estimates settle.
        lambda_t = lambda_ * math.exp(-t / tau)
        beta_t   = math.exp(-t / tau)

        # 5. Conditionally update the tree and capture per-node deltas.
        #    Deltas are only meaningful when an update actually occurs —
        #    skipped trials contribute 0.0 loss to keep the history aligned.
        if random.random() < beta_t:
            deltas = _update_tree(tree, config, lambda_t, alpha)
            loss_t = _compute_trial_loss(deltas)
        else:
            loss_t = 0.0

        # 6. Log trial results including loss
        loss_history.append(loss_t)
        rolling_loss = (
            sum(loss_history[-rolling_window:]) /
            min(t, rolling_window)
        )
        history.append({
            'params':       config,
            'objective':    obj_val,
            'loss':         loss_t,
            'rolling_loss': rolling_loss,
        })

        if verbose:
            print(
                f"Trial {t:>{len(str(budget))}}/{budget} | "
                f"Objective = {obj_val:.6f} | "
                f"Best so far = {best_value:.6f} | "
                f"Loss = {loss_t:.6f} | "
                f"Rolling loss = {rolling_loss:.6f} | "
                f"λ = {lambda_t:.4f} | β = {beta_t:.4f}"
            )

    return {
        'best_params':  best_params,
        'best_value':   best_value,
        'prob_tree':    tree,
        'history':      history,
        # Convenience top-level loss summary — avoids having to unpack
        # history just to plot or threshold on convergence
        'loss_history': loss_history,
        'rolling_loss': [h['rolling_loss'] for h in history],
    }
