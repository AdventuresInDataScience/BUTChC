import copy
import math
import random

from ._tree import _initialize_prob_tree
from ._sampling import _traverse_sample
from ._update import _update_tree
from ._utils import _compute_trial_loss


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
        searchspace:     Dict defining the parameter search space.
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
            'history'      — list of {'params', 'objective', 'loss',
                             'rolling_loss'} dicts, one per trial.
            'loss_history' — per-trial loss values (convenience copy).
            'rolling_loss' — per-trial rolling mean loss (convenience copy).
    """
    tree = (copy.deepcopy(start_prob_tree)
            if start_prob_tree is not None
            else _initialize_prob_tree(searchspace))

    best_value   = -float('inf')
    best_params  = None
    history      = []
    loss_history = []

    rolling_window = max(10, budget // 20)
    tau = budget

    for t in range(1, budget + 1):
        config  = _traverse_sample(tree, temp)
        obj_val = objective(config, *args, **kwargs)

        if obj_val > best_value:
            best_value  = obj_val
            best_params = config

        lambda_t = lambda_ * math.exp(-t / tau)
        beta_t   = math.exp(-t / tau)

        if random.random() < beta_t:
            deltas = _update_tree(tree, config, lambda_t, alpha)
            loss_t = _compute_trial_loss(deltas)
        else:
            loss_t = 0.0

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
        'loss_history': loss_history,
        'rolling_loss': [h['rolling_loss'] for h in history],
    }
