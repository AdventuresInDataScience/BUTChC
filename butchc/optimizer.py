"""The BUTChC optimisation loop."""

import bisect
import copy
import random

from ._parallel import evaluate_batch, resolve_batch
from ._sampling import traverse_sample
from ._tree import initialize_prob_tree
from ._update import quality_weight, update_tree
from ._utils import compute_trial_loss, is_finite_number
from ._validate import (
    check_tree_matches_searchspace,
    validate_hyperparameters,
    validate_prob_tree,
    validate_searchspace,
)


#: How sharply categorical probabilities are peaked at the *end* of a run.
#: Choice values are raised to ``COMMITMENT * progress``, so the exponent
#: ramps from 0 — where every choice is equally likely whatever the values
#: say — up to ``COMMITMENT``.
#:
#: The ramp starts at 0, not 1. An exponent of 1 is already decisive: a branch
#: with four times its sibling's mean score takes four times the budget from
#: the first trials on, which is enough to starve a branch whose sub-space has
#: not been tuned yet — and a wide sub-space always scores badly until it is.
#: The "Branch trap" problem in benchmarks/problems.py isolates this case.
COMMITMENT = 8.0

#: Smallest per-node delta counted as real movement. Recomputing a
#: distribution that has not changed still returns deltas of order 1e-17 from
#: floating-point rounding, and without a floor those are counted as updates.
MOVEMENT_EPSILON = 1e-12


def BUTChC_optimize(
    searchspace,
    objective,
    budget,
    lambda_=2.0,
    alpha=3.0,
    temp=1.0,
    start_prob_tree=None,
    verbose=True,
    *,
    gamma=0.85,
    explore=0.05,
    n_warmup=None,
    seed=None,
    batch=1,
    executor=None,
    **kwargs,
):
    """
    Run the BUTChC optimisation loop (maximisation).

    Each trial samples a configuration from the probability tree, evaluates
    it, and updates the tree in proportion to how well it scored relative to
    every trial seen so far.

    Args:
        searchspace:     Dict defining the parameter search space, or a
                         ``ConfigSpace.ConfigurationSpace``. A
                         ``ConfigurationSpace`` is converted on entry; see
                         ``butchc.interop``, and note that conjunctions,
                         multiple parents and forbidden clauses have no tree
                         representation and are refused rather than
                         approximated.
        objective:       ``Callable(config, **kwargs) -> float``. Higher is
                         better. Non-finite returns are recorded but excluded
                         from ranking and from best-tracking.
        budget:          Number of objective evaluations.
        lambda_:         Sharpness of the continuous archive's rank weighting.
                         Higher concentrates the KDE on the best archived
                         points. Categorical nodes ignore it.
        alpha:           Smoothing for categorical nodes, in pseudo-visits.
                         Higher keeps probabilities nearer uniform for longer.
        temp:            Softmax temperature for categorical sampling. ``> 1``
                         explores more, ``< 1`` exploits more.
        start_prob_tree: A previous run's ``prob_tree``, to warm-start. It is
                         checked against ``searchspace`` and copied, never
                         mutated.
        verbose:         Print per-trial progress to stdout.
        gamma:           Quantile gate on the *continuous* archive. Only
                         trials ranking above this quantile are archived.
                         Categorical nodes score every trial by rank and
                         ignore it.
        explore:         Per-node probability of drawing uniformly rather than
                         from the model. The floor is per node, not per
                         configuration. Set to 0 to disable.
        n_warmup:        Trials drawn uniformly from every node before
                         modelling begins. Defaults to 0. These trials do not
                         reach any continuous archive — ``quality`` is held at
                         0 — but categorical nodes still record their visit and
                         rank, because that unbiased sample is what stops a
                         branch being judged before it has been tried. Not
                         skipped when ``start_prob_tree`` is given; pass 0 if a
                         warm-started run should go straight to modelling.
        seed:            Seed for this run's private RNG. ``None`` draws from
                         global entropy. Global ``random`` state is never
                         touched or relied upon.
        batch:           Configurations drawn per model update. ``1`` is
                         sequential, and is what earns model-based search its
                         sample efficiency. ``k > 1`` draws ``k`` configs from
                         one state of the tree, so the model is stale for the
                         other ``k - 1`` — pay that to keep an ``executor``
                         busy, not for smoother convergence. ``-1`` resolves
                         to ``os.cpu_count()``.
        executor:        Anything exposing an order-preserving
                         ``map(fn, iterable)``, or a ``submit(fn, arg)``
                         returning futures — a ``concurrent.futures``
                         executor, a ``multiprocessing.Pool``, joblib, dask.
                         ``None`` evaluates the batch inline. BUTChC never
                         creates a pool of its own.
        **kwargs:        Forwarded to every call of ``objective``.

    Returns:
        Dict with keys:
            ``best_params``   Config with the highest objective, or None if
                              no trial returned a finite value.
            ``best_value``    Its objective value, or ``-inf``.
            ``prob_tree``     Final probability tree; pass as
                              ``start_prob_tree`` to continue.
            ``history``       Per-trial dicts with ``params``, ``objective``,
                              ``loss``, ``rolling_loss``, ``quality``,
                              ``updated``, ``gated`` and ``batch_index``.
            ``loss_history``  Per-trial loss values.
            ``rolling_loss``  Per-trial rolling mean loss.
            ``n_updates``     Number of trials that moved the tree.
            ``n_gated``       Number of trials that cleared the quality gate
                              and so reached a continuous archive.
            ``n_batches``     Number of model updates performed. Equals
                              ``budget`` when ``batch=1``.

    Raises:
        SearchSpaceError: If ``searchspace`` is malformed.
        ValueError:       If a hyperparameter is out of range, or if
                          ``start_prob_tree`` does not match ``searchspace``.
        TypeError:        If ``objective`` is not callable or returns a
                          non-numeric value, or ``executor`` has neither
                          ``map`` nor ``submit``.
        RuntimeError:     If the objective cannot be sent to ``executor``.
    """
    if not callable(objective):
        raise TypeError(f"objective must be callable, got {type(objective).__name__}")

    searchspace, objective = _coerce_searchspace(searchspace, objective)
    validate_searchspace(searchspace)

    if n_warmup is None:
        # A freshly initialised tree is an even grid at even weights, so its
        # early draws are already close to uniform and a separate warm-up
        # phase buys a second copy of that at the cost of budget. The quality
        # gate suppresses updates from the first trials in any case.
        n_warmup = 0
    validate_hyperparameters(budget, lambda_, alpha, temp, gamma, explore,
                             n_warmup, batch)
    batch_size = resolve_batch(batch)

    if start_prob_tree is not None:
        validate_prob_tree(start_prob_tree)
        check_tree_matches_searchspace(start_prob_tree, searchspace)
        tree = copy.deepcopy(start_prob_tree)
    else:
        tree = initialize_prob_tree(searchspace)

    rng = random.Random(seed)

    best_value = -float("inf")
    best_params = None
    history = []
    loss_history = []
    observed = []          # sorted finite objectives, for rank computation
    n_updates = 0
    n_gated = 0
    n_batches = 0

    rolling_window = max(10, budget // 20)
    width = len(str(budget))

    done = 0
    while done < budget:
        # The final batch is truncated rather than allowed to overrun, so a
        # budget is exactly a budget however it divides.
        size = min(batch_size, budget - done)
        n_batches += 1

        # --- draw the whole batch from one state of the tree ----------------
        drawn = []
        for offset in range(size):
            t = done + offset + 1
            warming = t <= n_warmup
            # Warm-up forces every node to draw uniformly.
            config, trace = traverse_sample(
                tree, temp, rng, explore=1.0 if warming else explore
            )
            drawn.append((t, warming, config, trace))

        raw = evaluate_batch(objective, [d[2] for d in drawn], executor, kwargs)

        # --- rank every member against the state it was sampled from --------
        # Ranking a batch member against its own batch-mates would give it
        # information the sampler did not have, and would make its score depend
        # on its position in the batch. Ranking all of them against the
        # pre-batch observations is exactly what "the model was stale for k-1
        # trials" means. At batch=1 it is identical to ranking after insertion.
        had_history = len(observed) >= 1
        scored = []
        for (t, warming, config, trace), value in zip(drawn, raw):
            obj_val = _as_float(value, t)
            finite = is_finite_number(obj_val)

            rank = 0.5
            quality = 0.0
            if finite:
                rank = _quantile_rank(observed, obj_val)
                if not warming and had_history:
                    quality = quality_weight(rank, gamma)
            scored.append((t, config, trace, obj_val, finite, rank, quality))

        for entry in scored:
            if entry[4]:
                bisect.insort(observed, entry[3])

        # --- apply the updates in sample order ------------------------------
        # Sample order, not completion order: a seeded run must give the same
        # answer whichever worker finishes first.
        for t, config, trace, obj_val, finite, rank, quality in scored:
            if finite and obj_val > best_value:
                best_value = obj_val
                best_params = config

            # Categorical scores are means over visits, so every trial counts
            # against its branch whether or not it clears the gate — including
            # warm-up trials, which are the unbiased sample that stops a branch
            # being judged before it has been tried.
            # ``neutral`` and ``discount`` are left unpassed so that
            # ``update_tree`` resolves them from ``_update``'s module
            # constants at call time. Passing a literal here would silently
            # override ``NEUTRAL_QUALITY``, which is documented as patchable
            # and swept by benchmarks/tune.py — the knob would look live and
            # do nothing.
            deltas = update_tree(
                tree, trace, lambda_, alpha, quality, obj_val, rng,
                sharpen=COMMITMENT * (t / budget),
                rank=rank,
            )
            gated = quality > 0.0
            # "Moved" means a node's distribution actually shifted. An updated
            # node that lands on the same numbers contributes a delta of 0 and
            # does not count.
            updated = any(d > MOVEMENT_EPSILON for d in deltas)
            loss_t = compute_trial_loss(deltas)
            n_updates += int(updated)
            n_gated += int(gated)

            loss_history.append(loss_t)
            rolling = sum(loss_history[-rolling_window:]) / min(t, rolling_window)

            history.append(
                {
                    "params": config,
                    "objective": obj_val,
                    "loss": loss_t,
                    "rolling_loss": rolling,
                    "quality": quality,
                    "updated": updated,
                    "gated": gated,
                    "batch_index": n_batches - 1,
                }
            )

            if verbose:
                print(
                    f"Trial {t:>{width}}/{budget} | "
                    f"Objective = {obj_val:.6f} | "
                    f"Best so far = {best_value:.6f} | "
                    f"Quality = {quality:.3f} | "
                    f"Loss = {loss_t:.6f} | "
                    f"Rolling loss = {rolling:.6f}"
                    + ("  [warmup]" if t <= n_warmup else "")
                )

        done += size

    return {
        "best_params": best_params,
        "best_value": best_value,
        "prob_tree": tree,
        "history": history,
        "loss_history": loss_history,
        "rolling_loss": [h["rolling_loss"] for h in history],
        "n_updates": n_updates,
        "n_gated": n_gated,
        "n_batches": n_batches,
    }


def _coerce_searchspace(searchspace, objective):
    """
    Accept a ConfigSpace ``ConfigurationSpace`` wherever a dict is expected.

    Detection is by duck-typing rather than ``isinstance``, so ConfigSpace is
    never imported unless one is actually passed and the package stays
    dependency-free.

    Conversion also yields the constants ConfigSpace holds outside the search
    and the casts that log-scaled integers need, so the objective is wrapped
    here. Returning those to the caller to re-apply themselves would be a
    half-integration, and a silent one: the objective would simply receive
    incomplete configs.
    """
    if isinstance(searchspace, dict):
        return searchspace, objective

    # Three tests, because ConfigSpace has renamed accessors across major
    # versions and pinning detection to any single one makes a future rename
    # look like "you passed the wrong type". The module check is the reliable
    # one; the others keep subclasses and wrappers working.
    module = (type(searchspace).__module__ or "").split(".")[0]
    duck = (
        module == "ConfigSpace"
        or hasattr(searchspace, "get_hyperparameters")
        or (hasattr(searchspace, "values")
            and (hasattr(searchspace, "get_default_configuration")
                 or hasattr(searchspace, "sample_configuration")))
    )
    if not duck:
        raise TypeError(
            f"searchspace must be a dict or a ConfigSpace ConfigurationSpace, "
            f"got {type(searchspace).__name__}"
        )

    from .interop import from_configspace, wrap_objective

    space, fixed, casts = from_configspace(searchspace)
    return space, wrap_objective(objective, fixed, casts)


def _as_float(value, trial):
    """
    Coerce an objective's return to a float, or explain why it cannot be.

    Anything with a real-number conversion is accepted, so ``numpy.float32``,
    ``numpy.int64``, ``Decimal`` and single-element arrays all work. Booleans
    and strings are rejected: both convert cleanly and neither is plausibly an
    objective value.
    """
    if isinstance(value, bool) or isinstance(value, (str, bytes)):
        raise TypeError(
            f"objective returned {type(value).__name__} on trial {trial}; "
            f"a float is required"
        )
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"objective returned {type(value).__name__} on trial {trial}; "
            f"a float is required ({exc})"
        ) from exc


def _quantile_rank(sorted_values, value):
    """
    Midrank of ``value`` among previously observed objectives.

    Ties count half. Counting only values *strictly* below would rank a trial
    as though it lost to everything it drew with, which puts the joint-best
    trials of a plateaued objective — accuracy on a small validation set, any
    integer score, anything that saturates — at the bottom of their own tie
    group, where the gate rejects nearly all of them.

    Returns 0.5 for the first observation, because there is nothing to compare
    it against and 0.5 is the rank of a configuration drawn at random. The
    continuous archive ignores the first trial anyway, but categorical nodes
    consume every rank, so returning 1.0 would credit whichever branch trial 1
    happened to draw with a maximal score it had not earned.
    """
    n = len(sorted_values)
    if not n:
        return 0.5
    below = bisect.bisect_left(sorted_values, value)
    at_or_below = bisect.bisect_right(sorted_values, value)
    return (below + at_or_below) / (2.0 * n)
