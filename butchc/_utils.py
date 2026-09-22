"""Numeric helpers shared across the package."""

import math
import sys

#: Maximum number of points retained in a continuous node's KDE reservoir.
#:
#: Reduced from 50 by the 0.6.0 sweep, and it carries more of that sweep's gain
#: than any other knob: reverting this one value alone, with every other new
#: default kept, turns a 13-better/4-worse result into 5-better/12-worse. A
#: smaller archive is a sharper archive, since the weights decay over rank and
#: there are fewer ranks to spread across.
KDE_RESERVOIR_SIZE = 25

#: Bandwidth floor, as a fraction of a node's internal range. Prevents the
#: KDE from collapsing to a delta function and losing all local exploration.
#:
#: Measured in two stages. Against the 0.01 this replaces, 0.001 wins 200-34
#: across all 18 problems at 20 paired seeds, with 20-0 sweeps on `5D sphere`,
#: `10D sphere` and `20D sphere`. The floor exists to stop the KDE collapsing
#: to a delta function, and 0.01 of a node's range was far more jitter than
#: that needs — enough to stop the archive ever resolving an optimum finely.
#:
#: The sweep then moved it to 0.0003, but only in its second round, after
#: `KDE_RESERVOIR_SIZE` had dropped to 25. The two interact: both concentrate
#: the continuous model, so the best floor depends on the archive size. At
#: reservoir 50 the ordering reverses and 0.001 is better. Retune the pair
#: together, never one alone.
MIN_BANDWIDTH_FRACTION = 0.0003

#: Grid points a continuous node always retains when seeded with a prior, so
#: that no prior_strength can make the rest of the range unreachable.
MIN_GRID_POINTS = 10


def softmax(logits, temp=1.0):
    """
    Temperature-scaled softmax over log-space scores.

    Args:
        logits: Unnormalised log-scores (NOT raw probabilities).
        temp:   ``> 1`` flattens the distribution (more exploration);
                ``< 1`` sharpens it (more exploitation). Must be > 0.

    Returns:
        List of probabilities summing to 1.

    Raises:
        ValueError: If ``temp <= 0`` or ``logits`` is empty.
    """
    if temp <= 0:
        raise ValueError(f"temp must be > 0, got {temp}")
    if not logits:
        raise ValueError("softmax requires at least one logit")

    scaled = [l / temp for l in logits]
    max_val = max(scaled)
    exp_vals = [math.exp(v - max_val) for v in scaled]
    s = sum(exp_vals)
    return [v / s for v in exp_vals]


def weighted_mean(values, weights):
    """Weighted mean of ``values``. Assumes ``weights`` sums to 1."""
    return sum(w * x for w, x in zip(weights, values))


def effective_sample_size(weights):
    """
    Kish effective sample size, ``1 / sum(w^2)``, for normalised weights.

    Reports how many independent observations the reservoir is effectively
    worth. A value far below ``len(weights)`` means the distribution has
    collapsed onto a handful of points.
    """
    return 1.0 / max(sum(w * w for w in weights), 1e-12)


def silverman_bandwidth(reservoir, weights, span):
    """
    Silverman's rule-of-thumb KDE bandwidth from a weighted reservoir.

    ``h = 1.06 * sigma * n_eff^(-1/5)``, where ``sigma`` is the weighted
    standard deviation and ``n_eff`` is the Kish effective sample size.
    The result is floored at ``MIN_BANDWIDTH_FRACTION * span`` and capped
    at ``span`` so that jitter is never degenerate and never absurd.

    Args:
        reservoir: Sample values, in the node's internal coordinate space.
        weights:   Corresponding normalised weights (sum to 1).
        span:      ``max - min`` for the node, in internal space.

    Returns:
        Scalar bandwidth ``h > 0``.
    """
    floor = MIN_BANDWIDTH_FRACTION * span if span > 0 else 1e-9

    if len(reservoir) < 2:
        return floor

    mu = weighted_mean(reservoir, weights)
    var = sum(w * (x - mu) ** 2 for w, x in zip(weights, reservoir))
    std = math.sqrt(max(var, 0.0))

    n_eff = effective_sample_size(weights)
    h = 1.06 * std * (n_eff ** -0.2)

    if span > 0:
        return min(max(h, floor), span)
    return max(h, floor)


def reflect(x, lo, hi):
    """
    Fold ``x`` back into ``[lo, hi]`` by reflection.

    Used instead of clipping when applying KDE jitter. Clipping piles
    probability mass onto the interval endpoints, biasing the search toward
    the bounds; reflection preserves the density shape near the edges.
    """
    if hi <= lo:
        return lo
    span = hi - lo
    y = (x - lo) % (2.0 * span)
    if y < 0:
        y += 2.0 * span
    if y > span:
        y = 2.0 * span - y
    return lo + y


def compute_trial_loss(deltas):
    """
    Aggregate per-node deltas from one trial into a scalar.

    Loss measures how far the tree moved on this trial, not how good the
    configuration was. A rolling loss that decays toward zero means the
    distribution has settled.

    Returns:
        Mean of ``deltas``, or 0.0 if no nodes were updated.
    """
    if not deltas:
        return 0.0
    return sum(deltas) / len(deltas)


def is_finite_number(value):
    """True if ``value`` is a real number that is neither NaN nor infinite."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


#: Score tier for the flat uniform grid a continuous node starts with. Any
#: real observation, and any user-supplied prior, outranks it.
GRID_SCORE = -math.inf

#: Score tier for user-supplied prior points. Outranks the flat grid, and is
#: outranked by every finite objective value, so a prior guides early trials
#: and then fades as evidence arrives.
PRIOR_SCORE = -sys.float_info.max

#: Rank-weighting sharpness at ``lambda_ = 1``. The best archive entry is
#: weighted ``exp(sharpness)`` times the worst.
RANK_SHARPNESS = 3.0

#: Ceiling on sharpness, so a large ``lambda_`` cannot collapse the KDE onto
#: a single point.
MAX_SHARPNESS = 10.0


def rank_weights(scores, lambda_):
    """
    Geometric weights over within-archive score rank.

    The best entry receives ``exp(sharpness)`` times the weight of the worst,
    where ``sharpness = min(RANK_SHARPNESS * lambda_, MAX_SHARPNESS)``. Rank
    rather than raw score keeps the KDE insensitive to the objective's scale
    and stops one outlier dominating the archive.

    Tied scores receive the *average* of the ranks they span. Without that,
    a stable sort breaks ties by list position, and the identically scored
    points of a fresh uniform grid come out weighted up to ``exp(sharpness)``
    apart purely by index — a systematic bias toward the top of every range.
    """
    n = len(scores)
    if n == 1:
        return [1.0]

    sharpness = min(RANK_SHARPNESS * lambda_, MAX_SHARPNESS)
    order = sorted(range(n), key=lambda i: scores[i])

    normalised_rank = [0.0] * n
    start = 0
    while start < n:
        stop = start
        while stop + 1 < n and scores[order[stop + 1]] == scores[order[start]]:
            stop += 1
        shared = (start + stop) / 2.0 / (n - 1)
        for position in range(start, stop + 1):
            normalised_rank[order[position]] = shared
        start = stop + 1

    raw = [math.exp(sharpness * (u - 1.0)) for u in normalised_rank]
    total = sum(raw)
    return [w / total for w in raw]
