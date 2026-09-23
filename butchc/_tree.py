"""Probability-tree construction and coordinate transforms.

Continuous parameters are modelled in an *internal* coordinate space and
reported to the user in the *external* space they defined:

    plain    internal == external
    log      internal = log10(external), so the KDE is uniform in decades
    int      internal is continuous, external is the nearest integer

Modelling log-scaled parameters in log space matters: a search over
``lr in [1e-4, 1e-1]`` sampled uniformly puts 99.9% of its mass above 1e-3,
leaving the bottom two decades effectively unreachable.
"""

import math

from ._utils import (
    GRID_SCORE,
    KDE_RESERVOIR_SIZE,
    MIN_GRID_POINTS,
    PRIOR_SCORE,
    rank_weights,
)


def to_internal(node, value):
    """Map an external parameter value into the node's internal space."""
    if node.get("log"):
        return math.log10(value)
    return float(value)


def to_external(node, x):
    """Map an internal coordinate back to the user-facing value."""
    if node.get("log"):
        return 10.0 ** x
    if node.get("int"):
        # Integer nodes model a half-open interval half a unit wider than the
        # declared range (see initialize_prob_tree), so the rounded value can
        # land one outside it. Clamp to what the user asked for.
        value = int(round(x))
        if "ext_min" in node:
            value = min(max(value, math.ceil(node["ext_min"])),
                        math.floor(node["ext_max"]))
        return value
    return x


def snap_internal(node, x):
    """
    Internal coordinate of the value the objective will actually receive.

    For integer nodes the emitted value is rounded, so the tree must learn
    from the rounded value rather than the raw draw; otherwise the model and
    the objective disagree about what was evaluated.
    """
    if node.get("int"):
        return float(round(x))
    return x


def initialize_prob_tree(searchspace):
    """
    Recursively build a probability tree from a validated search space.

    Node forms:
        Categorical  ``{'counts', 'prob', ['next_level']}``
        Continuous   ``{'min', 'max', 'log', 'int', 'reservoir', 'weights'}``
                     where 'min'/'max' are in internal space and
                     'ext_min'/'ext_max' preserve what the user wrote.

    Continuous nodes start as an evenly spaced grid of ``KDE_RESERVOIR_SIZE``
    points at equal weight — a flat, uninformative prior.

    Args:
        searchspace: Validated dict of parameter definitions.

    Returns:
        Nested dict representing the initialised probability tree.
    """
    tree = {}

    for key, spec in searchspace.items():
        if "values" in spec:
            node = _categorical_node(spec)
        else:
            node = {
                "log": bool(spec.get("log", False)),
                "int": bool(spec.get("int", False)),
                "ext_min": spec["min"],
                "ext_max": spec["max"],
            }
            lo = to_internal(node, spec["min"])
            hi = to_internal(node, spec["max"])
            if node["int"]:
                # Rounding maps [k-0.5, k+0.5) to k, so an internal range of
                # exactly [min, max] gives the endpoint integers half-width
                # bins and half the mass of every interior value. Widening by
                # half a unit makes uniform draws uniform over the integers;
                # to_external clamps the result back into range.
                lo -= 0.5
                hi += 0.5
            node["min"] = lo
            node["max"] = hi

            n = KDE_RESERVOIR_SIZE
            if n > 1:
                step = (hi - lo) / (n - 1)
                reservoir = [lo + i * step for i in range(n)]
            else:
                reservoir = [(lo + hi) / 2.0]

            node["reservoir"] = reservoir
            node["scores"] = [GRID_SCORE] * len(reservoir)
            _seed_continuous_prior(node, spec)
            node["weights"] = rank_weights(node["scores"], 1.0)

        tree[key] = node

    return tree


def reservoir_summary(node):
    """
    Human-readable summary of a continuous node.

    Both ``mean`` and ``std`` are reported in the units the user defined, so a
    log-scaled parameter comes back on its original scale rather than in
    log10 units.

    Returns:
        ``{'mean', 'std', 'ess', 'n'}`` — weighted mean and standard
        deviation of the learned distribution in external units, its
        effective sample size, and the reservoir size.
    """
    from ._utils import effective_sample_size, weighted_mean

    weights = node["weights"]
    externals = [to_external(node, x) for x in node["reservoir"]]
    mu = weighted_mean(externals, weights)
    var = sum(w * (x - mu) ** 2 for w, x in zip(weights, externals))

    return {
        "mean": mu,
        "std": math.sqrt(max(var, 0.0)),
        "ess": effective_sample_size(weights),
        "n": len(externals),
    }


DEFAULT_PRIOR_STRENGTH = 10.0


def _categorical_node(spec):
    """
    Build a categorical node, storing any prior alongside the counts.

    ``counts`` and ``visits`` start empty and hold evidence only. The prior
    is kept in its own fields and applied multiplicatively at sampling time
    by ``_apply_prior``, with an exponent that decays as visits accumulate.
    ``prob`` is recomputed from counts on every update, so the value stored
    here is only the starting distribution.
    """
    values = spec["values"]
    prior = spec.get("prior")
    strength = float(spec.get("prior_strength", DEFAULT_PRIOR_STRENGTH))

    counts = {k: 0.0 for k in values}
    visits = {k: 0.0 for k in values}

    if prior is None:
        shares = {k: 1.0 / len(values) for k in values}
        strength = 0.0
        probs = dict(shares)
    else:
        shares = {k: float(prior.get(k, 0.0)) for k in values}
        probs = dict(shares)

    node = {
        "counts": counts,
        "visits": visits,
        "prob": probs,
        "sub_size": 0,
        # Kept apart from `counts`/`visits` because those are discounted every
        # trial: a stated belief should be overturned by contrary evidence, not
        # quietly forgotten because time passed. `prior_strength` is how many
        # real visits it takes to outweigh.
        "prior": shares,
        "prior_strength": strength,
    }
    if spec.get("next_level"):
        node["next_level"] = {
            choice: initialize_prob_tree(sub)
            for choice, sub in spec["next_level"].items()
        }
        node["sub_size"] = max(
            _subspace_size(sub) for sub in spec["next_level"].values()
        )
    return node


def _subspace_size(searchspace):
    """
    Most parameters one configuration drawn from this sub-space can carry.

    Used to slow how fast a node commits: a choice that opens a sub-space
    cannot be judged until that sub-space has been tuned, and how long that
    takes scales with how many parameters are in it.

    Every parameter at this level counts once, because they all appear
    together. A branching parameter adds its *largest* branch on top, because
    only one branch is ever taken. Siblings that both branch therefore both
    contribute — a config drawn from them holds parameters from each. The
    result is independent of dict key order.
    """
    total = 0
    for spec in searchspace.values():
        total += 1
        branches = spec.get("next_level", {})
        if branches:
            total += max(_subspace_size(sub) for sub in branches.values())
    return total


def _seed_continuous_prior(node, spec):
    """
    Replace part of the uniform grid with points drawn from the user's prior.

    Prior points carry a score tier above the grid and below any real
    objective value, so they outrank the flat start, guide early trials, and
    are then displaced by evidence rather than competing with it forever.

    ``prior`` is either an explicit list of believed-good values, or
    ``{'mean': m, 'std': s}``. Values are given in the units the user
    defined; log parameters are converted internally.
    """
    prior = spec.get("prior")
    if prior is None:
        return

    # A prior is a starting point, not a constraint, so MIN_GRID_POINTS of the
    # flat grid always survive and keep the rest of the range reachable, no
    # matter how large prior_strength is.
    capacity = len(node["reservoir"]) - MIN_GRID_POINTS
    n_points = int(min(float(spec.get("prior_strength", DEFAULT_PRIOR_STRENGTH)),
                       capacity))
    if n_points < 1:
        return

    lo, hi = node["min"], node["max"]

    if isinstance(prior, dict):
        centre = to_internal(node, prior["mean"])
        spread = prior["std"]
        if node.get("log"):
            # A std given in external units is meaningless across decades;
            # interpret it as a multiplicative factor.
            spread = math.log10(1.0 + spread / prior["mean"])
        points = []
        for i in range(n_points):
            # Deterministic quantiles of the prior, so seeding does not
            # depend on RNG state and is reproducible.
            q = (i + 0.5) / n_points
            points.append(centre + spread * _inverse_normal_cdf(q))
    else:
        raw = [to_internal(node, v) for v in prior]
        points = [raw[i % len(raw)] for i in range(n_points)]

    points = [snap_internal(node, min(max(p, lo), hi)) for p in points]

    # Thin the grid evenly rather than truncating it, so the surviving points
    # still span the full range wherever the prior happens to sit.
    node["reservoir"] = _thin(node["reservoir"], n_points) + points
    node["scores"] = [GRID_SCORE] * (len(node["reservoir"]) - n_points) \
        + [PRIOR_SCORE] * n_points


def _thin(grid, n_drop):
    """Drop ``n_drop`` points from ``grid``, spreading the loss evenly."""
    keep = len(grid) - n_drop
    if keep <= 0:
        return []
    if keep == 1:
        return [grid[len(grid) // 2]]
    # Evenly spaced indices including both endpoints, so the surviving grid
    # still spans the full range.
    step = (len(grid) - 1) / (keep - 1)
    return [grid[int(round(i * step))] for i in range(keep)]


def _inverse_normal_cdf(q):
    """
    Standard-normal quantile via Acklam's rational approximation.

    Accurate to about 1e-9 across the unit interval, which is far beyond what
    seeding a 50-point reservoir needs, and avoids a scipy dependency.
    """
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    p_low, p_high = 0.02425, 1 - 0.02425

    if q < p_low:
        r = math.sqrt(-2 * math.log(q))
        return (((((c[0]*r+c[1])*r+c[2])*r+c[3])*r+c[4])*r+c[5]) / \
               ((((d[0]*r+d[1])*r+d[2])*r+d[3])*r+1)
    if q > p_high:
        r = math.sqrt(-2 * math.log(1 - q))
        return -(((((c[0]*r+c[1])*r+c[2])*r+c[3])*r+c[4])*r+c[5]) / \
                ((((d[0]*r+d[1])*r+d[2])*r+d[3])*r+1)
    r = q - 0.5
    s = r * r
    return (((((a[0]*s+a[1])*s+a[2])*s+a[3])*s+a[4])*s+a[5])*r / \
           (((((b[0]*s+b[1])*s+b[2])*s+b[3])*s+b[4])*s+1)
