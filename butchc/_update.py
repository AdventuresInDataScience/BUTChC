"""The update rule — where the objective value enters the model.

**Categorical nodes** score each choice by its *mean* rank per visit rather
than by an accumulated total. A mean separates "this branch is good" from
"this branch was tried a lot"; a total does not, because a branch sampled
three times as often accumulates roughly three times the score at equal
quality and so amplifies its own early luck.

Choices are smoothed toward ``NEUTRAL_QUALITY`` with strength ``alpha``
pseudo-visits, and the resulting values are raised to a power that grows over
the run, so a node explores early and commits late.

**Continuous nodes** keep an *elite archive*: the reservoir retains the
best-scoring observations seen for that parameter, and eviction removes the
worst-scoring entry rather than the least-weighted one. Weights decay
geometrically with within-archive rank, so the KDE sits over the good region
and tightens as the archive improves.

Evicting by weight instead cannot work: a fresh point competes against a full
reservoir whose mass is spread over ``KDE_RESERVOIR_SIZE`` entries, so it
either swamps the archive or is itself the minimum-weight entry and is pruned
on arrival.
"""

from ._utils import KDE_RESERVOIR_SIZE, rank_weights, weighted_mean

#: Per-visit discount on categorical statistics. ``1.0`` is a lifetime mean;
#: lower forgets faster. The effective window is ``1 / (1 - DISCOUNT)`` visits.
DISCOUNT = 0.98

#: Score credited to a choice nobody has tried. Categorical choices are scored
#: by mean *rank*, and a trial drawn at random has an expected rank of 0.5.
NEUTRAL_QUALITY = 0.5

def quality_weight(rank, gamma):
    """
    Convert a trial's quantile rank into an update weight in ``[0, 1]``.

    Trials at or below the ``gamma`` quantile contribute nothing; above it the
    weight rises linearly to 1 for the best trial seen. Rank rather than raw
    objective makes the rule invariant to the objective's scale, offset and
    outliers — an objective in the millions behaves like one in ``[0, 1]``.

    Args:
        rank:  Quantile rank of this trial's objective in ``[0, 1]``.
        gamma: Quantile below which trials are ignored, in ``[0, 1)``.

    Returns:
        Update weight in ``[0, 1]``.
    """
    if rank <= gamma:
        return 0.0
    return (rank - gamma) / (1.0 - gamma)


def update_tree(tree, trace, lambda_, alpha, quality, score, rng,
                sharpen=1.0, neutral=None, discount=None, rank=None):
    """
    Update the probability tree in-place from one evaluated trial.

    Call this for *every* trial. Categorical nodes update unconditionally,
    because they are scored by mean rank and a trial that failed the gate is
    still evidence against its branch. Continuous archives move only when
    ``quality > 0``.

    Categorical nodes::

        visits[k]      *= discount          # every choice, every visit
        counts[k]      *= discount
        visits[chosen] += 1
        counts[chosen] += rank
        value[k]        = (counts[k] + alpha * neutral) / (visits[k] + alpha)
        prob[k]        ∝ value[k] ** sharpen  * prior[k] ** (decaying weight)

    Continuous nodes (elite KDE archive)::

        append (value, score) to the reservoir
        if over KDE_RESERVOIR_SIZE: evict the worst-scoring entry
        weights <- geometric decay over within-archive score rank

    Args:
        tree:           Probability tree, mutated in place.
        trace:          Structural trace from ``traverse_sample``.
        lambda_:        Sharpness of the continuous archive's rank weighting
                        (> 0). Continuous nodes only: categorical nodes are
                        means over visits, and scaling their quality by a
                        learning rate would just make later trials look worse
                        than earlier ones.
        alpha:          Smoothing strength for categorical nodes, in
                        pseudo-visits (> 0).
        quality:        Gated, rank-derived weight in ``[0, 1]``. Drives the
                        continuous archive; zero leaves it alone.
        score:          This trial's objective value, used to rank archive
                        entries.
        rng:            ``random.Random`` instance, used to break eviction ties.
        sharpen:        Exponent on categorical values, >= 1. Grows over the
                        run so the node commits late rather than early.
        neutral:        Score an unvisited choice is credited with — the
                        expected midrank of a configuration drawn at random,
                        ``0.5``.
        discount:       Per-visit decay on categorical statistics, in
                        ``(0, 1]``. Below 1 the score becomes a recency-
                        weighted mean; see ``_update_categorical``.
        rank:            Ungated midrank in ``[0, 1]``. Scores categorical
                        choices. Defaults to ``quality`` for callers that do
                        not distinguish them.

    Returns:
        List of normalised per-node deltas in ``[0, 1]``, one per node that
        was updated: every categorical node on the sampled path, plus the
        continuous nodes when the gate passed. A delta of 0 means the node was
        updated but did not move.
    """
    deltas = []
    passing = quality > 0.0
    # Module constants are read at call time, not bound as default arguments,
    # so a sweep can override them by patching the module.
    neutral = NEUTRAL_QUALITY if neutral is None else neutral
    discount = DISCOUNT if discount is None else discount
    rank = quality if rank is None else rank

    for key, entry in trace.items():
        node = tree[key]
        value = entry["internal"]

        if "prob" in node:
            # Categorical nodes move on every trial, so their delta is always
            # recorded. Reporting it only when the gate passed made the loss
            # read zero on trials that had in fact shifted the distribution.
            deltas.append(_update_categorical(node, value, alpha, rank,
                                              sharpen, neutral, discount))
            if entry["sub"] is not None:
                deltas.extend(update_tree(
                    node["next_level"][value],
                    entry["sub"],
                    lambda_,
                    alpha,
                    quality,
                    score,
                    rng,
                    sharpen,
                    neutral,
                    discount,
                    rank,
                ))
        elif passing:
            deltas.append(_update_continuous(node, value, score, lambda_, rng))

    return deltas


def _update_categorical(node, value, alpha, rank, sharpen, neutral, discount):
    """
    Record one visit to ``value`` and the rank it earned.

    Choices are scored by ungated mean *rank*, not by the gated ``quality``
    the continuous archive uses. The gate is self-referential: a trial is
    ranked against the optimiser's own increasingly biased history, so a
    choice that comes to dominate sampling stops clearing the gate it is
    winning and its discounted counts decay away. Mean rank keeps separating
    under the same pressure — a choice taking 90% of the budget and winning
    every time still sits near 0.55 against a loser's 0.05.

    Both statistics are discounted on every trial that reaches this node, so
    the score is a recency-weighted mean over roughly ``1 / (1 - discount)``
    recent visits rather than a lifetime average. These arms are not
    stationary: a branch's reward distribution improves as its own sub-model
    learns, so a branch with a wide or sensitive sub-space scores badly
    exactly while it is still being tuned. A lifetime mean would fix that
    early verdict permanently, and a starved branch stops being visited, so
    the stale estimate would never be revised. Discounting decays an unvisited
    branch's evidence toward ``neutral``, distinguishing "tried and bad" from
    "not tried lately".
    """
    before = dict(node["prob"])
    if discount < 1.0:
        for k in node["counts"]:
            node["counts"][k] *= discount
            node["visits"][k] *= discount
    node["visits"][value] = node["visits"].get(value, 0.0) + 1.0
    node["counts"][value] += rank
    _recompute_prob(node, alpha, sharpen, neutral)
    n_choices = len(node["counts"])
    return sum(abs(node["prob"][k] - before[k]) for k in node["prob"]) / n_choices


def _recompute_prob(node, alpha, sharpen, neutral=None):
    """
    Recompute a categorical node's probabilities from its counts and visits.

    Each choice is scored by mean rank per visit, smoothed toward ``neutral``
    — the score an untried choice is presumed to have — with ``alpha``
    pseudo-visits of weight. Because the score is a midrank, ``neutral`` is
    knowable in advance rather than estimated: a configuration drawn at random
    has expected rank 0.5. A choice scoring above that gains mass, one scoring
    below loses mass, and one nobody has tried sits at the middle rather than
    at zero.
    """
    neutral = NEUTRAL_QUALITY if neutral is None else neutral
    counts, visits = node["counts"], node["visits"]
    keys = list(counts)

    values = {
        k: (counts[k] + alpha * neutral) / (visits.get(k, 0.0) + alpha)
        for k in keys
    }
    # A node whose choices open sub-spaces commits more slowly, in proportion
    # to how much there is to tune underneath. Without this, one exponent has
    # to serve both a flat six-way choice of activation function — where fast
    # commitment is free — and a choice of model family, where committing
    # early means judging a branch on an untuned sub-model.
    # Floored at 1: below that the node ignores what it knows, which is right
    # while a sub-space is still being tuned but wrong for a flat choice, where
    # it would also throw away the user's prior.
    exponent = max(sharpen, 1.0) / (1.0 + node.get("sub_size", 0))
    if exponent != 1.0:
        values = {k: max(v, 0.0) ** exponent for k, v in values.items()}

    values = _apply_prior(node, values, keys)

    total = sum(values.values())
    if total <= 0.0:
        for k in keys:
            node["prob"][k] = 1.0 / len(keys)
        return
    for k in keys:
        node["prob"][k] = values[k] / total


def _apply_prior(node, values, keys):
    """
    Bias the values by the user's stated prior, fading as evidence arrives.

    The prior multiplies rather than contributing pseudo-evidence, and its
    exponent decays as ``strength / (strength + visits)``: with no trials yet
    the probabilities are exactly the stated prior, and every visit dilutes it.

    Folding the prior into the counts instead would not work, because belief
    and evidence are not on the same scale here. The score is a rank, so a
    choice that dominates sampling is increasingly ranked against itself and
    its mean decays back toward neutral however much better it truly is. A
    prior pinned to a fixed score could therefore sit permanently above
    anything evidence can reach. A multiplicative bias that decays with visits
    has no such ceiling.
    """
    strength = node.get("prior_strength", 0.0)
    if strength <= 0.0:
        return values

    shares = node.get("prior") or {}
    seen = sum(node["visits"].get(k, 0.0) for k in keys)
    weight = strength / (strength + seen)
    return {
        k: values[k] * max(shares.get(k, 0.0), 1e-12) ** weight
        for k in keys
    }


def _update_continuous(node, value, score, lambda_, rng):
    reservoir, scores = node["reservoir"], node["scores"]
    span = node["max"] - node["min"]
    mean_before = weighted_mean(reservoir, node["weights"])

    reservoir.append(value)
    scores.append(score)

    if len(reservoir) > KDE_RESERVOIR_SIZE:
        _evict_worst(reservoir, scores, rng)

    node["weights"] = rank_weights(scores, lambda_)

    mean_after = weighted_mean(reservoir, node["weights"])
    return abs(mean_after - mean_before) / span if span > 0 else 0.0


def _evict_worst(reservoir, scores, rng):
    """
    Remove the worst-scoring archive entry, breaking ties uniformly.

    Ties matter at start-up: the initial grid carries a sentinel score of
    ``-inf``, and always evicting the lowest index would erode the grid from
    one end, skewing early coverage toward the upper half of the range.
    """
    worst = min(scores)
    candidates = [i for i, s in enumerate(scores) if s == worst]
    idx = candidates[0] if len(candidates) == 1 else rng.choice(candidates)
    reservoir.pop(idx)
    scores.pop(idx)
