"""Reduce a search space using what a finished run learned about it.

The use case is a handoff. BUTChC's clearest margins on conditional problems
come from deciding *which branch* — `Branch trap` and `Categorical mix` — while
on `Nested pipeline`, where a branch must be tuned well once chosen, it only
ties TPE. Spending a fraction of the budget on BUTChC to throw away the
branches that do not matter, then handing the survivors to a second run or to
another optimiser, plays to that split.

The output is an ordinary search space: the same dicts BUTChC accepts, so it
can be fed back into `BUTChC_optimize`, or through `butchc.interop` into
ConfigSpace, Optuna, SMAC or anything else. Pass `return_tree=True` to also get
a probability tree consistent with that smaller space, for warm-starting a
follow-up BUTChC run — see `return_tree` below for why the tree from the
original run cannot be reused directly.

What this is not
----------------

It is not a confidence statement. A branch's probability says where the model
would spend the next trial, which is not the same as a claim that the branch
contains no good configurations. `Branch trap` exists in the benchmark suite
precisely because a branch can have a poor average and an excellent best, and
that is the shape most likely to be destroyed by careless pruning. Three
defaults exist to make that harder:

- `protect_best` keeps whichever branch produced the best configuration seen,
  whatever its probability. Deleting the branch holding your best result is
  the one unrecoverable mistake here, so it is on by default.
- `keep_min` keeps at least two choices per node, so a node cannot collapse to
  a single option on the evidence of a short run.
- `threshold` is a fraction of uniform, so a choice must be well behind the
  even split before it is removed rather than merely behind.

Prune early in a run and none of that saves you: probabilities move a great
deal in the first trials, and the commitment exponent has barely ramped. Give
BUTChC enough budget for the tree to mean something — as a rule of thumb the
same budget you would consider too small to trust the answer from.
"""

import math

from ._tree import initialize_prob_tree, reservoir_summary, to_internal, _subspace_size
from ._utils import GRID_SCORE
from ._validate import (
    SearchSpaceError,
    check_tree_matches_searchspace,
    validate_searchspace,
)

#: A choice is a candidate for removal below this fraction of the uniform
#: probability for its node — 1/3 of uniform is 0.083 on a four-way choice and
#: 0.017 on a twenty-way one. Expressed relative to uniform rather than as a
#: bare probability because an absolute cut means different things at different
#: arities: 0.05 is near-abandonment on a three-way node and above average on a
#: thirty-way one. The value itself is a judgement call from a handful of
#: examples, not a tuned constant; `prune_report` exists so you can see what it
#: did before trusting it.
DEFAULT_THRESHOLD = 1.0 / 3.0

#: Choices kept per categorical node regardless of probability.
DEFAULT_KEEP_MIN = 2


def prune(searchspace, result, threshold=DEFAULT_THRESHOLD,
          keep_min=DEFAULT_KEEP_MIN, protect_best=True, narrow=None,
          return_tree=False):
    """Return a smaller search space, keeping the branches worth searching.

    Args:
        searchspace:  The space the run used, in BUTChC's format.
        result:       The dict returned by `BUTChC_optimize`. Its `prob_tree`
                      supplies the probabilities and `best_params` the branch
                      to protect.
        threshold:    Fraction of uniform probability below which a choice is
                      dropped. On a four-way node uniform is 0.25, so the
                      default of 1/3 drops anything at or below 0.083.
                      `0.0` drops nothing; `1.0` drops everything below
                      average.
        keep_min:     Minimum choices to keep per categorical node, taken in
                      probability order. Raise it to prune more timidly.
        protect_best: Keep the choices taken by `result['best_params']` even
                      where probability says otherwise.
        narrow:       If set, also narrow continuous bounds to
                      `mean ± narrow * std` of the learned distribution,
                      clipped to the original bounds. `2.0` is a reasonable
                      starting point. Off by default, because unlike a dropped
                      branch a narrowed bound can exclude the optimum without
                      leaving any trace that it did.
        return_tree:  If True, also return a probability tree consistent with
                      the pruned space, suitable as `start_prob_tree` for a
                      follow-up run. `result['prob_tree']` itself is *not*
                      usable there: it still carries every dropped choice and
                      the original, wider continuous bounds, and
                      `BUTChC_optimize` requires an exact match between a
                      warm-start tree and the space it is given. The returned tree
                      keeps the learned statistics for every surviving choice
                      and archive point, so the follow-up run resumes rather
                      than restarting; dropped choices and narrowed-away
                      archive points are simply not there to resume from.

    Returns:
        The pruned search space. The input is not modified. If `return_tree`
        is set, a `(searchspace, prob_tree)` tuple instead.

    Raises:
        SearchSpaceError: If `result` is not a `BUTChC_optimize` result, if
            `keep_min < 1`, or if `narrow <= 0`.

    Example:
        >>> first = BUTChC_optimize(space, objective, 200, seed=0)
        >>> smaller, tree = prune(space, first, return_tree=True)
        >>> second = BUTChC_optimize(smaller, objective, 200, seed=1,
        ...                          start_prob_tree=tree)
    """
    if not isinstance(result, dict) or "prob_tree" not in result:
        raise SearchSpaceError(
            "result must be the dict returned by BUTChC_optimize; "
            "got something without a 'prob_tree' key"
        )
    if keep_min < 1:
        raise SearchSpaceError(f"keep_min must be at least 1, got {keep_min}")
    if narrow is not None and narrow <= 0:
        raise SearchSpaceError(f"narrow must be positive, got {narrow}")

    best = result.get("best_params") or {} if protect_best else {}
    pruned, pruned_tree = _prune_level(searchspace, result["prob_tree"], best,
                                       threshold, keep_min, narrow)
    validate_searchspace(pruned)
    if not return_tree:
        return pruned

    # A self-check, not a formality: if the tree-construction logic above ever
    # drifts out of sync with the space-construction logic, this turns that
    # into an immediate, loud error here rather than a confusing one deep
    # inside a follow-up BUTChC_optimize call that a caller would have no
    # reason to connect back to prune().
    check_tree_matches_searchspace(pruned_tree, pruned)
    return pruned, pruned_tree


def _prune_level(space, tree, best, threshold, keep_min, narrow):
    out_space, out_tree = {}, {}
    for key, spec in space.items():
        node = tree.get(key)
        if node is None:
            # A parameter the tree never saw: keep it untouched rather than
            # silently dropping something the objective may require. There is
            # no learned data to carry forward either, so the matching tree
            # node is simply a fresh one for this parameter alone.
            out_space[key] = _copy_spec(spec)
            out_tree[key] = initialize_prob_tree({key: spec})[key]
            continue

        if "values" in spec:
            out_space[key], out_tree[key] = _prune_categorical(
                key, spec, node, best, threshold, keep_min, narrow)
        else:
            out_space[key], out_tree[key] = _narrow_continuous(spec, node, narrow)
    return out_space, out_tree


def _prune_categorical(key, spec, node, best, threshold, keep_min, narrow):
    prob = node.get("prob", {})
    values = [v for v in spec["values"] if v in prob] or list(spec["values"])
    ranked = sorted(values, key=lambda v: prob.get(v, 0.0), reverse=True)

    uniform = 1.0 / len(spec["values"]) if spec["values"] else 0.0
    cut = threshold * uniform
    keep = {v for v in ranked if prob.get(v, 0.0) > cut}
    keep |= set(ranked[:max(keep_min, 1)])
    if best.get(key) in spec["values"]:
        keep.add(best[key])

    kept = [v for v in spec["values"] if v in keep]
    if not kept:  # pragma: no cover - keep_min >= 1 makes this unreachable
        raise SearchSpaceError(f"{key}: pruning removed every choice")

    out_spec = _copy_spec(spec)
    out_spec["values"] = kept

    # A prior over dropped choices no longer sums to anything meaningful, and
    # renormalising it would invent a belief the run did not express. Warm
    # starting instead carries the learned distribution across explicitly —
    # see `prune`'s `return_tree`.
    out_spec.pop("prior", None)
    out_spec.pop("prior_strength", None)

    sub_spaces = spec.get("next_level")
    out_next_level = {}
    if sub_spaces:
        sub_trees = node.get("next_level", {})
        next_level = {}
        for choice in kept:
            if choice not in sub_spaces:
                continue
            sub_space, sub_tree = _prune_level(
                sub_spaces[choice], sub_trees.get(choice, {}),
                best, threshold, keep_min, narrow)
            next_level[choice] = sub_space
            out_next_level[choice] = sub_tree
        if next_level:
            out_spec["next_level"] = next_level

    out_node = _pruned_categorical_node(node, kept, out_spec.get("next_level"),
                                        out_next_level)
    return out_spec, out_node


def _pruned_categorical_node(node, kept, pruned_next_level, next_level_trees):
    """Build a categorical tree node restricted to `kept`, renormalised.

    `counts` and `visits` are the learned evidence for the choices that
    survive, carried across unchanged — that evidence is the entire point of
    warm starting rather than restarting. `prob` is renormalised over the
    survivors so it still sums to 1; nothing here re-derives it from `counts`
    via `_recompute_prob`, because the surviving choices' relative standing is
    exactly what pruning decided to trust.
    """
    counts = {k: node.get("counts", {}).get(k, 0.0) for k in kept}
    visits = {k: node.get("visits", {}).get(k, 0.0) for k in kept}
    raw_prob = {k: node.get("prob", {}).get(k, 0.0) for k in kept}
    total = sum(raw_prob.values())
    prob = ({k: v / total for k, v in raw_prob.items()} if total > 0
            else {k: 1.0 / len(kept) for k in kept})

    out_node = {
        "counts": counts,
        "visits": visits,
        "prob": prob,
        # Recomputed from the *pruned* next_level, not copied from the
        # original node: sub_size governs how slowly this node commits, in
        # proportion to how much sub-space its choices still open, and a
        # dropped branch can change that. Carrying the old, larger value
        # across would commit slower than the pruned space actually needs to.
        "sub_size": (max((_subspace_size(sub) for sub in pruned_next_level.values()),
                        default=0) if pruned_next_level else 0),
        # Kept apart from counts/visits for the same reason a fresh tree keeps
        # them apart (see _tree.py): strength 0 makes _apply_prior a no-op, so
        # the node is scored on carried-over evidence alone, not on a prior
        # the pruned space no longer states.
        "prior": dict(prob),
        "prior_strength": 0.0,
    }
    if next_level_trees:
        out_node["next_level"] = next_level_trees
    return out_node


def _narrow_continuous(spec, node, narrow):
    out = _copy_spec(spec)
    if narrow is None or "reservoir" not in node or not node["reservoir"]:
        return out, _copied_continuous_node(node)

    summary = reservoir_summary(node)
    lo = max(spec["min"], summary["mean"] - narrow * summary["std"])
    hi = min(spec["max"], summary["mean"] + narrow * summary["std"])

    if spec.get("int"):
        # Widen to whole numbers so the range still holds integers, then clamp
        # again: rounding outwards can otherwise push a bound past the original
        # one, which would hand the next optimiser a space the user never
        # authorised. A range too tight to contain two integers is left alone.
        lo = max(math.floor(lo), math.ceil(spec["min"]))
        hi = min(math.ceil(hi), math.floor(spec["max"]))
        if hi - lo < 1:
            return out, _copied_continuous_node(node)
    if spec.get("log"):
        lo = max(lo, 1e-300)

    # A collapsed or inverted range means the archive concentrated below the
    # resolution of the bounds. Keeping the original is the safe reading:
    # there is nothing left to narrow towards.
    if not lo < hi:
        return out, _copied_continuous_node(node)

    out["min"], out["max"] = lo, hi
    return out, _narrowed_continuous_node(node, lo, hi)


def _copied_continuous_node(node):
    """Deep-copy a continuous tree node's mutable fields.

    Used wherever `_narrow_continuous` leaves the space untouched (`narrow`
    unset, or the computed range collapsed): the original node already
    matches an untouched space exactly, so nothing about it needs to change.
    """
    out = dict(node)
    out["reservoir"] = list(node["reservoir"])
    out["scores"] = list(node["scores"])
    out["weights"] = list(node["weights"])
    return out


def _narrowed_continuous_node(node, lo_ext, hi_ext):
    """Build a continuous tree node matching a narrowed `[lo_ext, hi_ext]`.

    Archive points outside the narrowed range are dropped and the remaining
    weights renormalised — conditioning the learned distribution on the
    region that survived is what warm starting into a narrower space should
    mean, rather than inventing a new bandwidth for the discarded mass.
    """
    out_node = dict(node)
    out_node["ext_min"], out_node["ext_max"] = lo_ext, hi_ext

    lo_int, hi_int = to_internal(node, lo_ext), to_internal(node, hi_ext)
    if node.get("int"):
        # Mirrors the half-unit widening initialize_prob_tree applies: an
        # internal range of exactly [lo, hi] gives the endpoint integers half
        # the mass of an interior one under rounding.
        lo_int -= 0.5
        hi_int += 0.5
    out_node["min"], out_node["max"] = lo_int, hi_int

    survivors = [(r, s, w) for r, s, w in
                zip(node["reservoir"], node["scores"], node["weights"])
                if lo_int <= r <= hi_int]
    if not survivors:
        # A heavy-tailed or multimodal archive can leave nothing inside even
        # a moderate narrowing. Reseeding one point at the centre keeps the
        # node valid rather than handing back an empty, unusable reservoir;
        # the next run's own trials repopulate it from there.
        centre = (lo_int + hi_int) / 2.0
        survivors = [(centre, GRID_SCORE, 1.0)]

    reservoir, scores, weights = (list(v) for v in zip(*survivors))
    total_w = sum(weights)
    out_node["reservoir"] = reservoir
    out_node["scores"] = scores
    out_node["weights"] = ([w / total_w for w in weights] if total_w > 0
                           else [1.0 / len(weights)] * len(weights))
    return out_node


def _copy_spec(spec):
    """Deep-copy a spec node without importing copy for the nested case."""
    out = {}
    for key, value in spec.items():
        if key == "next_level":
            out[key] = {c: {k: _copy_spec(v) for k, v in sub.items()}
                        for c, sub in value.items()}
        elif isinstance(value, dict):
            out[key] = dict(value)
        elif isinstance(value, list):
            out[key] = list(value)
        else:
            out[key] = value
    return out


def prune_report(searchspace, pruned):
    """Describe what `prune` removed, for printing before you trust it.

    Returns:
        A list of human-readable lines. Empty when nothing changed.
    """
    lines = []
    _report_level(searchspace, pruned, "", lines)
    return lines


def _report_level(before, after, prefix, lines):
    for key, spec in before.items():
        name = prefix + key
        new = after.get(key)
        if new is None:
            lines.append(f"{name}: removed")
            continue
        if "values" in spec:
            dropped = [v for v in spec["values"] if v not in new["values"]]
            if dropped:
                lines.append(f"{name}: dropped {dropped}, kept {new['values']}")
            for choice, sub in spec.get("next_level", {}).items():
                if choice in new.get("next_level", {}):
                    _report_level(sub, new["next_level"][choice],
                                  f"{name}={choice}:", lines)
        elif (spec.get("min"), spec.get("max")) != (new.get("min"), new.get("max")):
            lines.append(
                f"{name}: narrowed [{spec['min']:.6g}, {spec['max']:.6g}] "
                f"-> [{new['min']:.6g}, {new['max']:.6g}]"
            )
