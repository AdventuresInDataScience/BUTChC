"""Up-front validation of search spaces, hyperparameters and warm-start trees.

Everything detectable before the first objective evaluation is caught here,
with a message naming the offending parameter path, so a malformed definition
costs nothing when trials are expensive.
"""

import math

_CATEGORICAL_KEYS = {"values", "next_level", "prior", "prior_strength"}
_CONTINUOUS_KEYS = {"min", "max", "log", "int", "prior", "prior_strength"}


class SearchSpaceError(ValueError):
    """Raised when a search-space definition is malformed."""


def _path(prefix, key):
    return f"{prefix}.{key}" if prefix else key


def integer_span(lo, hi):
    """Count of integers reachable in ``[lo, hi]`` after rounding to nearest."""
    return math.floor(hi + 0.5) - math.ceil(lo - 0.5) + 1


def validate_searchspace(searchspace, _prefix="", _ancestors=frozenset()):
    """
    Recursively validate a search-space definition.

    Sampled configurations are flattened into one dict, so two parameters that
    can appear in the same config and share a name would collide. Names may
    repeat across sibling branches of one ``next_level``, since those never
    co-occur; anything else is a hard error.

    Returns:
        Set of every parameter name reachable in this (sub-)space.

    Raises:
        SearchSpaceError: With a message identifying the offending path.
    """
    if not isinstance(searchspace, dict):
        raise SearchSpaceError(
            f"{_prefix or 'searchspace'}: must be a dict, "
            f"got {type(searchspace).__name__}"
        )
    if not searchspace:
        raise SearchSpaceError(f"{_prefix or 'searchspace'}: is empty")

    reachable = set()

    for key, spec in searchspace.items():
        here = _path(_prefix, key)

        if not isinstance(key, str):
            raise SearchSpaceError(f"parameter name {key!r} at {here} must be a string")
        if not isinstance(spec, dict):
            raise SearchSpaceError(
                f"{here}: definition must be a dict, got {type(spec).__name__}"
            )
        if key in _ancestors:
            raise SearchSpaceError(
                f"{here}: name '{key}' is already used by an ancestor node. "
                f"Sampled configs are flattened into one dict, so this would "
                f"silently overwrite the parent's value. Rename one of them."
            )

        is_categorical = "values" in spec
        is_continuous = "min" in spec or "max" in spec

        if is_categorical and is_continuous:
            raise SearchSpaceError(
                f"{here}: cannot define both 'values' and 'min'/'max'"
            )
        if not is_categorical and not is_continuous:
            raise SearchSpaceError(
                f"{here}: needs either 'values' (categorical) or 'min' and 'max' "
                f"(continuous). Got keys: {sorted(map(str, spec))}"
            )

        if is_categorical:
            names = _validate_categorical(here, key, spec, _ancestors)
        else:
            _validate_continuous(here, spec)
            names = {key}

        clash = names & reachable
        if clash:
            raise SearchSpaceError(
                f"{here}: name(s) {sorted(clash)} also appear elsewhere at this "
                f"level of the search space. Flattened configs cannot hold two "
                f"parameters with the same name. Rename one of them."
            )
        reachable |= names

    return reachable


def _validate_categorical(here, key, spec, ancestors):
    unknown = set(map(str, spec)) - _CATEGORICAL_KEYS
    if unknown:
        raise SearchSpaceError(
            f"{here}: unexpected keys {sorted(unknown)} on a categorical node. "
            f"Allowed: {sorted(_CATEGORICAL_KEYS)}"
        )

    values = spec["values"]
    if not isinstance(values, (list, tuple)):
        raise SearchSpaceError(
            f"{here}: 'values' must be a list, got {type(values).__name__}"
        )
    if len(values) == 0:
        raise SearchSpaceError(f"{here}: 'values' is empty")

    try:
        unique = set(values)
    except TypeError as exc:
        raise SearchSpaceError(f"{here}: all 'values' must be hashable ({exc})") from exc
    if len(unique) != len(values):
        raise SearchSpaceError(f"{here}: 'values' contains duplicates")

    _validate_prior_strength(here, spec)
    prior = spec.get("prior")
    if prior is not None:
        if not isinstance(prior, dict):
            raise SearchSpaceError(
                f"{here}: categorical 'prior' must be a dict of "
                f"{{choice: probability}}, got {type(prior).__name__}"
            )
        stray = set(prior) - unique
        if stray:
            raise SearchSpaceError(
                f"{here}: 'prior' has keys {sorted(map(str, stray))} that are "
                f"not listed in 'values'"
            )
        for choice, p in prior.items():
            if isinstance(p, bool) or not isinstance(p, (int, float)) or p < 0:
                raise SearchSpaceError(
                    f"{here}: prior for {choice!r} must be a non-negative "
                    f"number, got {p!r}"
                )
        total = sum(prior.values())
        if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-9):
            raise SearchSpaceError(
                f"{here}: 'prior' probabilities must sum to 1, got {total}"
            )

    names = {key}
    next_level = spec.get("next_level")
    if next_level is not None:
        if not isinstance(next_level, dict):
            raise SearchSpaceError(
                f"{here}: 'next_level' must be a dict, got {type(next_level).__name__}"
            )
        stray = set(next_level) - unique
        if stray:
            raise SearchSpaceError(
                f"{here}: 'next_level' has keys {sorted(map(str, stray))} that are "
                f"not listed in 'values'"
            )
        for choice, sub in next_level.items():
            names |= validate_searchspace(
                sub,
                _prefix=f"{here}[{choice!r}]",
                _ancestors=ancestors | {key},
            )
    return names


def _validate_continuous(here, spec):
    unknown = set(map(str, spec)) - _CONTINUOUS_KEYS
    if unknown:
        raise SearchSpaceError(
            f"{here}: unexpected keys {sorted(unknown)} on a continuous node. "
            f"Allowed: {sorted(_CONTINUOUS_KEYS)}"
        )
    if "min" not in spec or "max" not in spec:
        raise SearchSpaceError(f"{here}: continuous nodes need both 'min' and 'max'")

    lo, hi = spec["min"], spec["max"]
    for name, v in (("min", lo), ("max", hi)):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise SearchSpaceError(f"{here}: '{name}' must be a finite number, got {v!r}")
    if not lo < hi:
        raise SearchSpaceError(f"{here}: requires min < max, got min={lo}, max={hi}")

    if spec.get("log") and spec.get("int"):
        raise SearchSpaceError(f"{here}: 'int' and 'log' cannot be combined")
    if spec.get("log") and lo <= 0:
        raise SearchSpaceError(
            f"{here}: log-scaled parameters need min > 0, got min={lo}"
        )
    if spec.get("int") and integer_span(lo, hi) < 2:
        raise SearchSpaceError(
            f"{here}: integer parameter spans fewer than 2 integers "
            f"(min={lo}, max={hi})"
        )

    _validate_prior_strength(here, spec)
    _validate_continuous_prior(here, spec, lo, hi)


def _validate_prior_strength(here, spec):
    if "prior_strength" in spec and "prior" not in spec:
        raise SearchSpaceError(
            f"{here}: 'prior_strength' given without a 'prior'"
        )
    strength = spec.get("prior_strength")
    if strength is None:
        return
    if isinstance(strength, bool) or not isinstance(strength, (int, float)):
        raise SearchSpaceError(
            f"{here}: 'prior_strength' must be a number, got {strength!r}"
        )
    if strength <= 0:
        raise SearchSpaceError(
            f"{here}: 'prior_strength' must be > 0, got {strength}"
        )


def _validate_continuous_prior(here, spec, lo, hi):
    prior = spec.get("prior")
    if prior is None:
        return

    if isinstance(prior, dict):
        missing = {"mean", "std"} - set(prior)
        if missing:
            raise SearchSpaceError(
                f"{here}: continuous 'prior' dict needs {sorted(missing)}"
            )
        extra = set(map(str, prior)) - {"mean", "std"}
        if extra:
            raise SearchSpaceError(
                f"{here}: unexpected keys {sorted(extra)} in 'prior'. "
                f"Use {{'mean': ..., 'std': ...}} or a list of values."
            )
        mean, std = prior["mean"], prior["std"]
        for name, v in (("mean", mean), ("std", std)):
            if isinstance(v, bool) or not isinstance(v, (int, float)) \
                    or not math.isfinite(v):
                raise SearchSpaceError(
                    f"{here}: prior '{name}' must be a finite number, got {v!r}"
                )
        if std <= 0:
            raise SearchSpaceError(f"{here}: prior 'std' must be > 0, got {std}")
        if not spec["min"] <= mean <= spec["max"]:
            raise SearchSpaceError(
                f"{here}: prior 'mean' {mean} is outside [min, max] "
                f"[{spec['min']}, {spec['max']}]"
            )
        return

    if not isinstance(prior, (list, tuple)):
        raise SearchSpaceError(
            f"{here}: continuous 'prior' must be a list of values or "
            f"{{'mean': ..., 'std': ...}}, got {type(prior).__name__}"
        )
    if not prior:
        raise SearchSpaceError(f"{here}: 'prior' list is empty")
    for v in prior:
        if isinstance(v, bool) or not isinstance(v, (int, float)) \
                or not math.isfinite(v):
            raise SearchSpaceError(
                f"{here}: prior values must be finite numbers, got {v!r}"
            )
        if not spec["min"] <= v <= spec["max"]:
            raise SearchSpaceError(
                f"{here}: prior value {v} is outside [min, max] "
                f"[{spec['min']}, {spec['max']}]"
            )


def _is_integer(value):
    """True for ints and integer-like scalars such as ``numpy.int64``."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    try:
        value.__index__()
    except (AttributeError, TypeError):
        return False
    return True


def validate_hyperparameters(budget, lambda_, alpha, temp, gamma, explore, n_warmup,
                             batch=1):
    """Validate the optimiser's scalar arguments, failing before trial 1."""
    checks = (
        ("budget", budget, lambda v: _is_integer(v) and v >= 1, "must be an int >= 1"),
        ("lambda_", lambda_, lambda v: isinstance(v, (int, float)) and v > 0,
         "must be a number > 0"),
        ("alpha", alpha, lambda v: isinstance(v, (int, float)) and v > 0,
         "must be a number > 0"),
        ("temp", temp, lambda v: isinstance(v, (int, float)) and v > 0,
         "must be a number > 0"),
        ("gamma", gamma, lambda v: isinstance(v, (int, float)) and 0.0 <= v < 1.0,
         "must be a number in [0, 1)"),
        ("explore", explore, lambda v: isinstance(v, (int, float)) and 0.0 <= v <= 1.0,
         "must be a number in [0, 1]"),
        ("n_warmup", n_warmup, lambda v: _is_integer(v) and v >= 0,
         "must be an int >= 0"),
        ("batch", batch, lambda v: _is_integer(v) and (v >= 1 or v == -1),
         "must be an int >= 1, or -1 for one config per CPU"),
    )
    for name, value, ok, why in checks:
        if isinstance(value, bool) or not ok(value):
            raise ValueError(f"{name} {why}, got {value!r}")


def validate_prob_tree(tree, _prefix=""):
    """
    Structurally validate a warm-start probability tree.

    Raises:
        ValueError: If the tree is not a usable BUTChC probability tree.
    """
    label = f"start_prob_tree{'.' + _prefix if _prefix else ''}"
    if not isinstance(tree, dict) or not tree:
        raise ValueError(
            f"{label}: expected a non-empty dict from a previous run's 'prob_tree'"
        )

    for key, node in tree.items():
        here = _path(_prefix, key)
        if not isinstance(node, dict):
            raise ValueError(f"start_prob_tree at {here}: node must be a dict")

        if "prob" in node:
            for field in ("counts", "visits", "prior", "prior_strength"):
                if field not in node:
                    raise ValueError(
                        f"start_prob_tree at {here}: categorical node lacks "
                        f"{field!r}. Trees from before 0.4 do not carry visit "
                        f"counts and cannot be warm started; re-run instead."
                    )
            if set(node["prob"]) != set(node["counts"]):
                raise ValueError(
                    f"start_prob_tree at {here}: 'prob' and 'counts' keys disagree"
                )
            for choice, sub in node.get("next_level", {}).items():
                if choice not in node["prob"]:
                    raise ValueError(
                        f"start_prob_tree at {here}: next_level key {choice!r} is "
                        f"not a known choice"
                    )
                validate_prob_tree(sub, _prefix=f"{here}[{choice!r}]")

        elif "reservoir" in node:
            for field in ("weights", "scores", "min", "max"):
                if field not in node:
                    raise ValueError(
                        f"start_prob_tree at {here}: continuous node lacks {field!r}"
                    )
            if not node["reservoir"]:
                raise ValueError(f"start_prob_tree at {here}: reservoir is empty")
            if len(node["reservoir"]) != len(node["weights"]):
                raise ValueError(
                    f"start_prob_tree at {here}: reservoir and weights differ in length"
                )

        else:
            raise ValueError(
                f"start_prob_tree at {here}: node has neither 'prob' nor "
                f"'reservoir'. Got keys: {sorted(map(str, node))}"
            )


def check_tree_matches_searchspace(tree, searchspace, _prefix=""):
    """
    Verify a warm-start tree describes the same parameters as ``searchspace``.

    A tree trained on a different space is rejected rather than reinterpreted
    against the new one.
    """
    tree_keys = set(tree)
    space_keys = set(searchspace)
    if tree_keys != space_keys:
        missing = sorted(space_keys - tree_keys)
        extra = sorted(tree_keys - space_keys)
        raise ValueError(
            f"start_prob_tree does not match searchspace at "
            f"{_prefix or '<root>'}: missing {missing}, unexpected {extra}"
        )

    for key, spec in searchspace.items():
        node = tree[key]
        here = _path(_prefix, key)
        if "values" in spec:
            if "prob" not in node:
                raise ValueError(
                    f"start_prob_tree at {here}: expected a categorical node"
                )
            if set(node["prob"]) != set(spec["values"]):
                raise ValueError(
                    f"start_prob_tree at {here}: choices differ from searchspace"
                )
            for choice, sub_space in spec.get("next_level", {}).items():
                sub_tree = node.get("next_level", {}).get(choice)
                if sub_tree is None:
                    raise ValueError(
                        f"start_prob_tree at {here}: missing next_level for {choice!r}"
                    )
                check_tree_matches_searchspace(
                    sub_tree, sub_space, _prefix=f"{here}[{choice!r}]"
                )
        else:
            if "reservoir" not in node:
                raise ValueError(
                    f"start_prob_tree at {here}: expected a continuous node"
                )
            _check_continuous_matches(here, node, spec)


def _check_continuous_matches(here, node, spec):
    """
    Verify a warm-started continuous node describes the same interval.

    Names alone are not enough: a tree trained on ``[0, 1]`` carries archived
    points that all sit outside a space of ``[100, 200]``, so bounds and
    scale flags must match too.
    """
    for flag in ("log", "int"):
        if bool(node.get(flag)) != bool(spec.get(flag)):
            raise ValueError(
                f"start_prob_tree at {here}: '{flag}' is "
                f"{bool(node.get(flag))} in the tree but {bool(spec.get(flag))} "
                f"in the searchspace"
            )
    for field, want in (("ext_min", spec["min"]), ("ext_max", spec["max"])):
        have = node.get(field)
        if have is None or not math.isclose(
            float(have), float(want), rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError(
                f"start_prob_tree at {here}: bounds differ from the "
                f"searchspace ({field}={have!r}, expected {want!r}). Warm "
                f"starting only works on the space the tree was trained on."
            )
