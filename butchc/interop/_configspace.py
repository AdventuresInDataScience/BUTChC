"""Conversion between ``ConfigSpace.ConfigurationSpace`` and BUTChC dicts.

## ConfigSpace to BUTChC

What converts:

- ``CategoricalHyperparameter``    -> ``{'values': [...]}``
- ``OrdinalHyperparameter``        -> ``{'values': [...]}`` (order is not modelled)
- ``UniformFloatHyperparameter``   -> ``{'min', 'max', 'log'}``
- ``UniformIntegerHyperparameter`` -> ``{'min', 'max', 'int'}``
- ``Constant`` / ``UnParametrizedHyperparameter`` -> not searched; returned in
  ``fixed`` and merged into the config at evaluation time
- ``EqualsCondition(child, parent, v)``  -> child under ``parent.next_level[v]``
- ``InCondition(child, parent, [v, w])`` -> child under both branches

What does not, and why:

- **Conjunctions** (``AndConjunction``, ``OrConjunction``). A child gated on two
  independent parents has no single place to live in a tree.
- **Multiple parents** for one child, for the same reason.
- **Forbidden clauses.** BUTChC cannot express "this combination is invalid",
  so it would sample them and waste budget.
- **Log-scaled integers.** BUTChC rejects ``log`` and ``int`` together. These
  become a log-scaled *continuous* parameter plus an entry in ``casts``, which
  ``wrap_objective`` applies. The tree learns in log space and the objective
  still receives an int — the behaviour you want, and what ConfigSpace does
  internally.

Refusals raise ``UnsupportedSpace`` naming the parameter, so a benchmark that
cannot be represented fails loudly at setup rather than silently optimising a
different problem.

## BUTChC to ConfigSpace

Never refuses: every ``next_level`` branch becomes an ``EqualsCondition`` per
value, which is exactly the sub-DAG a tree describes. One reconciliation is
needed — BUTChC allows a name to repeat across sibling branches, ConfigSpace
requires global uniqueness — so colliding names are qualified and the mapping
is returned. Priors are dropped, since ConfigSpace's prior support does not
line up with BUTChC's pseudo-trial semantics and a silently reinterpreted prior
is worse than an absent one.
"""


class UnsupportedSpace(Exception):
    """Raised when a ConfigSpace object has no faithful tree representation."""


# --------------------------------------------------------------- accessors
# ConfigSpace 1.x renamed the collection accessors (``get_hyperparameters()``
# became iteration over ``space.values()``). Both spellings are in the wild, so
# every accessor tries the 1.x name first and falls back to the 0.x one.
#
# The 1.x spelling is tried first deliberately. On 1.x the 0.x accessors still
# exist and merely emit a DeprecationWarning, so probing for them first means
# the fallback never fires and every conversion warns — noise attributed to the
# caller's code, for a branch that was never taken. Ordering it this way also
# fails in the safe direction: when 2.x removes the old names entirely, nothing
# here has to change.

def _hyperparameters(space):
    if hasattr(space, "values"):
        return list(space.values())
    return list(space.get_hyperparameters())


def _conditions(space):
    if hasattr(space, "conditions"):
        return list(space.conditions)
    return list(space.get_conditions())


def _forbiddens(space):
    if hasattr(space, "forbidden_clauses"):
        return list(space.forbidden_clauses)
    if hasattr(space, "get_forbiddens"):
        return list(space.get_forbiddens())
    return []


def _class_name(obj):
    return type(obj).__name__


# ------------------------------------------------------ ConfigSpace -> BUTChC

def from_configspace(space, drop=()):
    """Convert a ``ConfigurationSpace`` to a BUTChC search space.

    Args:
        space: A ``ConfigSpace.ConfigurationSpace``.
        drop:  Parameter names to exclude from the search entirely. Fidelity
               parameters (``trainsize``, ``repl``, ``epoch``) and task
               selectors (``task_id``, ``OpenML_task_id``) belong here, since
               the benchmark fixes them rather than optimising them.

    Returns:
        ``(searchspace, fixed, casts)``.

        ``searchspace`` is ready for ``BUTChC_optimize``. ``fixed`` maps
        constant parameters to their values; merge it into every config before
        evaluating. ``casts`` maps parameter names to a callable applied to the
        sampled value — currently only rounding, for log-scaled integers.
        ``wrap_objective`` applies both for you, and ``BUTChC_optimize`` calls
        it automatically when handed a ``ConfigurationSpace`` directly.

    Raises:
        UnsupportedSpace: With a message naming the offending parameter.
    """
    drop = set(drop)
    forbidden = _forbiddens(space)
    if forbidden:
        raise UnsupportedSpace(
            f"{len(forbidden)} forbidden clause(s) present. BUTChC cannot "
            f"express an invalid combination, so it would spend budget on "
            f"configurations the benchmark rejects. Filter them in the "
            f"objective, or pick a scenario without them."
        )

    specs, fixed, casts = {}, {}, {}
    for hp in _hyperparameters(space):
        name = hp.name
        if name in drop:
            continue
        spec = _convert_one(hp, casts)
        if spec is None:
            fixed[name] = hp.value
        else:
            specs[name] = spec

    children = _index_conditions(space, drop, specs)
    return _assemble(specs, children), fixed, casts


def _convert_one(hp, casts):
    """One hyperparameter to a spec dict, or None if it is a constant."""
    kind = _class_name(hp)

    if kind in ("Constant", "UnParametrizedHyperparameter"):
        return None

    if kind == "CategoricalHyperparameter":
        return {"values": list(hp.choices)}

    if kind == "OrdinalHyperparameter":
        # Order is discarded: BUTChC has no ordinal node, so an ordinal becomes
        # an unordered choice. That loses the neighbourhood structure, which is
        # a real handicap on long sequences and worth noting when reading
        # results.
        return {"values": list(hp.sequence)}

    if kind == "UniformIntegerHyperparameter":
        if getattr(hp, "log", False):
            casts[hp.name] = _round_to_int
            return {"min": float(hp.lower), "max": float(hp.upper), "log": True}
        return {"min": int(hp.lower), "max": int(hp.upper), "int": True}

    if kind == "UniformFloatHyperparameter":
        return {"min": float(hp.lower), "max": float(hp.upper),
                "log": bool(getattr(hp, "log", False))}

    raise UnsupportedSpace(f"{hp.name}: unhandled hyperparameter type {kind}")


def _round_to_int(value):
    """Module-level so a wrapped objective stays picklable for process pools."""
    return int(round(value))


def _index_conditions(space, drop, specs):
    """Map each conditional parameter to ``(parent_name, [values])``.

    Raises:
        UnsupportedSpace: On conjunctions, multiple parents, or a parent that
            is not categorical — none of which a ``next_level`` tree can hold.
    """
    children = {}
    for condition in _conditions(space):
        kind = _class_name(condition)
        if kind in ("AndConjunction", "OrConjunction"):
            names = sorted({c.child.name for c in condition.components})
            raise UnsupportedSpace(
                f"{names}: gated by a {kind}. A tree gives each parameter one "
                f"parent, so a conjunction has no single place to live. "
                f"Flatten the condition or exclude the parameter."
            )

        child, parent = condition.child.name, condition.parent.name
        if child in drop or parent in drop:
            continue
        if child in children:
            raise UnsupportedSpace(
                f"{child}: conditioned on both {children[child][0]!r} and "
                f"{parent!r}. A tree gives each parameter one parent."
            )
        if parent not in specs or "values" not in specs[parent]:
            raise UnsupportedSpace(
                f"{child}: parent {parent!r} is not a categorical parameter, "
                f"so it cannot carry a next_level."
            )

        if kind == "EqualsCondition":
            values = [condition.value]
        elif kind == "InCondition":
            values = list(condition.values)
        else:
            raise UnsupportedSpace(
                f"{child}: unhandled condition type {kind}. Only equality and "
                f"set membership map onto a branch."
            )

        unknown = set(values) - set(specs[parent]["values"])
        if unknown:
            raise UnsupportedSpace(
                f"{child}: condition names value(s) {sorted(map(str, unknown))} "
                f"that are not choices of {parent!r}"
            )
        children[child] = (parent, values)
    return children


def _assemble(specs, children):
    """Nest each conditional parameter under its parent's chosen branches."""
    conditional = set(children)
    tree = {name: spec for name, spec in specs.items() if name not in conditional}

    # Deepest-first ordering is not needed: a child is attached to its parent's
    # spec object, which is shared, so a grandchild attached later lands inside
    # the already-nested parent. Iterating until nothing moves keeps that true
    # whatever order the conditions arrive in.
    pending = dict(children)
    while pending:
        progressed = False
        for name, (parent, values) in list(pending.items()):
            parent_spec = specs[parent]
            if parent not in tree and parent in pending:
                continue                      # attach the parent first
            parent_spec.setdefault("next_level", {})
            for value in values:
                parent_spec["next_level"].setdefault(value, {})[name] = specs[name]
            del pending[name]
            progressed = True
        if not progressed:
            raise UnsupportedSpace(
                f"cyclic or unreachable conditions among {sorted(pending)}"
            )
    return tree


class _WrappedObjective:
    """Restores constants and applies casts before calling the real objective.

    A class rather than a closure so that a wrapped objective can still be sent
    to a process pool, which transmits by pickle and cannot pickle a closure.
    """

    def __init__(self, objective, fixed, casts):
        self.objective = objective
        self.fixed = dict(fixed)
        self.casts = dict(casts)

    def __call__(self, config, **kwargs):
        full = dict(self.fixed)
        for key, value in config.items():
            cast = self.casts.get(key)
            full[key] = cast(value) if cast else value
        return self.objective(full, **kwargs)


def wrap_objective(objective, fixed, casts):
    """Wrap an objective so it receives complete, correctly typed configs.

    BUTChC hands the objective only the parameters it searched. A benchmark
    built from a ``ConfigurationSpace`` expects the constants back, and expects
    log-scaled integers as integers. With nothing to restore, the objective is
    returned unchanged.
    """
    if not fixed and not casts:
        return objective
    return _WrappedObjective(objective, fixed, casts)


# ------------------------------------------------------ BUTChC -> ConfigSpace

def to_configspace(searchspace, name="butchc", seed=None):
    """Convert a BUTChC search space to a ``ConfigurationSpace``.

    Useful for handing your space to a benchmark harness, or to SMAC or Optuna
    for a like-for-like comparison. Unlike the reverse direction this never
    refuses: every tree is expressible as a DAG.

    One structural difference has to be reconciled. BUTChC lets the same name
    appear in sibling branches — ``lr`` under both ``adam`` and ``sgd`` — since
    those can never co-occur in one config. ConfigSpace requires globally unique
    names. Colliding parameters are therefore qualified as
    ``parent:value:name``, which is the convention auto-sklearn uses, and the
    returned mapping records it.

    Priors are **not** carried across. ConfigSpace's prior support does not
    share BUTChC's pseudo-trial semantics for ``prior_strength``, and a prior
    that silently means something else is worse than one that is absent.

    Args:
        searchspace: A BUTChC search-space dict.
        name:        Name for the resulting ``ConfigurationSpace``.
        seed:        Seed passed through to ``ConfigurationSpace``.

    Returns:
        ``(space, names)``. ``names`` maps each ConfigSpace name back to its
        BUTChC name, and is the identity for every parameter that did not need
        qualifying.

    Raises:
        ImportError: If ConfigSpace is not installed.
    """
    try:
        import ConfigSpace as CS
        import ConfigSpace.hyperparameters as CSH
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "to_configspace requires the ConfigSpace package: pip install ConfigSpace"
        ) from exc

    space = CS.ConfigurationSpace(name=name, seed=seed)
    builder = _Emitter(CS, CSH, space)
    builder.emit(searchspace, parent=None, values=None, prefix="")
    for hp in builder.hyperparameters:
        _add(space, hp)
    for condition in builder.conditions:
        _add(space, condition)
    return space, builder.names


def _add(space, item):
    """Add to a ConfigurationSpace across the 0.x and 1.x APIs."""
    if hasattr(space, "add"):
        space.add(item)
    elif hasattr(item, "child"):
        space.add_condition(item)
    else:
        space.add_hyperparameter(item)


class _Emitter:
    """Walks a BUTChC tree, collecting hyperparameters and their conditions.

    Hyperparameters are collected rather than added as they are found, because
    a condition can only be added once both of its endpoints exist and the walk
    reaches children before it has finished with their siblings.
    """

    def __init__(self, CS, CSH, space):
        self.CS = CS
        self.CSH = CSH
        self.hyperparameters = []
        self.conditions = []
        self.names = {}          # ConfigSpace name -> BUTChC name
        self._taken = set()

    def emit(self, subspace, parent, values, prefix):
        for key, spec in subspace.items():
            cs_name = self._unique(key, prefix)
            hp = _to_hyperparameter(cs_name, spec, self.CSH)
            self.hyperparameters.append(hp)
            self.names[cs_name] = key

            if parent is not None:
                self.conditions.append(_condition(self.CS, hp, parent, values))

            for choice, sub in (spec.get("next_level") or {}).items():
                self.emit(sub, parent=hp, values=[choice],
                          prefix=f"{key}:{choice}:")

    def _unique(self, key, prefix):
        """Qualify only on collision, so unambiguous names stay readable."""
        if key not in self._taken:
            self._taken.add(key)
            return key
        candidate = f"{prefix}{key}"
        suffix = 2
        while candidate in self._taken:
            candidate = f"{prefix}{key}_{suffix}"
            suffix += 1
        self._taken.add(candidate)
        return candidate


def _condition(CS, child, parent, values):
    if len(values) == 1:
        return CS.EqualsCondition(child, parent, values[0])
    return CS.InCondition(child, parent, list(values))


def _to_hyperparameter(name, spec, CSH):
    if "values" in spec:
        return CSH.CategoricalHyperparameter(name, list(spec["values"]))
    if spec.get("int"):
        lower, upper = int(spec["min"]), int(spec["max"])
        # default_value is passed explicitly because ConfigSpace 0.6.x types
        # the Cython argument as int and rejects the None it defaults to, so
        # omitting it raises TypeError on a version pyproject.toml declares as
        # supported. 1.x derives the same midpoint when told nothing, so this
        # changes no behaviour there.
        return CSH.UniformIntegerHyperparameter(
            name, lower=lower, upper=upper,
            default_value=lower + (upper - lower) // 2,
        )
    return CSH.UniformFloatHyperparameter(
        name,
        lower=float(spec["min"]),
        upper=float(spec["max"]),
        log=bool(spec.get("log", False)),
    )
