"""Round-tripping search spaces through ConfigSpace and JSON.

ConfigSpace itself is not a test dependency. The stubs below reproduce only the
attributes the bridge reads — class name, ``name``, bounds, ``choices``, and
``child``/``parent``/``value`` on conditions. That covers the part that can
actually be wrong: the DAG-to-tree assembly, the refusals, the name qualifying,
and whether the result is a space BUTChC will accept.

It does not cover API drift in ConfigSpace itself. ``dev/eval/run_real.py``
does that, against a real install.
"""

import json

import pytest

from butchc import BUTChC_optimize, SearchSpaceError
from butchc._validate import validate_searchspace
from butchc.interop import (
    UnsupportedSpace,
    from_configspace,
    from_json,
    to_configspace,
    to_json,
    wrap_objective,
)

# --------------------------------------------------------------------- stubs


class CategoricalHyperparameter:
    def __init__(self, name, choices):
        self.name, self.choices = name, choices


class OrdinalHyperparameter:
    def __init__(self, name, sequence):
        self.name, self.sequence = name, sequence


class UniformFloatHyperparameter:
    # default_value is accepted and ignored: the real 0.6.x constructors
    # require it, so the code under test passes it, and a stub that rejected it
    # would fail for a reason the real library does not have.
    def __init__(self, name, lower, upper, log=False, default_value=None):
        self.name, self.lower, self.upper, self.log = name, lower, upper, log


class UniformIntegerHyperparameter:
    def __init__(self, name, lower, upper, log=False, default_value=None):
        self.name, self.lower, self.upper, self.log = name, lower, upper, log


class Constant:
    def __init__(self, name, value):
        self.name, self.value = name, value


class EqualsCondition:
    def __init__(self, child, parent, value):
        self.child, self.parent, self.value = child, parent, value


class InCondition:
    def __init__(self, child, parent, values):
        self.child, self.parent, self.values = child, parent, values


class AndConjunction:
    def __init__(self, *components):
        self.components = components


class GreaterThanCondition:
    def __init__(self, child, parent, value):
        self.child, self.parent, self.value = child, parent, value


class ConfigurationSpace:
    def __init__(self, hyperparameters=(), conditions=(), forbiddens=()):
        self._hps = list(hyperparameters)
        self._conditions = list(conditions)
        self._forbiddens = list(forbiddens)

    def get_hyperparameters(self):
        return self._hps

    def get_conditions(self):
        return self._conditions

    def get_forbiddens(self):
        return self._forbiddens


def pipeline_space():
    """A two-level AutoML pipeline, the shape rbv2_super has."""
    learner = CategoricalHyperparameter("learner", ["svm", "rpart", "ranger"])
    return ConfigurationSpace(
        hyperparameters=[
            learner,
            CategoricalHyperparameter("kernel", ["rbf", "linear"]),
            UniformFloatHyperparameter("cost", 1e-4, 1e4, log=True),
            UniformIntegerHyperparameter("maxdepth", 1, 30),
            UniformIntegerHyperparameter("num.trees", 1, 2000, log=True),
            Constant("task_type", "classif"),
            UniformFloatHyperparameter("trainsize", 0.03, 1.0),
        ],
        conditions=[
            EqualsCondition(CategoricalHyperparameter("kernel", []), learner, "svm"),
            EqualsCondition(UniformFloatHyperparameter("cost", 0, 1), learner, "svm"),
            EqualsCondition(
                UniformIntegerHyperparameter("maxdepth", 0, 1), learner, "rpart"
            ),
            EqualsCondition(
                UniformIntegerHyperparameter("num.trees", 0, 1), learner, "ranger"
            ),
        ],
    )


# ------------------------------------------------------- ConfigSpace -> BUTChC


class TestFromConfigSpace:
    def test_produces_a_space_butchc_accepts(self):
        space, _fixed, _casts = from_configspace(pipeline_space(),
                                                 drop=["trainsize"])
        validate_searchspace(space)

    def test_conditionals_land_under_their_parent(self):
        space, _, _ = from_configspace(pipeline_space(), drop=["trainsize"])
        levels = space["learner"]["next_level"]
        assert set(levels["svm"]) == {"kernel", "cost"}
        assert set(levels["rpart"]) == {"maxdepth"}
        assert set(levels["ranger"]) == {"num.trees"}

    def test_unconditional_parameters_stay_at_the_top(self):
        space, _, _ = from_configspace(pipeline_space(), drop=["trainsize"])
        assert set(space) == {"learner"}

    def test_constants_are_returned_not_searched(self):
        space, fixed, _ = from_configspace(pipeline_space(), drop=["trainsize"])
        assert fixed == {"task_type": "classif"}
        assert "task_type" not in space

    def test_dropped_parameters_disappear_entirely(self):
        space, fixed, _ = from_configspace(pipeline_space(), drop=["trainsize"])
        assert "trainsize" not in space and "trainsize" not in fixed

    def test_log_float_keeps_its_scale(self):
        space, _, _ = from_configspace(pipeline_space(), drop=["trainsize"])
        assert space["learner"]["next_level"]["svm"]["cost"]["log"] is True

    def test_plain_integer_becomes_an_int_parameter(self):
        space, _, _ = from_configspace(pipeline_space(), drop=["trainsize"])
        spec = space["learner"]["next_level"]["rpart"]["maxdepth"]
        assert spec["int"] is True and "log" not in spec

    def test_log_integer_becomes_a_log_float_plus_a_cast(self):
        space, _, casts = from_configspace(pipeline_space(), drop=["trainsize"])
        spec = space["learner"]["next_level"]["ranger"]["num.trees"]
        assert spec["log"] is True and "int" not in spec
        assert casts["num.trees"](7.6) == 8

    def test_ordinal_becomes_an_unordered_choice(self):
        space, _, _ = from_configspace(ConfigurationSpace(
            [OrdinalHyperparameter("depth", [1, 2, 4, 8])]
        ))
        assert space["depth"] == {"values": [1, 2, 4, 8]}

    def test_in_condition_attaches_to_every_named_branch(self):
        parent = CategoricalHyperparameter("model", ["a", "b", "c"])
        space, _, _ = from_configspace(ConfigurationSpace(
            hyperparameters=[parent, UniformFloatHyperparameter("lr", 0.0, 1.0)],
            conditions=[
                InCondition(UniformFloatHyperparameter("lr", 0, 1), parent, ["a", "c"])
            ],
        ))
        levels = space["model"]["next_level"]
        assert set(levels) == {"a", "c"}
        assert "lr" in levels["a"] and "lr" in levels["c"]


class TestRefusals:
    def test_forbidden_clauses_are_refused(self):
        space = ConfigurationSpace(
            [CategoricalHyperparameter("a", ["x", "y"])], forbiddens=["anything"]
        )
        with pytest.raises(UnsupportedSpace, match="forbidden"):
            from_configspace(space)

    def test_conjunctions_are_refused_by_name(self):
        parent = CategoricalHyperparameter("model", ["a", "b"])
        child = UniformFloatHyperparameter("lr", 0.0, 1.0)
        space = ConfigurationSpace(
            hyperparameters=[parent, child],
            conditions=[AndConjunction(
                EqualsCondition(child, parent, "a"),
                EqualsCondition(child, parent, "b"),
            )],
        )
        with pytest.raises(UnsupportedSpace, match="lr"):
            from_configspace(space)

    def test_two_parents_for_one_child_are_refused(self):
        p1 = CategoricalHyperparameter("model", ["a", "b"])
        p2 = CategoricalHyperparameter("solver", ["p", "q"])
        child = UniformFloatHyperparameter("lr", 0.0, 1.0)
        space = ConfigurationSpace(
            hyperparameters=[p1, p2, child],
            conditions=[
                EqualsCondition(child, p1, "a"),
                EqualsCondition(child, p2, "p"),
            ],
        )
        with pytest.raises(UnsupportedSpace, match="one parent"):
            from_configspace(space)

    def test_continuous_parent_is_refused(self):
        parent = UniformFloatHyperparameter("threshold", 0.0, 1.0)
        child = UniformFloatHyperparameter("lr", 0.0, 1.0)
        space = ConfigurationSpace(
            hyperparameters=[parent, child],
            conditions=[GreaterThanCondition(child, parent, 0.5)],
        )
        with pytest.raises(UnsupportedSpace, match="not a categorical"):
            from_configspace(space)

    def test_unknown_hyperparameter_type_is_refused(self):
        class WeirdHyperparameter:
            name = "w"

        with pytest.raises(UnsupportedSpace, match="unhandled"):
            from_configspace(ConfigurationSpace([WeirdHyperparameter()]))

    def test_condition_on_an_unlisted_value_is_refused(self):
        parent = CategoricalHyperparameter("model", ["a", "b"])
        child = UniformFloatHyperparameter("lr", 0.0, 1.0)
        space = ConfigurationSpace(
            hyperparameters=[parent, child],
            conditions=[EqualsCondition(child, parent, "zzz")],
        )
        with pytest.raises(UnsupportedSpace, match="zzz"):
            from_configspace(space)


class TestWrapObjective:
    def test_constants_are_restored(self):
        seen = {}

        def objective(config):
            seen.update(config)
            return 0.0

        wrap_objective(objective, {"task": "classif"}, {})({"x": 1.0})
        assert seen == {"task": "classif", "x": 1.0}

    def test_casts_are_applied(self):
        seen = {}

        def objective(config):
            seen.update(config)
            return 0.0

        wrap_objective(objective, {}, {"n": round})({"n": 7.6})
        assert seen == {"n": 8}

    def test_kwargs_pass_through(self):
        def objective(config, factor):
            return factor

        assert wrap_objective(objective, {"a": 1}, {})({}, factor=3) == 3

    def test_nothing_to_restore_returns_the_original(self):
        def objective(config):
            return 0.0

        assert wrap_objective(objective, {}, {}) is objective


class TestOptimizerAcceptsConfigSpace:
    def test_configurationspace_is_converted_on_entry(self):
        seen = []

        def objective(config):
            seen.append(config)
            return -abs(config.get("maxdepth", 0) - 10)

        result = BUTChC_optimize(
            _small_space(), objective, 30, seed=0, verbose=False
        )
        assert result["best_params"] is not None
        # Constants restored on every call, without the caller doing anything.
        assert all(c["task_type"] == "classif" for c in seen)

    def test_a_non_space_non_dict_is_rejected(self):
        with pytest.raises(TypeError, match="ConfigurationSpace"):
            BUTChC_optimize(["not", "a", "space"], lambda c: 0.0, 5,
                            verbose=False)

    def test_dicts_are_untouched(self):
        space = {"x": {"min": 0.0, "max": 1.0}}
        result = BUTChC_optimize(space, lambda c: -c["x"], 10, seed=0,
                                 verbose=False)
        assert set(result["best_params"]) == {"x"}


def _small_space():
    learner = CategoricalHyperparameter("learner", ["rpart"])
    return ConfigurationSpace(
        hyperparameters=[
            learner,
            UniformIntegerHyperparameter("maxdepth", 1, 30),
            Constant("task_type", "classif"),
        ],
        conditions=[
            EqualsCondition(
                UniformIntegerHyperparameter("maxdepth", 0, 1), learner, "rpart"
            )
        ],
    )


# ------------------------------------------------------- BUTChC -> ConfigSpace


class FakeCS:
    """Just enough ConfigSpace surface for to_configspace to be exercised."""

    CategoricalHyperparameter = CategoricalHyperparameter
    UniformFloatHyperparameter = UniformFloatHyperparameter
    UniformIntegerHyperparameter = UniformIntegerHyperparameter
    EqualsCondition = EqualsCondition
    InCondition = InCondition

    class ConfigurationSpace:
        def __init__(self, name=None, seed=None):
            self.name, self.seed = name, seed
            self.hyperparameters = []
            self.conditions = []

        def add(self, item):
            if hasattr(item, "child"):
                self.conditions.append(item)
            else:
                self.hyperparameters.append(item)


def _to_cs(space):
    """Run to_configspace against the stubs by injecting them as the modules."""
    import sys
    import types

    cs = types.ModuleType("ConfigSpace")
    for attr in ("CategoricalHyperparameter", "UniformFloatHyperparameter",
                 "UniformIntegerHyperparameter", "EqualsCondition",
                 "InCondition", "ConfigurationSpace"):
        setattr(cs, attr, getattr(FakeCS, attr))
    hp = types.ModuleType("ConfigSpace.hyperparameters")
    for attr in ("CategoricalHyperparameter", "UniformFloatHyperparameter",
                 "UniformIntegerHyperparameter"):
        setattr(hp, attr, getattr(FakeCS, attr))
    cs.hyperparameters = hp

    saved = {k: sys.modules.get(k) for k in
             ("ConfigSpace", "ConfigSpace.hyperparameters")}
    sys.modules["ConfigSpace"] = cs
    sys.modules["ConfigSpace.hyperparameters"] = hp
    try:
        return to_configspace(space)
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


NESTED = {
    "optimizer": {
        "values": ["adam", "sgd"],
        "next_level": {
            "adam": {"lr": {"min": 1e-4, "max": 1e-2, "log": True}},
            "sgd": {
                "lr": {"min": 1e-3, "max": 1e-1, "log": True},
                "momentum": {"min": 0.0, "max": 0.99},
            },
        },
    },
    "epochs": {"min": 1, "max": 50, "int": True},
}


class TestToConfigSpace:
    def test_every_parameter_is_emitted(self):
        space, names = _to_cs(NESTED)
        assert len(space.hyperparameters) == 5
        assert set(names.values()) == {"optimizer", "lr", "momentum", "epochs"}

    def test_sibling_name_collisions_are_qualified(self):
        _space, names = _to_cs(NESTED)
        cs_names = set(names)
        assert "lr" in cs_names
        assert "optimizer:sgd:lr" in cs_names
        assert names["optimizer:sgd:lr"] == "lr"

    def test_unambiguous_names_are_left_alone(self):
        _space, names = _to_cs(NESTED)
        assert names["momentum"] == "momentum"
        assert names["epochs"] == "epochs"

    def test_conditions_are_emitted_for_conditional_parameters(self):
        space, _ = _to_cs(NESTED)
        pairs = {(c.child.name, c.value) for c in space.conditions}
        assert ("lr", "adam") in pairs
        assert ("optimizer:sgd:lr", "sgd") in pairs
        assert ("momentum", "sgd") in pairs

    def test_unconditional_parameters_get_no_condition(self):
        space, _ = _to_cs(NESTED)
        assert "epochs" not in {c.child.name for c in space.conditions}

    def test_scales_survive(self):
        space, _ = _to_cs(NESTED)
        by_name = {h.name: h for h in space.hyperparameters}
        assert by_name["lr"].log is True
        assert isinstance(by_name["epochs"], UniformIntegerHyperparameter)
        assert isinstance(by_name["optimizer"], CategoricalHyperparameter)

    def test_flat_space_needs_no_conditions(self):
        space, names = _to_cs({"x": {"min": 0.0, "max": 1.0}})
        assert space.conditions == []
        assert names == {"x": "x"}


# ------------------------------------------------------------------ JSON


class TestJson:
    def test_round_trip_preserves_a_flat_space(self):
        space = {
            "x": {"min": -1.0, "max": 1.0},
            "n": {"min": 1, "max": 10, "int": True},
            "k": {"values": ["a", "b"]},
        }
        assert from_json(to_json(space)) == space

    def test_round_trip_preserves_nesting(self):
        assert from_json(to_json(NESTED)) == NESTED

    def test_non_string_branch_keys_are_restored(self):
        # JSON stringifies object keys, so an integer-valued parent would come
        # back with next_level keyed by "1" and its branch unreachable.
        space = {
            "n_layers": {
                "values": [1, 2],
                "next_level": {
                    1: {"width": {"min": 8, "max": 64, "int": True}},
                    2: {"drop": {"min": 0.0, "max": 0.5}},
                },
            }
        }
        restored = from_json(to_json(space))
        assert set(restored["n_layers"]["next_level"]) == {1, 2}
        assert restored == space

    def test_round_tripped_space_still_optimises(self):
        space = from_json(to_json(NESTED))
        validate_searchspace(space)
        result = BUTChC_optimize(
            space, lambda c: -abs(c["epochs"] - 20), 40, seed=0, verbose=False
        )
        assert result["best_params"] is not None

    def test_output_is_real_json(self):
        assert json.loads(to_json(NESTED, indent=2))["epochs"]["int"] is True

    def test_priors_survive_the_round_trip(self):
        space = {
            "opt": {
                "values": ["adam", "sgd"],
                "prior": {"adam": 0.7, "sgd": 0.3},
                "prior_strength": 20,
            }
        }
        assert from_json(to_json(space)) == space

    def test_malformed_space_still_fails_validation_after_a_round_trip(self):
        with pytest.raises(SearchSpaceError):
            validate_searchspace(from_json('{"x": {"min": 1.0}}'))


# ------------------------------------------------- against a real ConfigSpace
#
# Everything above runs against stubs, which is what makes the suite
# dependency-free — but stubs are written from the same reading of the API that
# the bridge was, so they cannot catch a misreading. These close that gap where
# ConfigSpace is installed and skip where it is not, so `pip install
# butchc[configspace] && pytest` is a real check of the API surface.
#
# The version accessors are duplicated from the bridge on purpose: writing them
# out here means a 0.x/1.x rename shows up as a failing test rather than as two
# copies of the same wrong assumption.


def _cs():
    """Import ConfigSpace or skip. Returns (CS, CSH)."""
    # `reason=` is not accepted by every supported pytest, so the
    # install hint lives in the module docstring instead.
    CS = pytest.importorskip("ConfigSpace")
    import ConfigSpace.hyperparameters as CSH
    return CS, CSH


def _cs_add(space, item):
    if hasattr(space, "add"):
        space.add(item)
    elif hasattr(item, "child"):
        space.add_condition(item)
    elif "Forbidden" in type(item).__name__:
        # 0.6.x has no unified `add`, and a forbidden clause carries no
        # `.child`, so without this branch it fell through to
        # add_hyperparameter and raised about the wrong thing entirely.
        space.add_forbidden_clause(item)
    else:
        space.add_hyperparameter(item)


def _cs_names(space):
    # 1.x spelling first: on 1.x the 0.x accessor still exists and only warns,
    # so probing for it first would warn on every call. Mirrors the ordering in
    # butchc.interop._configspace.
    if hasattr(space, "keys"):
        return set(space.keys())
    return set(space.get_hyperparameter_names())


class TestAgainstRealConfigSpace:
    """Skipped unless ConfigSpace is installed."""

    def _pipeline(self):
        """A conditional space of the shape the bridge exists to convert."""
        CS, CSH = _cs()
        space = CS.ConfigurationSpace(name="real", seed=0)
        model = CSH.CategoricalHyperparameter("model", choices=["svm", "rf"])
        C = CSH.UniformFloatHyperparameter("C", lower=1e-3, upper=1e3, log=True)
        # default_value is explicit because ConfigSpace 0.6.x rejects the None
        # it otherwise defaults to; see _to_hyperparameter for the same reason.
        trees = CSH.UniformIntegerHyperparameter("n_estimators", lower=10,
                                                 upper=500, default_value=255)
        seed = CSH.Constant("random_state", 0)
        for hp in (model, C, trees, seed):
            _cs_add(space, hp)
        _cs_add(space, CS.EqualsCondition(C, model, "svm"))
        _cs_add(space, CS.EqualsCondition(trees, model, "rf"))
        return space

    def test_conditions_become_branches(self):
        space, fixed, casts = from_configspace(self._pipeline())
        validate_searchspace(space)
        assert set(space["model"]["next_level"]["svm"]) == {"C"}
        assert set(space["model"]["next_level"]["rf"]) == {"n_estimators"}
        assert space["model"]["next_level"]["svm"]["C"]["log"] is True
        assert space["model"]["next_level"]["rf"]["n_estimators"]["int"] is True

    def test_constants_are_returned_not_searched(self):
        space, fixed, casts = from_configspace(self._pipeline())
        assert fixed == {"random_state": 0}
        assert "random_state" not in space

    def test_forbidden_clauses_are_refused(self):
        CS, CSH = _cs()
        space = self._pipeline()
        model = space["model"] if hasattr(space, "__getitem__") else None
        if model is None:  # pragma: no cover - very old ConfigSpace
            pytest.skip("no item access on this ConfigSpace version")
        _cs_add(space, CS.ForbiddenEqualsClause(model, "rf"))
        with pytest.raises(UnsupportedSpace, match="forbidden"):
            from_configspace(space)

    def test_optimizer_accepts_a_real_configuration_space(self):
        seen = []

        def objective(config):
            seen.append(dict(config))
            if config["model"] == "svm":
                return -abs(config["C"] - 1.0)
            return -abs(config["n_estimators"] - 100) / 100.0

        result = BUTChC_optimize(self._pipeline(), objective, 40, seed=0,
                                 verbose=False)
        assert result["best_params"] is not None
        # Constants are restored into what the objective receives, not into
        # what the search reports: `best_params` holds the searched parameters
        # only, and every evaluated config carries the constant.
        assert all(config["random_state"] == 0 for config in seen)
        assert "random_state" not in result["best_params"]

    def test_to_configspace_builds_a_sampleable_space(self):
        _cs()
        space, names = to_configspace(NESTED, seed=0)
        assert _cs_names(space) >= {"optimizer", "epochs"}
        # A drawn configuration must carry the branch's own parameters and
        # nothing from the sibling branch.
        for _ in range(20):
            drawn = dict(space.sample_configuration())
            chosen = drawn["optimizer"]
            other = "sgd" if chosen == "adam" else "adam"
            assert not any(key.startswith(f"optimizer:{other}:")
                           for key in drawn)

    def test_round_trip_preserves_the_tree(self):
        _cs()
        space, names = to_configspace(NESTED, seed=0)
        back, fixed, casts = from_configspace(space)
        validate_searchspace(back)
        assert set(back["optimizer"]["values"]) == {"adam", "sgd"}
        # `lr` collides across siblings, so it is qualified on the way out.
        # What must survive is the structure, not the spelling.
        adam = back["optimizer"]["next_level"]["adam"]
        sgd = back["optimizer"]["next_level"]["sgd"]
        assert len(adam) == 1 and len(sgd) == 2
        assert all(names[key] in ("lr", "momentum")
                   for key in list(adam) + list(sgd))

    def test_qualified_names_map_back(self):
        _cs()
        _, names = to_configspace(NESTED, seed=0)
        assert names["epochs"] == "epochs"
        assert sorted(v for v in names.values() if v == "lr") == ["lr", "lr"]
