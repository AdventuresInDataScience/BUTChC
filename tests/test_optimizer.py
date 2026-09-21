"""Integration tests for butchc.BUTChC_optimize.

The class that matters most is TestOptimizationQuality. v0.1 shipped a
convergence suite whose every assertion passed under pure random search — for
instance, asserting that 300 draws from [-5, 5] land within 1.5 of the optimum
at least once, which fails with probability 0.7**300. Those tests could not
detect that the optimiser ignored the objective. The tests here compare
against a random-search baseline on the same budget, which is the only
comparison that can.
"""

import math
import random
import statistics

import pytest

from butchc import BUTChC_optimize, SearchSpaceError, reservoir_summary

FLAT_SPACE = {
    "x": {"min": -5.0, "max": 5.0},
    "method": {"values": ["a", "b", "c"]},
}

HIERARCHICAL_SPACE = {
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
    "batch_size": {"values": [16, 32, 64]},
}


def quadratic(config):
    """Maximise -(x-1)^2; peak at x=1."""
    return -((config["x"] - 1.0) ** 2)


def constant_zero(config):
    return 0.0


def _run(space=FLAT_SPACE, obj=quadratic, budget=30, **kw):
    kw.setdefault("seed", 0)
    return BUTChC_optimize(
        space, obj, budget=budget, lambda_=1.0, alpha=1.0, verbose=False, **kw
    )


def random_search(space, obj, budget, seed):
    """Uniform baseline over a flat continuous/categorical space."""
    rng = random.Random(seed)
    best = -float("inf")
    for _ in range(budget):
        config = {
            k: (rng.uniform(v["min"], v["max"]) if "min" in v else rng.choice(v["values"]))
            for k, v in space.items()
        }
        best = max(best, obj(config))
    return best


def median(values):
    return sorted(values)[len(values) // 2]


# ---------------------------------------------------------------------------
# Optimisation quality — the tests that actually matter
# ---------------------------------------------------------------------------

class TestOptimizationQuality:
    def test_objective_direction_changes_the_learned_tree(self):
        # v0.1 regression: the update rule never saw the objective, so
        # optimising f and -f produced bit-identical trees.
        space = {"choice": {"values": ["a", "b", "c"]}}
        scores = {"a": 1.0, "b": 0.5, "c": 0.1}

        def run(sign):
            return BUTChC_optimize(
                space,
                lambda c: sign * scores[c["choice"]],
                budget=300,
                lambda_=1.0,
                alpha=1.0,
                verbose=False,
                seed=7,
            )["prob_tree"]["choice"]["prob"]

        forward, reversed_ = run(1.0), run(-1.0)
        assert forward["a"] > forward["c"]
        assert reversed_["c"] > reversed_["a"]

    def test_categorical_probability_concentrates_on_the_best_choice(self):
        # Note this asserts on the *learned distribution*, not on whether the
        # best value was ever sampled — the latter is guaranteed by chance.
        scores = {"a": 1.0, "b": 0.5, "c": 0.1}
        tree = BUTChC_optimize(
            {"choice": {"values": ["a", "b", "c"]}},
            lambda c: scores[c["choice"]],
            budget=300,
            lambda_=1.0,
            alpha=1.0,
            verbose=False,
            seed=1,
        )["prob_tree"]
        assert tree["choice"]["prob"]["a"] > 0.5

    def test_continuous_reservoir_concentrates_on_the_optimum(self):
        result = BUTChC_optimize(
            {"x": {"min": -5.0, "max": 5.0}},
            lambda c: -((c["x"] - 2.0) ** 2),
            budget=300,
            lambda_=1.0,
            alpha=1.0,
            verbose=False,
            seed=2,
        )
        summary = reservoir_summary(result["prob_tree"]["x"])
        assert abs(summary["mean"] - 2.0) < 0.5

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_beats_random_search_on_5d_sphere(self, seed):
        space = {k: {"min": -5.0, "max": 5.0} for k in "abcde"}

        def obj(c):
            return -sum((c[k] - 2.0) ** 2 for k in "abcde")

        tuned = BUTChC_optimize(
            space, obj, budget=400, lambda_=1.0, alpha=1.0, verbose=False, seed=seed
        )["best_value"]
        baseline = random_search(space, obj, 400, seed)
        assert tuned > baseline

    def test_beats_random_search_on_10d_sphere(self):
        space = {f"x{i}": {"min": -5.0, "max": 5.0} for i in range(10)}

        def obj(c):
            return -sum((v - 1.0) ** 2 for v in c.values())

        tuned = median(
            [
                BUTChC_optimize(
                    space, obj, budget=600, lambda_=1.0, alpha=1.0, verbose=False, seed=s
                )["best_value"]
                for s in range(5)
            ]
        )
        baseline = median([random_search(space, obj, 600, s) for s in range(5)])
        assert tuned > baseline

    def test_log_scale_finds_small_magnitudes(self):
        # Uniform sampling on [1e-6, 1] puts ~99.9% of its mass above 1e-3,
        # so the target at 1e-5 is effectively unreachable without log=True.
        result = BUTChC_optimize(
            {"lr": {"min": 1e-6, "max": 1.0, "log": True}},
            lambda c: -abs(math.log10(c["lr"]) + 5.0),
            budget=200,
            lambda_=1.0,
            alpha=1.0,
            verbose=False,
            seed=0,
        )
        assert abs(math.log10(result["best_params"]["lr"]) + 5.0) < 0.2

    def test_conditional_branches_learn_independent_optima(self):
        # Both branches peak at 0, so a converging optimiser would rightly
        # starve one of them. A large alpha holds the categorical near
        # uniform, which is what isolates the property under test: each
        # branch's continuous node learns from its own trials only.
        space = {
            "mode": {
                "values": ["left", "right"],
                "next_level": {
                    "left": {"v": {"min": 0.0, "max": 10.0}},
                    "right": {"v2": {"min": 0.0, "max": 10.0}},
                },
            }
        }

        def obj(c):
            return -abs(c["v"] - 2.0) if c["mode"] == "left" else -abs(c["v2"] - 8.0)

        tree = BUTChC_optimize(
            space, obj, budget=600, lambda_=1.0, alpha=500.0, verbose=False, seed=4
        )["prob_tree"]
        left = reservoir_summary(tree["mode"]["next_level"]["left"]["v"])["mean"]
        right = reservoir_summary(tree["mode"]["next_level"]["right"]["v2"])["mean"]
        assert abs(left - 2.0) < 1.0
        assert abs(right - 8.0) < 1.0

    def test_budget_concentrates_on_the_better_branch(self):
        space = {
            "mode": {
                "values": ["good", "bad"],
                "next_level": {
                    "good": {"v": {"min": 0.0, "max": 10.0}},
                    "bad": {"w": {"min": 0.0, "max": 10.0}},
                },
            }
        }

        def obj(c):
            return -abs(c["v"] - 2.0) if c["mode"] == "good" else -20.0 - abs(c["w"] - 5.0)

        result = BUTChC_optimize(
            space, obj, budget=400, lambda_=1.0, alpha=1.0, verbose=False, seed=4
        )
        tree = result["prob_tree"]
        assert tree["mode"]["prob"]["good"] > 0.8
        assert abs(reservoir_summary(tree["mode"]["next_level"]["good"]["v"])["mean"] - 2.0) < 1.0


# ---------------------------------------------------------------------------
# Return structure
# ---------------------------------------------------------------------------

class TestReturnStructure:
    EXPECTED_KEYS = {
        "best_params", "best_value", "prob_tree",
        "history", "loss_history", "rolling_loss", "n_updates", "n_gated",
        "n_batches",
    }

    def test_all_keys_present(self):
        assert self.EXPECTED_KEYS == set(_run().keys())

    @pytest.mark.parametrize("budget", [1, 7, 13])
    def test_history_length_equals_budget(self, budget):
        result = _run(budget=budget)
        assert len(result["history"]) == budget
        assert len(result["loss_history"]) == budget
        assert len(result["rolling_loss"]) == budget

    def test_history_entries_have_required_keys(self):
        for entry in _run(budget=5)["history"]:
            assert {"params", "objective", "loss", "rolling_loss", "quality",
                    "updated", "gated"}.issubset(entry)

    def test_convenience_copies_match_history(self):
        result = _run(budget=20)
        assert result["loss_history"] == [h["loss"] for h in result["history"]]
        assert result["rolling_loss"] == [h["rolling_loss"] for h in result["history"]]

    def test_types(self):
        result = _run()
        assert isinstance(result["best_params"], dict)
        assert isinstance(result["best_value"], float)
        assert isinstance(result["prob_tree"], dict)
        assert isinstance(result["n_updates"], int)


class TestBestTracking:
    def test_best_value_equals_max_observed(self):
        result = _run(budget=60)
        assert result["best_value"] == max(h["objective"] for h in result["history"])

    def test_best_params_reproduces_best_value(self):
        result = _run(budget=60)
        assert quadratic(result["best_params"]) == pytest.approx(result["best_value"])

    def test_budget_one_has_valid_best(self):
        result = _run(budget=1)
        assert result["best_params"] is not None
        assert result["best_value"] == result["history"][0]["objective"]


# ---------------------------------------------------------------------------
# Loss / convergence signal
# ---------------------------------------------------------------------------

class TestLossProperties:
    def test_losses_non_negative(self):
        result = _run(budget=60)
        assert all(l >= 0.0 for l in result["loss_history"])
        assert all(l >= 0.0 for l in result["rolling_loss"])

    def test_gated_flag_matches_nonzero_quality(self):
        for entry in _run(budget=80)["history"]:
            assert entry["gated"] == (entry["quality"] > 0.0)

    def test_n_gated_counts_gated_trials(self):
        result = _run(budget=80)
        assert result["n_gated"] == sum(h["gated"] for h in result["history"])

    def test_categorical_movement_is_reported_outside_the_gate(self):
        # A space whose only node is categorical still moves on every trial,
        # so trials that failed the gate must not all report zero loss.
        result = BUTChC_optimize(
            {"a": {"values": ["x", "y", "z"]}},
            lambda c: {"x": 1.0, "y": 0.0, "z": 0.0}[c["a"]],
            budget=40, verbose=False, seed=0,
        )
        ungated = [h for h in result["history"] if not h["gated"]]
        assert ungated
        assert any(h["loss"] > 0.0 for h in ungated)

    def test_n_updates_counts_updated_trials(self):
        result = _run(budget=80)
        assert result["n_updates"] == sum(h["updated"] for h in result["history"])

    def test_warmup_trials_never_clear_the_gate(self):
        result = _run(budget=100, n_warmup=25)
        assert not any(h["gated"] for h in result["history"][:25])

    def test_warmup_still_scores_categorical_nodes(self):
        # Warm-up suppresses the continuous archive, not the whole model.
        # Categorical choices are scored by mean rank over every visit, and
        # warm-up is the unbiased sample that keeps an untried branch from
        # being written off — so it deliberately does count. Documented in
        # docs/api.md; the docs claimed "updating nothing" for three releases.
        space = {"k": {"values": ["a", "b", "c"]}}
        ordered = {"a": 1.0, "b": 2.0, "c": 3.0}
        result = BUTChC_optimize(
            space, lambda c: ordered[c["k"]], budget=30, n_warmup=30,
            verbose=False, seed=0,
        )
        assert result["n_gated"] == 0                # no archive was touched
        assert result["n_updates"] > 0               # but the tree moved
        prob = result["prob_tree"]["k"]["prob"]
        assert prob["c"] > prob["b"] > prob["a"]     # and moved the right way

    def test_warmup_is_not_skipped_for_a_warm_started_run(self):
        # There is no "this tree already knows something" detection, and
        # docs/examples.md used to claim there was.
        space = {"k": {"values": ["a", "b", "c"]}}
        ordered = {"a": 1.0, "b": 2.0, "c": 3.0}
        first = BUTChC_optimize(space, lambda c: ordered[c["k"]], budget=40,
                                verbose=False, seed=0)
        second = BUTChC_optimize(
            space, lambda c: ordered[c["k"]], budget=40, n_warmup=40,
            start_prob_tree=first["prob_tree"], verbose=False, seed=1,
        )
        assert second["n_gated"] == 0
        assert all(h["quality"] == 0.0 for h in second["history"])

    def test_rolling_loss_declines_as_tree_settles(self):
        result = BUTChC_optimize(
            {"x": {"min": -5.0, "max": 5.0}},
            quadratic,
            budget=400,
            lambda_=1.0,
            alpha=1.0,
            verbose=False,
            seed=0,
        )
        early = result["rolling_loss"][80]
        late = result["rolling_loss"][-1]
        assert late < early


# ---------------------------------------------------------------------------
# Objective plumbing
# ---------------------------------------------------------------------------

class TestObjectiveForwarding:
    def test_objective_called_exactly_budget_times(self):
        calls = []
        _run(obj=lambda c: calls.append(1) or 0.0, budget=17)
        assert len(calls) == 17

    def test_kwargs_forwarded(self):
        seen = []

        def obj(config, a=0, b=0):
            seen.append((a, b))
            return 0.0

        BUTChC_optimize(
            {"x": {"min": 0.0, "max": 1.0}}, obj, budget=3,
            lambda_=1.0, alpha=1.0, verbose=False, seed=0, a=3, b=9,
        )
        assert all(pair == (3, 9) for pair in seen)

    def test_config_contains_all_root_params(self):
        seen = []
        _run(obj=lambda c: seen.append(c) or 0.0, budget=5)
        assert all({"x", "method"} <= set(c) for c in seen)

    def test_rejects_non_callable_objective(self):
        with pytest.raises(TypeError):
            BUTChC_optimize(FLAT_SPACE, "not callable", budget=2,
                            lambda_=1.0, alpha=1.0, verbose=False)

    def test_rejects_non_numeric_return(self):
        with pytest.raises(TypeError, match="objective returned"):
            _run(obj=lambda c: "nonsense", budget=2)

    def test_rejects_none_return(self):
        with pytest.raises(TypeError, match="objective returned"):
            _run(obj=lambda c: None, budget=2)


class TestNonFiniteObjectives:
    def test_nan_trials_do_not_crash(self):
        result = _run(obj=lambda c: float("nan"), budget=20)
        assert result["best_params"] is None
        assert result["best_value"] == -float("inf")

    def test_nan_trials_never_update_the_tree(self):
        assert _run(obj=lambda c: float("nan"), budget=40)["n_updates"] == 0

    def test_mixed_nan_and_finite_finds_the_finite_best(self):
        def obj(config):
            return float("nan") if config["x"] < 0 else config["x"]

        result = _run(obj=obj, budget=100)
        assert result["best_value"] > 0
        assert result["best_params"]["x"] > 0


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_reproduces_run(self):
        a, b = _run(budget=40, seed=99), _run(budget=40, seed=99)
        assert a["best_value"] == b["best_value"]
        assert a["loss_history"] == b["loss_history"]

    def test_different_seeds_diverge(self):
        a, b = _run(budget=40, seed=1), _run(budget=40, seed=2)
        assert a["loss_history"] != b["loss_history"]

    def test_does_not_depend_on_global_random_state(self):
        random.seed(1234)
        first = _run(budget=30, seed=5)["best_value"]
        random.seed(999999)
        second = _run(budget=30, seed=5)["best_value"]
        assert first == second

    def test_does_not_disturb_global_random_state(self):
        random.seed(4321)
        expected = [random.random() for _ in range(3)]
        random.seed(4321)
        _run(budget=30, seed=5)
        assert [random.random() for _ in range(3)] == expected


# ---------------------------------------------------------------------------
# Warm starting
# ---------------------------------------------------------------------------

class TestWarmStart:
    def test_chaining_runs(self):
        first = _run(budget=60)
        second = BUTChC_optimize(
            FLAT_SPACE, quadratic, budget=60, lambda_=1.0, alpha=1.0,
            verbose=False, seed=1, start_prob_tree=first["prob_tree"],
        )
        assert len(second["history"]) == 60

    def test_input_tree_is_not_mutated(self):
        import copy

        first = _run(budget=60)
        snapshot = copy.deepcopy(first["prob_tree"])
        BUTChC_optimize(
            FLAT_SPACE, quadratic, budget=30, lambda_=1.0, alpha=1.0,
            verbose=False, seed=1, start_prob_tree=first["prob_tree"],
        )
        assert first["prob_tree"] == snapshot

    def test_warm_start_retains_learned_information(self):
        first = _run(budget=300, obj=quadratic)
        second = BUTChC_optimize(
            FLAT_SPACE, quadratic, budget=30, lambda_=1.0, alpha=1.0,
            verbose=False, seed=3, start_prob_tree=first["prob_tree"],
        )
        # Continuing a converged run should not restart from the flat prior.
        assert abs(reservoir_summary(second["prob_tree"]["x"])["mean"] - 1.0) < 1.0

    def test_warm_start_skips_warmup(self):
        first = _run(budget=200)
        second = BUTChC_optimize(
            FLAT_SPACE, quadratic, budget=40, lambda_=1.0, alpha=1.0,
            verbose=False, seed=3, start_prob_tree=first["prob_tree"],
        )
        assert second["n_updates"] > 0

    def test_mismatched_searchspace_is_rejected(self):
        # v0.1 silently ignored searchspace when a tree was supplied.
        first = _run(budget=20)
        with pytest.raises(ValueError, match="does not match searchspace"):
            BUTChC_optimize(
                {"totally": {"values": ["different"]}}, quadratic, budget=5,
                lambda_=1.0, alpha=1.0, verbose=False, start_prob_tree=first["prob_tree"],
            )

    def test_garbage_tree_is_rejected(self):
        with pytest.raises(ValueError):
            BUTChC_optimize(
                FLAT_SPACE, quadratic, budget=5, lambda_=1.0, alpha=1.0,
                verbose=False, start_prob_tree={"x": {"junk": 1}},
            )


# ---------------------------------------------------------------------------
# Hierarchical spaces
# ---------------------------------------------------------------------------

class TestHierarchicalSpace:
    def test_only_relevant_params_appear(self):
        seen = []
        BUTChC_optimize(
            HIERARCHICAL_SPACE, lambda c: seen.append(c) or 0.0, budget=200,
            lambda_=1.0, alpha=1.0, verbose=False, seed=42,
        )
        adam = [c for c in seen if c["optimizer"] == "adam"]
        sgd = [c for c in seen if c["optimizer"] == "sgd"]
        assert adam and sgd
        assert all("momentum" not in c for c in adam)
        assert all("momentum" in c for c in sgd)
        assert all({"optimizer", "lr", "batch_size"} <= set(c) for c in seen)

    def test_branch_specific_bounds_are_respected(self):
        seen = []
        BUTChC_optimize(
            HIERARCHICAL_SPACE, lambda c: seen.append(c) or 0.0, budget=200,
            lambda_=1.0, alpha=1.0, verbose=False, seed=42,
        )
        for c in seen:
            lo, hi = (1e-4, 1e-2) if c["optimizer"] == "adam" else (1e-3, 1e-1)
            assert lo <= c["lr"] <= hi

    def test_choice_without_next_level_is_allowed(self):
        space = {
            "opt": {
                "values": ["a", "b"],
                "next_level": {"a": {"p": {"min": 0.0, "max": 1.0}}},
            }
        }
        seen = []
        BUTChC_optimize(
            space, lambda c: seen.append(c) or 0.0, budget=60,
            lambda_=1.0, alpha=1.0, verbose=False, seed=0,
        )
        assert any(c["opt"] == "b" and "p" not in c for c in seen)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestSearchSpaceValidation:
    @pytest.mark.parametrize(
        "space",
        [
            {},
            {"x": {}},
            {"x": {"minimum": 0.0, "maximum": 1.0}},
            {"x": {"min": 0.0}},
            {"x": {"min": 1.0, "max": 0.0}},
            {"x": {"min": 0.0, "max": 1.0, "values": [1]}},
            {"a": {"values": []}},
            {"a": {"values": ["x", "x"]}},
            {"a": {"values": "not a list"}},
            {"lr": {"min": 0.0, "max": 1.0, "log": True}},
            {"n": {"min": 0.0, "max": 0.2, "int": True}},
            {"n": {"min": 1.0, "max": 10.0, "int": True, "log": True}},
            {"a": {"values": ["x"], "next_level": {"nope": {"p": {"min": 0.0, "max": 1.0}}}}},
            {"a": {"values": ["x"], "typo": 1}},
        ],
    )
    def test_malformed_spaces_rejected_before_any_evaluation(self, space):
        calls = []
        with pytest.raises(SearchSpaceError):
            BUTChC_optimize(space, lambda c: calls.append(1) or 0.0, budget=5,
                            lambda_=1.0, alpha=1.0, verbose=False)
        assert calls == []

    def test_error_message_names_the_offending_path(self):
        space = {"a": {"values": ["x"], "next_level": {"x": {"deep": {"min": 5.0, "max": 1.0}}}}}
        with pytest.raises(SearchSpaceError, match="deep"):
            BUTChC_optimize(space, constant_zero, budget=5, lambda_=1.0,
                            alpha=1.0, verbose=False)

    def test_duplicate_name_across_levels_rejected(self):
        # v0.1 crashed with a bare KeyError partway through a run.
        space = {"m": {"values": ["x"], "next_level": {"x": {"m": {"values": ["p"]}}}}}
        with pytest.raises(SearchSpaceError, match="ancestor"):
            BUTChC_optimize(space, constant_zero, budget=5, lambda_=1.0,
                            alpha=1.0, verbose=False)

    def test_duplicate_name_across_siblings_rejected(self):
        space = {
            "a": {"values": ["x"], "next_level": {"x": {"shared": {"min": 0.0, "max": 1.0}}}},
            "shared": {"min": 0.0, "max": 1.0},
        }
        with pytest.raises(SearchSpaceError, match="shared"):
            BUTChC_optimize(space, constant_zero, budget=5, lambda_=1.0,
                            alpha=1.0, verbose=False)


class TestHyperparameterValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"budget": 0}, {"budget": -1}, {"budget": 1.5},
            {"temp": 0.0}, {"temp": -1.0},
            {"lambda_": 0.0}, {"lambda_": -1.0},
            {"alpha": 0.0}, {"alpha": -2.0},
            {"gamma": 1.0}, {"gamma": -0.1},
            {"explore": 1.5}, {"explore": -0.1},
            {"n_warmup": -1},
        ],
    )
    def test_out_of_range_values_rejected(self, kwargs):
        params = {"budget": 10, "lambda_": 1.0, "alpha": 1.0}
        params.update(kwargs)
        with pytest.raises(ValueError):
            BUTChC_optimize(FLAT_SPACE, quadratic, verbose=False, **params)


# ---------------------------------------------------------------------------
# Verbose output and edge cases
# ---------------------------------------------------------------------------

class TestVerbose:
    def test_silent_when_disabled(self, capsys):
        _run(budget=3)
        assert capsys.readouterr().out == ""

    def test_one_line_per_trial(self, capsys):
        BUTChC_optimize(FLAT_SPACE, quadratic, budget=6, lambda_=1.0,
                        alpha=1.0, verbose=True, seed=0)
        lines = [l for l in capsys.readouterr().out.splitlines() if l.strip()]
        assert len(lines) == 6


class TestEdgeCases:
    def test_constant_objective_does_not_crash(self):
        assert _run(obj=constant_zero, budget=60)["best_value"] == 0.0

    @pytest.mark.parametrize("temp", [0.01, 0.5, 10.0, 100.0])
    def test_extreme_temperatures(self, temp):
        assert _run(budget=20, temp=temp)["best_value"] > -float("inf")

    @pytest.mark.parametrize("lambda_", [1e-6, 0.1, 10.0, 1000.0])
    def test_extreme_lambda(self, lambda_):
        result = BUTChC_optimize(FLAT_SPACE, quadratic, budget=40, lambda_=lambda_,
                                 alpha=1.0, verbose=False, seed=0)
        assert result["best_value"] > -float("inf")

    def test_explore_zero_and_one(self):
        for explore in (0.0, 1.0):
            assert _run(budget=40, explore=explore)["best_value"] > -float("inf")

    def test_warmup_longer_than_budget_is_clamped(self):
        assert len(_run(budget=10, n_warmup=1000)["history"]) == 10

    def test_single_value_categorical(self):
        result = _run(space={"a": {"values": ["only"]}}, obj=constant_zero, budget=20)
        assert result["best_params"]["a"] == "only"

    def test_huge_objective_magnitudes(self):
        # Rank-based weighting must be immune to objective scale.
        result = _run(obj=lambda c: -1e12 * (c["x"] - 1.0) ** 2, budget=200)
        assert abs(result["best_params"]["x"] - 1.0) < 1.0

    def test_large_budget_completes(self):
        assert len(_run(obj=constant_zero, budget=1000)["history"]) == 1000


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------

class TestPriors:
    def test_categorical_prior_survives_updates(self):
        # Regression: setting prob by hand is wiped on the first update,
        # because prob is recomputed every trial. The prior must be stored.
        # It is deliberately diluted by evidence — prior_strength is denominated
        # in trials — so after 60 trials against a strength of 50 it has faded
        # part of the way toward uniform rather than holding at 0.9.
        space = {"opt": {"values": ["a", "b"], "prior": {"a": 0.9, "b": 0.1},
                         "prior_strength": 50}}
        result = BUTChC_optimize(
            space, lambda c: 0.5 if c["opt"] == "a" else 0.5,
            budget=60, verbose=False, seed=0,
        )
        assert result["prob_tree"]["opt"]["prob"]["a"] > 0.7

    def test_prior_biases_which_configs_are_tried(self):
        space = {"opt": {"values": ["a", "b"], "prior": {"a": 0.95, "b": 0.05},
                         "prior_strength": 50}}
        seen = []
        BUTChC_optimize(space, lambda c: seen.append(c["opt"]) or 0.0,
                        budget=200, verbose=False, seed=0)
        assert seen.count("a") > seen.count("b") * 3

    def test_evidence_overrides_a_wrong_prior(self):
        space = {"opt": {"values": ["a", "b"], "prior": {"a": 0.9, "b": 0.1},
                         "prior_strength": 10}}
        result = BUTChC_optimize(
            space, lambda c: 1.0 if c["opt"] == "b" else 0.0,
            budget=600, verbose=False, seed=0,
        )
        assert result["prob_tree"]["opt"]["prob"]["b"] > 0.6

    def test_stronger_prior_resists_evidence_longer(self):
        def final_b(strength):
            space = {"opt": {"values": ["a", "b"], "prior": {"a": 0.9, "b": 0.1},
                             "prior_strength": strength}}
            return BUTChC_optimize(
                space, lambda c: 1.0 if c["opt"] == "b" else 0.0,
                budget=200, verbose=False, seed=0,
            )["prob_tree"]["opt"]["prob"]["b"]

        assert final_b(5) > final_b(400)

    def test_continuous_prior_concentrates_early_trials(self):
        space = {"x": {"min": 0.0, "max": 100.0,
                       "prior": {"mean": 80.0, "std": 2.0}, "prior_strength": 40}}
        seen = []
        BUTChC_optimize(space, lambda c: seen.append(c["x"]) or 0.0,
                        budget=100, verbose=False, seed=0, explore=0.0)
        near = sum(1 for v in seen if 70.0 < v < 90.0)
        # The bar is stated against the uniform baseline it has to beat. The
        # window (70, 90) is a fifth of [0, 100], so uniform sampling scores
        # 0.2 and anything well above that is the prior doing its job. A bare
        # 0.6 was really a statement about how much of the reservoir a prior of
        # strength 40 could occupy, which is capped at
        # KDE_RESERVOIR_SIZE - MIN_GRID_POINTS and so changes when either does.
        uniform_share = 0.2
        assert near > len(seen) * 2 * uniform_share

    def test_correct_prior_speeds_convergence(self):
        space_bare = {"x": {"min": 0.0, "max": 100.0}}
        space_prior = {"x": {"min": 0.0, "max": 100.0},
                       "prior": {"mean": 80.0, "std": 5.0}, "prior_strength": 25}
        space_prior = {"x": dict(space_prior["x"],
                                 prior={"mean": 80.0, "std": 5.0},
                                 prior_strength=25)}

        def obj(c):
            return -abs(c["x"] - 80.0)

        bare = median([BUTChC_optimize(space_bare, obj, budget=40, verbose=False,
                                       seed=s)["best_value"] for s in range(7)])
        primed = median([BUTChC_optimize(space_prior, obj, budget=40, verbose=False,
                                         seed=s)["best_value"] for s in range(7)])
        assert primed > bare

    def test_prior_disables_warmup_by_default(self):
        space = {"opt": {"values": ["a", "b"], "prior": {"a": 0.7, "b": 0.3}}}
        result = BUTChC_optimize(space, lambda c: 1.0 if c["opt"] == "a" else 0.0,
                                 budget=60, verbose=False, seed=0)
        assert result["history"][1]["gated"] or result["history"][2]["gated"]

    def test_explicit_warmup_still_honoured_with_a_prior(self):
        space = {"opt": {"values": ["a", "b"], "prior": {"a": 0.7, "b": 0.3}}}
        result = BUTChC_optimize(space, lambda c: 1.0 if c["opt"] == "a" else 0.0,
                                 budget=60, verbose=False, seed=0, n_warmup=20)
        assert not any(h["gated"] for h in result["history"][:20])

    def test_prior_works_inside_a_conditional_branch(self):
        space = {
            "mode": {
                "values": ["x"],
                "next_level": {
                    "x": {"v": {"min": 0.0, "max": 10.0, "prior": [9.0],
                                "prior_strength": 40}}
                },
            }
        }
        seen = []
        BUTChC_optimize(space, lambda c: seen.append(c["v"]) or 0.0,
                        budget=60, verbose=False, seed=0, explore=0.0)
        assert sum(1 for v in seen if v > 8.0) > len(seen) * 0.5

    @pytest.mark.parametrize(
        "space",
        [
            {"opt": {"values": ["a", "b"], "prior": {"a": 0.5, "b": 0.9}}},
            {"opt": {"values": ["a", "b"], "prior": {"c": 1.0}}},
            {"opt": {"values": ["a", "b"], "prior": {"a": -1.0, "b": 2.0}}},
            {"opt": {"values": ["a", "b"], "prior": [0.5, 0.5]}},
            {"opt": {"values": ["a", "b"], "prior_strength": 5}},
            {"opt": {"values": ["a", "b"], "prior": {"a": 1.0}, "prior_strength": 0}},
            {"x": {"min": 0.0, "max": 1.0, "prior": [5.0]}},
            {"x": {"min": 0.0, "max": 1.0, "prior": []}},
            {"x": {"min": 0.0, "max": 1.0, "prior": {"mean": 0.5}}},
            {"x": {"min": 0.0, "max": 1.0, "prior": {"mean": 0.5, "std": 0.0}}},
            {"x": {"min": 0.0, "max": 1.0, "prior": {"mean": 9.0, "std": 0.1}}},
            {"x": {"min": 0.0, "max": 1.0, "prior": {"mu": 0.5, "std": 0.1}}},
        ],
    )
    def test_malformed_priors_rejected_before_any_evaluation(self, space):
        calls = []
        with pytest.raises(SearchSpaceError):
            BUTChC_optimize(space, lambda c: calls.append(1) or 0.0,
                            budget=5, verbose=False)
        assert calls == []


class TestRegressions040:
    def test_numpy_scalars_are_accepted(self):
        np = pytest.importorskip("numpy")
        for dtype in (np.float64, np.float32, np.int64):
            result = BUTChC_optimize(
                {"x": {"min": 0.0, "max": 1.0}},
                lambda c, d=dtype: d(c["x"]),
                budget=5, verbose=False, seed=0,
            )
            assert isinstance(result["best_value"], float)

    @pytest.mark.parametrize("bad", [True, "0.5", None, [1.0]])
    def test_non_numeric_returns_still_rejected(self, bad):
        with pytest.raises(TypeError):
            BUTChC_optimize({"x": {"min": 0.0, "max": 1.0}},
                            lambda c: bad, budget=2, verbose=False, seed=0)

    def test_warm_start_rejects_a_different_range(self):
        trained = BUTChC_optimize(
            {"x": {"min": 0.0, "max": 1.0}}, lambda c: -abs(c["x"] - 0.9),
            budget=40, verbose=False, seed=0,
        )
        with pytest.raises(ValueError, match="bounds differ"):
            BUTChC_optimize(
                {"x": {"min": 100.0, "max": 200.0}}, lambda c: -c["x"],
                budget=5, verbose=False, seed=0,
                start_prob_tree=trained["prob_tree"],
            )

    def test_warm_start_rejects_a_different_scale(self):
        trained = BUTChC_optimize(
            {"x": {"min": 1.0, "max": 100.0}}, lambda c: -c["x"],
            budget=20, verbose=False, seed=0,
        )
        with pytest.raises(ValueError, match="'log' is"):
            BUTChC_optimize(
                {"x": {"min": 1.0, "max": 100.0, "log": True}}, lambda c: -c["x"],
                budget=5, verbose=False, seed=0,
                start_prob_tree=trained["prob_tree"],
            )

    def test_tied_objective_still_teaches_the_tree(self):
        # Ranking by strict inequality put a trial below everything it tied
        # with, so on a two-valued objective the gate rejected nearly all of
        # them and the tree learned essentially nothing.
        result = BUTChC_optimize(
            {"opt": {"values": ["a", "b"]}},
            lambda c: 1.0 if c["opt"] == "b" else 0.0,
            budget=300, verbose=False, seed=0,
        )
        assert result["prob_tree"]["opt"]["prob"]["b"] > 0.9

    def test_a_dominant_choice_is_not_forgotten(self):
        # The gate is self-referential: once a choice takes most of the budget
        # it stops clearing the quantile it is winning, and with discounting
        # its evidence decayed to nothing and the node drifted to uniform.
        counts = {"a": 0, "b": 0}

        def objective(config):
            counts[config["opt"]] += 1
            return 1.0 if config["opt"] == "b" else 0.0

        result = BUTChC_optimize({"opt": {"values": ["a", "b"]}}, objective,
                                 budget=800, verbose=False, seed=0)
        assert counts["b"] > counts["a"] * 4
        assert result["prob_tree"]["opt"]["prob"]["b"] > 0.9

    def test_a_wide_branch_is_not_abandoned_before_it_is_tuned(self):
        # The branch holding the global optimum has the worst average, because
        # its sub-space is wide. v0.3 committed to the safe branch and finished
        # below random search.
        space = {"model": {
            "values": ["wide", "safe"],
            "next_level": {
                "wide": {"a": {"min": -5.0, "max": 5.0},
                         "b": {"min": -5.0, "max": 5.0}},
                "safe": {"c": {"min": -5.0, "max": 5.0}},
            },
        }}

        def objective(config):
            if config["model"] == "wide":
                return -(config["a"] - 2.0) ** 2 - (config["b"] + 1.0) ** 2
            return -abs(config["c"]) - 1.0

        best = [BUTChC_optimize(space, objective, budget=300, verbose=False,
                                seed=s)["best_value"] for s in range(5)]
        assert statistics.median(best) > -0.5
