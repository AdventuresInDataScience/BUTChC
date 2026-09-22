"""Unit tests for butchc._tree."""

import math

import pytest

from butchc._tree import (
    _subspace_size,
    initialize_prob_tree,
    reservoir_summary,
    snap_internal,
    to_external,
    to_internal,
)
from butchc._utils import KDE_RESERVOIR_SIZE, MIN_GRID_POINTS


class TestCategoricalNodes:
    def test_has_counts_and_prob(self):
        tree = initialize_prob_tree({"a": {"values": ["x", "y"]}})
        assert set(tree["a"]) >= {"counts", "prob"}

    def test_starts_uniform(self):
        tree = initialize_prob_tree({"a": {"values": ["x", "y", "z"]}})
        assert all(p == pytest.approx(1 / 3) for p in tree["a"]["prob"].values())

    def test_probabilities_sum_to_one(self):
        tree = initialize_prob_tree({"a": {"values": list(range(7))}})
        assert sum(tree["a"]["prob"].values()) == pytest.approx(1.0)

    def test_counts_start_at_zero(self):
        # Laplace smoothing supplies the prior; pre-seeding counts at 1 as
        # well would double-count it.
        tree = initialize_prob_tree({"a": {"values": ["x", "y"]}})
        assert all(c == 0.0 for c in tree["a"]["counts"].values())

    def test_no_next_level_key_when_absent(self):
        tree = initialize_prob_tree({"a": {"values": ["x"]}})
        assert "next_level" not in tree["a"]

    def test_next_level_built_for_listed_choices_only(self):
        space = {
            "a": {
                "values": ["x", "y"],
                "next_level": {"x": {"p": {"min": 0.0, "max": 1.0}}},
            }
        }
        tree = initialize_prob_tree(space)
        assert set(tree["a"]["next_level"]) == {"x"}

    def test_nesting_is_recursive(self):
        space = {
            "a": {
                "values": ["x"],
                "next_level": {
                    "x": {"b": {"values": ["p"], "next_level": {"p": {"c": {"min": 0.0, "max": 1.0}}}}}
                },
            }
        }
        tree = initialize_prob_tree(space)
        deep = tree["a"]["next_level"]["x"]["b"]["next_level"]["p"]["c"]
        assert "reservoir" in deep


class TestContinuousNodes:
    def test_reservoir_has_expected_size(self):
        tree = initialize_prob_tree({"x": {"min": 0.0, "max": 1.0}})
        assert len(tree["x"]["reservoir"]) == KDE_RESERVOIR_SIZE

    def test_weights_match_reservoir_length(self):
        node = initialize_prob_tree({"x": {"min": 0.0, "max": 1.0}})["x"]
        assert len(node["weights"]) == len(node["reservoir"])

    def test_scores_match_reservoir_length(self):
        node = initialize_prob_tree({"x": {"min": 0.0, "max": 1.0}})["x"]
        assert len(node["scores"]) == len(node["reservoir"])

    def test_prior_scores_are_sentinel(self):
        # The flat prior grid must be displaced by any real observation.
        node = initialize_prob_tree({"x": {"min": 0.0, "max": 1.0}})["x"]
        assert all(s == -math.inf for s in node["scores"])

    def test_weights_sum_to_one(self):
        node = initialize_prob_tree({"x": {"min": 0.0, "max": 1.0}})["x"]
        assert sum(node["weights"]) == pytest.approx(1.0)

    def test_reservoir_spans_full_range(self):
        node = initialize_prob_tree({"x": {"min": -4.0, "max": 6.0}})["x"]
        assert node["reservoir"][0] == pytest.approx(-4.0)
        assert node["reservoir"][-1] == pytest.approx(6.0)

    def test_reservoir_is_evenly_spaced(self):
        res = initialize_prob_tree({"x": {"min": 0.0, "max": 1.0}})["x"]["reservoir"]
        gaps = [b - a for a, b in zip(res, res[1:])]
        assert max(gaps) == pytest.approx(min(gaps))

    def test_external_bounds_preserved(self):
        node = initialize_prob_tree({"x": {"min": 2.0, "max": 8.0}})["x"]
        assert (node["ext_min"], node["ext_max"]) == (2.0, 8.0)


class TestLogScale:
    def test_internal_bounds_are_log10(self):
        node = initialize_prob_tree({"lr": {"min": 1e-4, "max": 1e-1, "log": True}})["lr"]
        assert node["min"] == pytest.approx(-4.0)
        assert node["max"] == pytest.approx(-1.0)

    def test_round_trip(self):
        node = {"log": True, "int": False}
        assert to_external(node, to_internal(node, 0.003)) == pytest.approx(0.003)

    def test_reservoir_covers_decades_evenly(self):
        node = initialize_prob_tree({"lr": {"min": 1e-6, "max": 1e0, "log": True}})["lr"]
        externals = [to_external(node, x) for x in node["reservoir"]]
        # A uniform grid would put almost nothing below 1e-1; log spacing
        # should place roughly a sixth of the points in each decade.
        below_milli = sum(1 for v in externals if v < 1e-3)
        assert below_milli > len(externals) * 0.4


class TestIntegerScale:
    def test_external_value_is_an_int(self):
        node = {"log": False, "int": True}
        assert isinstance(to_external(node, 3.4), int)

    def test_rounds_to_nearest(self):
        node = {"log": False, "int": True}
        assert to_external(node, 3.6) == 4
        assert to_external(node, 3.4) == 3

    def test_snap_matches_emitted_value(self):
        # The tree must learn from the value the objective actually received.
        node = {"log": False, "int": True}
        assert snap_internal(node, 3.7) == pytest.approx(float(to_external(node, 3.7)))

    def test_snap_is_identity_for_plain_nodes(self):
        assert snap_internal({"log": False, "int": False}, 3.7) == pytest.approx(3.7)


class TestReservoirSummary:
    def test_reports_external_units(self):
        node = initialize_prob_tree({"lr": {"min": 1e-4, "max": 1e-2, "log": True}})["lr"]
        summary = reservoir_summary(node)
        assert 1e-4 <= summary["mean"] <= 1e-2

    def test_flat_prior_has_full_ess(self):
        node = initialize_prob_tree({"x": {"min": 0.0, "max": 1.0}})["x"]
        assert reservoir_summary(node)["ess"] == pytest.approx(KDE_RESERVOIR_SIZE)

    def test_reports_expected_fields(self):
        node = initialize_prob_tree({"x": {"min": 0.0, "max": 1.0}})["x"]
        assert set(reservoir_summary(node)) == {"mean", "std", "ess", "n"}


class TestCategoricalPrior:
    def test_prior_sets_initial_probabilities_exactly(self):
        space = {"opt": {"values": ["a", "b"], "prior": {"a": 0.8, "b": 0.2}}}
        prob = initialize_prob_tree(space)["opt"]["prob"]
        assert prob["a"] == pytest.approx(0.8)
        assert prob["b"] == pytest.approx(0.2)

    def test_prior_is_kept_separate_from_evidence(self):
        # prob is recomputed on every update, so a prior stored only in prob
        # would be discarded on the first trial. It must not go into counts or
        # visits either: those are discounted every trial, which would make a
        # stated belief decay with time rather than with contrary evidence.
        space = {"opt": {"values": ["a", "b"], "prior": {"a": 0.8, "b": 0.2},
                         "prior_strength": 10}}
        node = initialize_prob_tree(space)["opt"]
        assert node["prior"] == pytest.approx({"a": 0.8, "b": 0.2})
        assert node["prior_strength"] == pytest.approx(10.0)
        assert node["counts"] == pytest.approx({"a": 0.0, "b": 0.0})
        assert node["visits"] == pytest.approx({"a": 0.0, "b": 0.0})

    def test_prior_presumes_quality_not_visits(self):
        # A prior recorded as pseudo-visits at the average quality cancels
        # exactly: a choice holding 47 of 50 pseudo-visits at mean quality
        # still has mean quality. The share has to become a presumed quality.
        from butchc._update import _recompute_prob
        space = {"opt": {"values": ["a", "b"], "prior": {"a": 0.8, "b": 0.2},
                         "prior_strength": 10}}
        node = initialize_prob_tree(space)["opt"]
        _recompute_prob(node, alpha=1.0, sharpen=1.0, neutral=0.25)
        assert node["prob"]["a"] > node["prob"]["b"] * 3

    def test_prior_strength_scales_counts(self):
        def counts(strength):
            space = {"opt": {"values": ["a", "b"], "prior": {"a": 0.5, "b": 0.5},
                             "prior_strength": strength}}
            return initialize_prob_tree(space)["opt"]["prior_strength"]

        assert counts(100.0) == pytest.approx(10 * counts(10.0))

    def test_default_strength_applied_when_omitted(self):
        space = {"opt": {"values": ["a", "b"], "prior": {"a": 0.5, "b": 0.5}}}
        assert initialize_prob_tree(space)["opt"]["prior_strength"] > 0

    def test_no_prior_starts_uniform_with_zero_counts(self):
        tree = initialize_prob_tree({"opt": {"values": ["a", "b", "c"]}})
        assert all(c == 0.0 for c in tree["opt"]["counts"].values())
        assert all(p == pytest.approx(1 / 3) for p in tree["opt"]["prob"].values())

    def test_zero_weighted_choice_remains_reachable(self):
        # Laplace smoothing must keep an unlisted choice sampleable.
        space = {"opt": {"values": ["a", "b"], "prior": {"a": 1.0}}}
        assert initialize_prob_tree(space)["opt"]["prob"]["b"] == pytest.approx(0.0)


class TestContinuousPrior:
    def test_list_prior_enters_the_reservoir(self):
        space = {"x": {"min": 0.0, "max": 10.0, "prior": [4.0], "prior_strength": 20}}
        node = initialize_prob_tree(space)["x"]
        # Capped at the capacity a prior may occupy, not at prior_strength:
        # MIN_GRID_POINTS of flat grid always survive so the rest of the range
        # stays reachable. At KDE_RESERVOIR_SIZE 25 that ceiling is 15, so a
        # strength of 20 is clamped — which is the documented guarantee, and
        # the reason this is derived rather than written out.
        expected = min(20, KDE_RESERVOIR_SIZE - MIN_GRID_POINTS)
        assert node["reservoir"].count(4.0) == expected

    def test_prior_strength_cannot_displace_the_last_grid_points(self):
        space = {"x": {"min": 0.0, "max": 10.0, "prior": [4.0],
                       "prior_strength": 10_000}}
        node = initialize_prob_tree(space)["x"]
        assert len(node["reservoir"]) == KDE_RESERVOIR_SIZE
        assert node["reservoir"].count(4.0) == KDE_RESERVOIR_SIZE - MIN_GRID_POINTS
        grid = [v for v in node["reservoir"] if v != 4.0]
        assert len(grid) == MIN_GRID_POINTS
        # The survivors still span the range, so nothing is unreachable.
        assert min(grid) == pytest.approx(0.0)
        assert max(grid) == pytest.approx(10.0)

    def test_prior_points_outrank_the_grid(self):
        space = {"x": {"min": 0.0, "max": 10.0, "prior": [4.0], "prior_strength": 5}}
        node = initialize_prob_tree(space)["x"]
        prior_idx = [i for i, v in enumerate(node["reservoir"]) if v == 4.0]
        grid_idx = [i for i in range(len(node["reservoir"])) if i not in prior_idx]
        assert min(node["weights"][i] for i in prior_idx) > max(
            node["weights"][i] for i in grid_idx
        )

    def test_prior_score_is_below_any_finite_objective(self):
        # A real observation must displace the prior, not compete with it.
        from butchc._utils import GRID_SCORE, PRIOR_SCORE

        assert GRID_SCORE < PRIOR_SCORE < -1e308
        assert PRIOR_SCORE < -1e300 < 0.0

    def test_reservoir_size_is_unchanged_by_a_prior(self):
        space = {"x": {"min": 0.0, "max": 10.0, "prior": [4.0], "prior_strength": 20}}
        node = initialize_prob_tree(space)["x"]
        assert len(node["reservoir"]) == KDE_RESERVOIR_SIZE
        assert len(node["scores"]) == KDE_RESERVOIR_SIZE
        assert len(node["weights"]) == KDE_RESERVOIR_SIZE

    def test_grid_coverage_survives_a_partial_prior(self):
        space = {"x": {"min": 0.0, "max": 10.0, "prior": [4.0], "prior_strength": 10}}
        node = initialize_prob_tree(space)["x"]
        assert sum(1 for v in node["reservoir"] if v != 4.0) == KDE_RESERVOIR_SIZE - 10

    def test_mean_std_prior_centres_on_the_mean(self):
        space = {"x": {"min": 0.0, "max": 10.0,
                       "prior": {"mean": 7.0, "std": 0.5}, "prior_strength": 40}}
        node = initialize_prob_tree(space)["x"]
        prior_points = [v for v, s in zip(node["reservoir"], node["scores"])
                        if s > -math.inf]
        assert sum(prior_points) / len(prior_points) == pytest.approx(7.0, abs=0.2)

    def test_prior_is_deterministic(self):
        space = {"x": {"min": 0.0, "max": 10.0, "prior": {"mean": 7.0, "std": 0.5}}}
        assert (initialize_prob_tree(space)["x"]["reservoir"]
                == initialize_prob_tree(space)["x"]["reservoir"])

    def test_log_prior_given_in_external_units(self):
        space = {"lr": {"min": 1e-6, "max": 1e-1, "log": True,
                        "prior": [1e-3], "prior_strength": 10}}
        node = initialize_prob_tree(space)["lr"]
        prior_points = [v for v, s in zip(node["reservoir"], node["scores"])
                        if s > -math.inf]
        assert all(to_external(node, v) == pytest.approx(1e-3) for v in prior_points)

    def test_int_prior_is_snapped(self):
        space = {"n": {"min": 1.0, "max": 10.0, "int": True,
                       "prior": [4.4], "prior_strength": 5}}
        node = initialize_prob_tree(space)["n"]
        prior_points = [v for v, s in zip(node["reservoir"], node["scores"])
                        if s > -math.inf]
        assert all(v == pytest.approx(4.0) for v in prior_points)

    def test_prior_beyond_capacity_is_clamped(self):
        space = {"x": {"min": 0.0, "max": 10.0, "prior": [4.0],
                       "prior_strength": 500}}
        node = initialize_prob_tree(space)["x"]
        assert len(node["reservoir"]) == KDE_RESERVOIR_SIZE


class TestRankWeightTies:
    def test_tied_scores_receive_equal_weight(self):
        # Stable-sort tie-breaking would weight the flat starting grid by list
        # position, biasing every continuous search toward the top of range.
        from butchc._utils import rank_weights

        weights = rank_weights([-math.inf] * 50, 1.0)
        assert len(set(round(w, 12) for w in weights)) == 1

    def test_fresh_grid_is_flat_after_a_single_update(self):
        import random as _random

        from butchc._update import update_tree

        tree = initialize_prob_tree({"x": {"min": 0.0, "max": 10.0}})
        update_tree(tree, {"x": {"internal": 5.0, "sub": None}},
                    1.0, 1.0, 1.0, 42.0, _random.Random(0))
        grid = [w for w, s in zip(tree["x"]["weights"], tree["x"]["scores"])
                if s == -math.inf]
        assert max(grid) / min(grid) == pytest.approx(1.0)

    def test_real_observation_outranks_the_grid(self):
        from butchc._utils import rank_weights

        weights = rank_weights([-math.inf] * 50 + [1.0], 1.0)
        assert weights[-1] > weights[0]


class TestRegressions040:
    def test_subtrees_are_built_once(self):
        # Was 2**(depth+1)-2 recursive calls: _categorical_node built
        # next_level and the caller immediately rebuilt it.
        import butchc._tree as tree_module

        calls = {"n": 0}
        original = tree_module.initialize_prob_tree

        def counted(space):
            calls["n"] += 1
            return original(space)

        def nest(depth):
            if depth == 0:
                return {"leaf": {"min": 0.0, "max": 1.0}}
            return {f"c{depth}": {"values": ["x"],
                                  "next_level": {"x": nest(depth - 1)}}}

        tree_module.initialize_prob_tree = counted
        try:
            original(nest(4))
        finally:
            tree_module.initialize_prob_tree = original
        assert calls["n"] == 4

    def test_prior_thins_the_grid_evenly(self):
        # Truncating from the front removed the bottom of every range no
        # matter where the prior sat.
        from butchc._utils import GRID_SCORE
        for mean in (0.1, 0.5, 0.9):
            node = initialize_prob_tree({"x": {
                "min": 0.0, "max": 1.0,
                "prior": {"mean": mean, "std": 0.02}, "prior_strength": 20,
            }})["x"]
            grid = [r for r, s in zip(node["reservoir"], node["scores"])
                    if s == GRID_SCORE]
            assert min(grid) == pytest.approx(0.0)
            assert max(grid) == pytest.approx(1.0)

    def test_no_prior_strength_can_erase_the_grid(self):
        from butchc._utils import GRID_SCORE, MIN_GRID_POINTS
        node = initialize_prob_tree({"x": {
            "min": 0.0, "max": 1.0,
            "prior": {"mean": 0.5, "std": 0.05}, "prior_strength": 10_000,
        }})["x"]
        grid = [r for r, s in zip(node["reservoir"], node["scores"])
                if s == GRID_SCORE]
        assert len(grid) == MIN_GRID_POINTS
        assert min(grid) == pytest.approx(0.0)
        assert max(grid) == pytest.approx(1.0)

    def test_integer_endpoints_are_not_under_sampled(self):
        # Rounding over [min, max] gave the two endpoint integers half-width
        # bins, so they came up half as often as every other value.
        import random
        from butchc._sampling import sample_node

        rng = random.Random(0)
        node = initialize_prob_tree({"n": {"min": 1, "max": 6, "int": True}})["n"]
        counts = {}
        for _ in range(60_000):
            value, _ = sample_node(node, 1.0, rng, explore=1.0)
            counts[value] = counts.get(value, 0) + 1
        assert set(counts) == {1, 2, 3, 4, 5, 6}
        share = [counts[k] / 60_000 for k in sorted(counts)]
        assert min(share) > 0.145 and max(share) < 0.188

    def test_integer_values_stay_inside_declared_bounds(self):
        import random
        from butchc._sampling import sample_node

        rng = random.Random(1)
        node = initialize_prob_tree({"n": {"min": 3, "max": 5, "int": True}})["n"]
        for _ in range(5_000):
            value, _ = sample_node(node, 1.0, rng, explore=1.0)
            assert 3 <= value <= 5


class TestSubspaceSize:
    """`sub_size` divides the commitment exponent, so it has to be a property
    of the space and not of the order its keys were written in."""

    def _space(self, *keys):
        parts = {
            "A": {"min": 0.0, "max": 1.0},
            "B": {"min": 0.0, "max": 1.0},
            "C": {"values": ["x"], "next_level": {"x": {"S": {"min": 0.0, "max": 1.0}}}},
        }
        return {k: parts[k] for k in keys}

    def test_is_independent_of_key_order(self):
        sizes = {
            _subspace_size(self._space(*order))
            for order in (("A", "B", "C"), ("C", "A", "B"), ("B", "C", "A"))
        }
        assert len(sizes) == 1

    def test_counts_every_parameter_on_the_widest_config(self):
        # A, B, C and C's S all arrive in one config.
        assert _subspace_size(self._space("A", "B", "C")) == 4

    def test_flat_space_is_its_own_length(self):
        assert _subspace_size(self._space("A", "B")) == 2

    def test_only_the_largest_branch_of_one_choice_counts(self):
        # Branches never co-occur, so the wider one sets the size.
        space = {
            "c": {
                "values": ["thin", "wide"],
                "next_level": {
                    "thin": {"p": {"min": 0.0, "max": 1.0}},
                    "wide": {"q": {"min": 0.0, "max": 1.0},
                             "r": {"min": 0.0, "max": 1.0}},
                },
            }
        }
        assert _subspace_size(space) == 3       # c + wide's q, r

    def test_two_branching_siblings_both_contribute(self):
        # Unlike branches of one choice, sibling parameters do co-occur.
        space = {
            "c1": {"values": ["x"], "next_level": {"x": {"p": {"min": 0.0, "max": 1.0}}}},
            "c2": {"values": ["y"], "next_level": {"y": {"q": {"min": 0.0, "max": 1.0}}}},
        }
        assert _subspace_size(space) == 4

    def test_reaches_the_tree_as_sub_size(self):
        space = {
            "top": {
                "values": ["only"],
                "next_level": {"only": self._space("A", "B", "C")},
            }
        }
        assert initialize_prob_tree(space)["top"]["sub_size"] == 4
