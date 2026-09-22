"""Unit tests for butchc._update.

Several of these are direct regressions against defects in v0.1:
the objective being ignored entirely, and the reservoir never accumulating.
"""

import math
import random

import pytest

from butchc._sampling import traverse_sample
from butchc._tree import initialize_prob_tree
from butchc._update import quality_weight, update_tree
from butchc._utils import KDE_RESERVOIR_SIZE, effective_sample_size, weighted_mean


@pytest.fixture
def rng():
    return random.Random(0)


def trace_for(tree, values):
    """Build a trace by hand for deterministic update tests."""
    return {k: {"internal": v, "sub": None} for k, v in values.items()}


class TestQualityWeight:
    def test_below_gate_is_zero(self):
        assert quality_weight(0.3, gamma=0.5) == 0.0

    def test_at_gate_is_zero(self):
        assert quality_weight(0.5, gamma=0.5) == 0.0

    def test_best_rank_is_one(self):
        assert quality_weight(1.0, gamma=0.5) == pytest.approx(1.0)

    def test_monotonic_above_gate(self):
        weights = [quality_weight(r / 10, 0.5) for r in range(6, 11)]
        assert weights == sorted(weights)

    def test_gamma_zero_keeps_everything_above_worst(self):
        assert quality_weight(0.1, gamma=0.0) == pytest.approx(0.1)

    def test_bounded_in_unit_interval(self):
        for r in (0.0, 0.25, 0.5, 0.75, 1.0):
            assert 0.0 <= quality_weight(r, 0.4) <= 1.0


class TestCategoricalUpdate:
    def test_chosen_value_gains_probability(self, rng):
        tree = initialize_prob_tree({"a": {"values": ["x", "y"]}})
        before = tree["a"]["prob"]["x"]
        update_tree(tree, trace_for(tree, {"a": "x"}), 1.0, 1.0, 1.0, 5.0, rng)
        assert tree["a"]["prob"]["x"] > before

    def test_unchosen_value_loses_probability(self, rng):
        tree = initialize_prob_tree({"a": {"values": ["x", "y"]}})
        before = tree["a"]["prob"]["y"]
        update_tree(tree, trace_for(tree, {"a": "x"}), 1.0, 1.0, 1.0, 5.0, rng)
        assert tree["a"]["prob"]["y"] < before

    def test_probabilities_stay_normalised(self, rng):
        tree = initialize_prob_tree({"a": {"values": ["x", "y", "z"]}})
        for _ in range(50):
            update_tree(tree, trace_for(tree, {"a": "z"}), 1.0, 1.0, 0.7, 1.0, rng)
        assert sum(tree["a"]["prob"].values()) == pytest.approx(1.0)

    def test_zero_quality_still_reports_categorical_movement(self, rng):
        # The node updates whether or not the gate passed, so the delta has to
        # be reported or the loss understates how far the tree moved.
        tree = initialize_prob_tree({"a": {"values": ["x", "y"]}})
        deltas = update_tree(tree, trace_for(tree, {"a": "x"}), 1.0, 1.0, 0.0, 5.0, rng)
        assert len(deltas) == 1
        assert deltas[0] > 0.0

    def test_zero_quality_reports_no_continuous_movement(self, rng):
        tree = initialize_prob_tree({"x": {"min": 0.0, "max": 1.0}})
        deltas = update_tree(tree, trace_for(tree, {"x": 0.42}), 1.0, 1.0, 0.0, 5.0, rng)
        assert deltas == []

    def test_zero_quality_still_counts_the_visit(self, rng):
        # Choices are scored by mean quality, so a trial that failed the gate
        # is evidence against its branch and has to be counted.
        tree = initialize_prob_tree({"a": {"values": ["x", "y"]}})
        before = tree["a"]["prob"]["x"]
        update_tree(tree, trace_for(tree, {"a": "x"}), 1.0, 1.0, 0.0, 5.0, rng)
        assert tree["a"]["visits"]["x"] == 1.0
        assert tree["a"]["prob"]["x"] < before

    def test_zero_quality_leaves_the_archive_untouched(self, rng):
        tree = initialize_prob_tree({"x": {"min": 0.0, "max": 1.0}})
        snapshot = list(tree["x"]["reservoir"])
        update_tree(tree, trace_for(tree, {"x": 0.42}), 1.0, 1.0, 0.0, 5.0, rng)
        assert tree["x"]["reservoir"] == snapshot

    def test_higher_quality_moves_further(self, rng):
        def move(quality):
            tree = initialize_prob_tree({"a": {"values": ["x", "y"]}})
            update_tree(tree, trace_for(tree, {"a": "x"}), 1.0, 1.0, quality, 1.0, rng)
            return tree["a"]["prob"]["x"]

        assert move(1.0) > move(0.2)

    def test_larger_alpha_keeps_distribution_flatter(self, rng):
        def prob_after(alpha):
            tree = initialize_prob_tree({"a": {"values": ["x", "y"]}})
            for _ in range(20):
                update_tree(tree, trace_for(tree, {"a": "x"}), 1.0, alpha, 1.0, 1.0, rng)
            return tree["a"]["prob"]["x"]

        assert prob_after(20.0) < prob_after(0.1)

    def test_recurses_into_chosen_branch_only(self, rng):
        space = {
            "opt": {
                "values": ["a", "b"],
                "next_level": {
                    "a": {"p": {"min": 0.0, "max": 1.0}},
                    "b": {"q": {"min": 0.0, "max": 1.0}},
                },
            }
        }
        tree = initialize_prob_tree(space)
        untouched = list(tree["opt"]["next_level"]["b"]["q"]["reservoir"])
        trace = {"opt": {"internal": "a", "sub": trace_for({}, {"p": 0.5})}}
        update_tree(tree, trace, 1.0, 1.0, 1.0, 5.0, rng)
        assert tree["opt"]["next_level"]["b"]["q"]["reservoir"] == untouched
        assert len(tree["opt"]["next_level"]["a"]["p"]["reservoir"]) == KDE_RESERVOIR_SIZE


class TestContinuousUpdate:
    def test_observation_is_retained_not_immediately_pruned(self, rng):
        # v0.1 regression: a new point entered below the weight of every
        # prior grid point, so it was evicted on the same call.
        tree = initialize_prob_tree({"x": {"min": 0.0, "max": 1.0}})
        update_tree(tree, trace_for(tree, {"x": 0.42}), 1.0, 1.0, 1.0, 9.0, rng)
        assert 0.42 in tree["x"]["reservoir"]

    def test_reservoir_never_exceeds_capacity(self, rng):
        tree = initialize_prob_tree({"x": {"min": 0.0, "max": 1.0}})
        for i in range(300):
            update_tree(tree, trace_for(tree, {"x": i / 300}), 1.0, 1.0, 1.0, float(i), rng)
        assert len(tree["x"]["reservoir"]) == KDE_RESERVOIR_SIZE
        assert len(tree["x"]["weights"]) == KDE_RESERVOIR_SIZE
        assert len(tree["x"]["scores"]) == KDE_RESERVOIR_SIZE

    def test_weights_stay_normalised(self, rng):
        tree = initialize_prob_tree({"x": {"min": 0.0, "max": 1.0}})
        for i in range(100):
            update_tree(tree, trace_for(tree, {"x": 0.3}), 1.0, 1.0, 1.0, float(i), rng)
        assert sum(tree["x"]["weights"]) == pytest.approx(1.0)

    def test_archive_keeps_high_scorers_and_drops_low(self, rng):
        tree = initialize_prob_tree({"x": {"min": 0.0, "max": 1.0}})
        for _ in range(60):
            update_tree(tree, trace_for(tree, {"x": 0.9}), 1.0, 1.0, 1.0, 100.0, rng)
        for _ in range(60):
            update_tree(tree, trace_for(tree, {"x": 0.1}), 1.0, 1.0, 1.0, -100.0, rng)
        # The good region should still dominate despite the later flood.
        assert weighted_mean(tree["x"]["reservoir"], tree["x"]["weights"]) > 0.7

    def test_reservoir_mean_tracks_good_observations(self, rng):
        tree = initialize_prob_tree({"x": {"min": -5.0, "max": 5.0}})
        for i in range(120):
            update_tree(tree, trace_for(tree, {"x": 3.0}), 1.0, 1.0, 1.0, float(i), rng)
        assert weighted_mean(tree["x"]["reservoir"], tree["x"]["weights"]) == pytest.approx(
            3.0, abs=0.3
        )

    def test_effective_sample_size_stays_healthy(self, rng):
        # v0.1 regression: a single observation took ~50% of the weight,
        # collapsing effective sample size to about an eighth of the archive.
        # The bar is a fraction of the archive, not a count: the failure being
        # guarded against is "the weights collapsed onto a handful of points",
        # which is proportional, and a literal count silently becomes a
        # different test whenever KDE_RESERVOIR_SIZE is retuned.
        tree = initialize_prob_tree({"x": {"min": 0.0, "max": 1.0}})
        for i in range(200):
            update_tree(
                tree, trace_for(tree, {"x": rng.random()}), 1.0, 1.0, 1.0, rng.random(), rng
            )
        ess = effective_sample_size(tree["x"]["weights"])
        assert ess > 0.4 * KDE_RESERVOIR_SIZE

    def test_prior_grid_is_displaced_by_real_observations(self, rng):
        tree = initialize_prob_tree({"x": {"min": 0.0, "max": 1.0}})
        for _ in range(KDE_RESERVOIR_SIZE):
            update_tree(tree, trace_for(tree, {"x": 0.5}), 1.0, 1.0, 1.0, 1.0, rng)
        assert all(s > -math.inf for s in tree["x"]["scores"])

    def test_larger_lambda_sharpens_weighting(self, rng):
        def spread(lambda_):
            tree = initialize_prob_tree({"x": {"min": 0.0, "max": 1.0}})
            for i in range(80):
                update_tree(
                    tree, trace_for(tree, {"x": i / 80}), lambda_, 1.0, 1.0, float(i), rng
                )
            return effective_sample_size(tree["x"]["weights"])

        assert spread(3.0) < spread(0.3)


class TestDeltas:
    def test_deltas_are_bounded(self, rng):
        tree = initialize_prob_tree({"a": {"values": ["x", "y"]}, "n": {"min": 0.0, "max": 1.0}})
        deltas = update_tree(
            tree, trace_for(tree, {"a": "x", "n": 0.5}), 1.0, 1.0, 1.0, 3.0, rng
        )
        assert all(0.0 <= d <= 1.0 for d in deltas)

    def test_one_delta_per_updated_node(self, rng):
        tree = initialize_prob_tree({"a": {"values": ["x"]}, "n": {"min": 0.0, "max": 1.0}})
        deltas = update_tree(
            tree, trace_for(tree, {"a": "x", "n": 0.5}), 1.0, 1.0, 1.0, 3.0, rng
        )
        assert len(deltas) == 2

    def test_deltas_shrink_as_the_tree_settles(self, rng):
        tree = initialize_prob_tree({"a": {"values": ["x", "y"]}})
        first = update_tree(tree, trace_for(tree, {"a": "x"}), 1.0, 1.0, 1.0, 1.0, rng)[0]
        for _ in range(200):
            update_tree(tree, trace_for(tree, {"a": "x"}), 1.0, 1.0, 1.0, 1.0, rng)
        last = update_tree(tree, trace_for(tree, {"a": "x"}), 1.0, 1.0, 1.0, 1.0, rng)[0]
        assert last < first


class TestNameCollisionSafety:
    def test_update_follows_the_trace_not_parameter_names(self, rng):
        # Updates key off tree structure, so a repeated name in a nested
        # branch cannot corrupt an unrelated node. (The search-space
        # validator rejects such spaces outright; this guards the mechanism.)
        space = {
            "outer": {"values": ["p", "q"], "next_level": {"p": {"inner": {"values": ["p", "q"]}}}}
        }
        tree = initialize_prob_tree(space)
        trace = {"outer": {"internal": "p", "sub": {"inner": {"internal": "q", "sub": None}}}}
        update_tree(tree, trace, 1.0, 1.0, 1.0, 1.0, rng)
        assert tree["outer"]["counts"]["p"] > 0
        assert tree["outer"]["counts"]["q"] == 0
        assert tree["outer"]["next_level"]["p"]["inner"]["counts"]["q"] > 0
