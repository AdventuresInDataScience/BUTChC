"""Tests for `prune`.

The interesting failures are all one-directional. Pruning too little wastes
budget; pruning away the branch holding the best configuration loses the
answer, and nothing downstream can recover it. Most of what follows is
therefore about what `prune` refuses to remove.
"""

import pytest

from butchc import BUTChC_optimize, SearchSpaceError, prune, prune_report
from butchc._validate import check_tree_matches_searchspace, validate_prob_tree

SPACE = {
    "model": {
        "values": ["a", "b", "c", "d"],
        "next_level": {c: {"x": {"min": 0.0, "max": 1.0}} for c in "abcd"},
    },
    "z": {"min": 0.0, "max": 10.0},
    "n": {"min": 1, "max": 200, "int": True},
}

BONUS = {"a": 1.0, "b": 0.2, "c": 0.0, "d": -0.5}


def objective(config):
    return (-((config["x"] - 0.3) ** 2)
            + BONUS[config["model"]]
            - ((config["z"] - 7.0) ** 2) / 100.0)


@pytest.fixture
def finished():
    return BUTChC_optimize(SPACE, objective, 200, verbose=False, seed=1)


class TestPruning:
    def test_drops_the_abandoned_branch(self, finished):
        smaller = prune(SPACE, finished)
        assert "d" not in smaller["model"]["values"]
        assert "a" in smaller["model"]["values"]

    def test_output_is_a_valid_searchspace(self, finished):
        smaller = prune(SPACE, finished)
        result = BUTChC_optimize(smaller, objective, 40, verbose=False, seed=2)
        assert result["best_params"]["model"] in smaller["model"]["values"]

    def test_input_is_not_modified(self, finished):
        before = {k: dict(v) for k, v in SPACE.items()}
        prune(SPACE, finished, narrow=2.0)
        assert SPACE["model"]["values"] == before["model"]["values"]
        assert SPACE["z"]["min"] == before["z"]["min"]

    def test_next_level_follows_the_surviving_choices(self, finished):
        smaller = prune(SPACE, finished)
        assert set(smaller["model"]["next_level"]) == set(
            smaller["model"]["values"])

    def test_threshold_zero_drops_nothing(self, finished):
        smaller = prune(SPACE, finished, threshold=0.0)
        assert smaller["model"]["values"] == SPACE["model"]["values"]


class TestWhatItRefusesToRemove:
    def test_best_branch_survives_any_threshold(self, finished):
        """The one unrecoverable mistake, so it is guarded unconditionally."""
        brutal = prune(SPACE, finished, threshold=0.99, keep_min=1)
        assert finished["best_params"]["model"] in brutal["model"]["values"]

    def test_protect_best_can_be_turned_off(self, finished):
        """Off, probability alone decides — which is the risky mode, by design."""
        kept = prune(SPACE, finished, threshold=0.99, keep_min=1,
                     protect_best=False)["model"]["values"]
        assert len(kept) == 1

    def test_keep_min_holds_a_floor(self, finished):
        smaller = prune(SPACE, finished, threshold=0.99, keep_min=3)
        assert len(smaller["model"]["values"]) >= 3

    def test_a_node_can_never_be_emptied(self, finished):
        smaller = prune(SPACE, finished, threshold=1.0, keep_min=1)
        assert smaller["model"]["values"]

    def test_parameters_absent_from_the_tree_are_kept(self, finished):
        """A tree that predates a new parameter must not silently delete it."""
        extended = dict(SPACE)
        extended["added_later"] = {"min": 0.0, "max": 1.0}
        smaller = prune(extended, finished)
        assert smaller["added_later"] == {"min": 0.0, "max": 1.0}


class TestNarrowing:
    def test_off_by_default(self, finished):
        smaller = prune(SPACE, finished)
        assert smaller["z"]["min"] == 0.0 and smaller["z"]["max"] == 10.0

    def test_narrow_tightens_around_what_was_learned(self, finished):
        smaller = prune(SPACE, finished, narrow=2.0)
        assert smaller["z"]["min"] > 0.0
        assert smaller["z"]["max"] < 10.0
        assert smaller["z"]["min"] < 7.0 < smaller["z"]["max"]

    def test_narrowed_bounds_stay_inside_the_originals(self, finished):
        smaller = prune(SPACE, finished, narrow=10.0)
        assert smaller["z"]["min"] >= SPACE["z"]["min"]
        assert smaller["z"]["max"] <= SPACE["z"]["max"]

    def test_integer_bounds_stay_usable(self, finished):
        smaller = prune(SPACE, finished, narrow=2.0)
        assert smaller["n"]["int"] is True
        assert smaller["n"]["max"] > smaller["n"]["min"]

    def test_integer_bounds_stay_inside_the_originals(self, finished):
        """Regression: rounding outwards used to push `max` past the original."""
        for k in (0.5, 1.0, 2.0, 5.0, 50.0):
            smaller = prune(SPACE, finished, narrow=k)
            assert smaller["n"]["min"] >= SPACE["n"]["min"], k
            assert smaller["n"]["max"] <= SPACE["n"]["max"], k

    def test_a_collapsed_range_keeps_the_original_bounds(self):
        """Zero spread would otherwise produce min == max and an invalid space."""
        flat = {"x": {"min": 0.0, "max": 1.0}}
        result = BUTChC_optimize(flat, lambda c: 0.0, 60, verbose=False, seed=3)
        for node in result["prob_tree"].values():
            node["reservoir"] = [0.5] * len(node["reservoir"])
        smaller = prune(flat, result, narrow=0.0001)
        assert smaller["x"]["min"] < smaller["x"]["max"]


class TestRefusals:
    def test_a_bare_tree_is_not_a_result(self, finished):
        with pytest.raises(SearchSpaceError, match="prob_tree"):
            prune(SPACE, finished["prob_tree"])

    def test_keep_min_below_one(self, finished):
        with pytest.raises(SearchSpaceError, match="keep_min"):
            prune(SPACE, finished, keep_min=0)

    def test_negative_narrow(self, finished):
        with pytest.raises(SearchSpaceError, match="narrow"):
            prune(SPACE, finished, narrow=-1.0)


class TestReport:
    def test_names_what_was_dropped(self, finished):
        smaller = prune(SPACE, finished)
        text = "\n".join(prune_report(SPACE, smaller))
        assert "model" in text and "'d'" in text

    def test_empty_when_nothing_changed(self, finished):
        smaller = prune(SPACE, finished, threshold=0.0)
        assert prune_report(SPACE, smaller) == []

    def test_reports_narrowing_with_both_ranges(self, finished):
        smaller = prune(SPACE, finished, narrow=2.0)
        text = "\n".join(prune_report(SPACE, smaller))
        assert "narrowed" in text and "->" in text


class TestReturnTree:
    """`result['prob_tree']` itself can never be a valid `start_prob_tree` for
    a pruned space: it still carries every dropped choice and the original,
    wider continuous bounds, and `BUTChC_optimize` requires an exact match.
    `return_tree=True` is what actually makes the documented scout-then-warm-
    start workflow in docs/examples.md work.
    """

    def test_default_return_is_unchanged(self, finished):
        """Existing callers see no difference: no tuple, no new keys."""
        assert isinstance(prune(SPACE, finished), dict)

    def test_tree_matches_the_pruned_space(self, finished):
        smaller, tree = prune(SPACE, finished, return_tree=True)
        check_tree_matches_searchspace(tree, smaller)  # raises if it doesn't

    def test_tree_is_structurally_valid_on_its_own(self, finished):
        smaller, tree = prune(SPACE, finished, narrow=2.0, return_tree=True)
        validate_prob_tree(tree)

    def test_original_tree_still_fails_against_the_pruned_space(self, finished):
        """The bug this guards against: passing result['prob_tree'] directly."""
        smaller, _ = prune(SPACE, finished, return_tree=True)
        with pytest.raises(ValueError):
            check_tree_matches_searchspace(finished["prob_tree"], smaller)

    def test_warm_start_actually_runs(self, finished):
        smaller, tree = prune(SPACE, finished, narrow=2.0, return_tree=True)
        result = BUTChC_optimize(smaller, objective, 30, verbose=False, seed=9,
                                 start_prob_tree=tree)
        assert result["best_params"]["model"] in smaller["model"]["values"]

    def test_warm_start_carries_evidence_forward(self, finished):
        """The point of the whole feature: the returned tree's probabilities
        should already reflect what the scout run learned, not reset to
        uniform over the survivors. A multi-seed statistical comparison of
        downstream optimisation outcomes would be noisier and less direct
        than checking the actual claim: the winning branch's probability is
        carried across, not reset."""
        smaller, tree = prune(SPACE, finished, return_tree=True)
        uniform = 1.0 / len(smaller["model"]["values"])
        best_branch = finished["best_params"]["model"]
        assert tree["model"]["prob"][best_branch] > uniform

    def test_dropped_categorical_choices_are_absent_from_the_tree(self, finished):
        smaller, tree = prune(SPACE, finished, threshold=0.99, keep_min=1,
                              return_tree=True)
        assert set(tree["model"]["prob"]) == set(smaller["model"]["values"])
        assert "d" not in tree["model"]["prob"]

    def test_continuous_archive_is_filtered_to_the_narrowed_range(self, finished):
        smaller, tree = prune(SPACE, finished, threshold=0.0, narrow=2.0,
                              return_tree=True)
        lo, hi = tree["z"]["min"], tree["z"]["max"]
        assert all(lo <= x <= hi for x in tree["z"]["reservoir"])
        assert abs(sum(tree["z"]["weights"]) - 1.0) < 1e-9

    def test_nested_next_level_is_recursively_consistent(self, finished):
        smaller, tree = prune(SPACE, finished, threshold=0.99, keep_min=1,
                              return_tree=True)
        for choice in smaller["model"]["values"]:
            assert choice in tree["model"]["next_level"]
            check_tree_matches_searchspace(
                tree["model"]["next_level"][choice],
                smaller["model"]["next_level"][choice],
            )

    def test_parameter_absent_from_the_tree_gets_a_fresh_valid_node(self, finished):
        extended = dict(SPACE)
        extended["added_later"] = {"min": 0.0, "max": 1.0}
        smaller, tree = prune(extended, finished, return_tree=True)
        check_tree_matches_searchspace(tree, smaller)

    def test_narrowing_that_empties_the_archive_reseeds_instead_of_breaking(self):
        """A heavy-tailed archive can leave nothing inside a tight narrowing;
        the tree must still come back structurally valid and usable."""
        flat = {"x": {"min": 0.0, "max": 1.0}}
        result = BUTChC_optimize(flat, lambda c: 0.0, 60, verbose=False, seed=3)
        for node in result["prob_tree"].values():
            node["reservoir"] = [0.01] * len(node["reservoir"])
        smaller, tree = prune(flat, result, threshold=0.0, narrow=0.001,
                              return_tree=True)
        assert tree["x"]["reservoir"]
        check_tree_matches_searchspace(tree, smaller)
        BUTChC_optimize(smaller, lambda c: 0.0, 10, verbose=False, seed=4,
                        start_prob_tree=tree)
