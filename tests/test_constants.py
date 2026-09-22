"""Every constant documented as patchable must actually reach the search.

The failure this guards against has happened twice. `KDE_RESERVOIR_SIZE` is
imported by name into two modules besides the one defining it, so a sweep that
patched only `_utils` measured a knob that was half-applied (v0.4). And
`optimizer` passed `neutral=0.5` as a literal to `update_tree`, so patching
`_update.NEUTRAL_QUALITY` moved nothing at all (v0.5.0).

Both are silent: the sweep runs, reports a winner, and the winner is noise.
A test that a knob *does something* is cheap; discovering months later that a
published default was selected under a knob that never moved is not.

Each case below patches one constant, asserts the outcome differs from the
default, and restores it. What the new value does is not the point — only that
it arrives.
"""

import pytest

import butchc._tree as tree_mod
import butchc._update as upd
import butchc._utils as utils
import butchc.optimizer as opt
from butchc import BUTChC_optimize
from butchc._tree import initialize_prob_tree

SPACE = {
    "model": {
        "values": ["a", "b", "c"],
        "next_level": {k: {"x": {"min": 0.0, "max": 1.0}} for k in "abc"},
    },
    "y": {"min": 0.0, "max": 10.0},
}

BONUS = {"a": 1.0, "b": 0.2, "c": 0.0}


def objective(config):
    return (
        -((config["x"] - 0.3) ** 2)
        - ((config["y"] - 7.0) ** 2) / 100.0
        + BONUS[config["model"]]
    )


def signature(**kwargs):
    """A seeded run reduced to what a changed constant would have to move."""
    result = BUTChC_optimize(SPACE, objective, budget=80, verbose=False,
                             seed=5, **kwargs)
    return (
        round(result["best_value"], 9),
        result["n_updates"],
        tuple(sorted(round(v, 9)
                     for v in result["prob_tree"]["model"]["prob"].values())),
    )


@pytest.fixture
def restore():
    """Snapshot every documented constant and put it back afterwards."""
    saved = [
        (opt, "COMMITMENT"),
        (opt, "MOVEMENT_EPSILON"),
        (upd, "DISCOUNT"),
        (upd, "NEUTRAL_QUALITY"),
        (utils, "KDE_RESERVOIR_SIZE"),
        (tree_mod, "KDE_RESERVOIR_SIZE"),
        (upd, "KDE_RESERVOIR_SIZE"),
        (utils, "MIN_BANDWIDTH_FRACTION"),
        (utils, "MIN_GRID_POINTS"),
        (tree_mod, "MIN_GRID_POINTS"),
        (utils, "RANK_SHARPNESS"),
        (utils, "MAX_SHARPNESS"),
    ]
    originals = [(mod, name, getattr(mod, name)) for mod, name in saved]
    yield
    for mod, name, value in originals:
        setattr(mod, name, value)


class TestConstantsAreLive:
    def test_commitment(self, restore):
        base = signature()
        opt.COMMITMENT = 2.0
        assert signature() != base

    def test_discount(self, restore):
        base = signature()
        upd.DISCOUNT = 0.8
        assert signature() != base

    def test_neutral_quality(self, restore):
        """Regression: `optimizer` used to pass `neutral=0.5` as a literal."""
        base = signature()
        upd.NEUTRAL_QUALITY = 0.1
        assert signature() != base

    def test_kde_reservoir_size(self, restore):
        """Regression: all three bindings must be set, and all three must count."""
        base = signature()
        for module in (utils, tree_mod, upd):
            module.KDE_RESERVOIR_SIZE = 20
        assert signature() != base

    def test_min_bandwidth_fraction(self, restore):
        base = signature()
        utils.MIN_BANDWIDTH_FRACTION = 0.2
        assert signature() != base

    def test_rank_sharpness(self, restore):
        base = signature()
        utils.RANK_SHARPNESS = 8.0
        assert signature() != base

    def test_max_sharpness(self, restore):
        """Only bites when `RANK_SHARPNESS * lambda_` exceeds it, so lambda_ is raised."""
        base = signature(lambda_=4.0)
        utils.MAX_SHARPNESS = 1.0
        assert signature(lambda_=4.0) != base

    def test_movement_epsilon(self, restore):
        """Reporting only — it cannot move the search, so assert on the report."""
        opt.MOVEMENT_EPSILON = 10.0
        result = BUTChC_optimize(SPACE, objective, budget=40, verbose=False,
                                 seed=5)
        assert result["n_updates"] == 0

    def test_min_grid_points(self, restore):
        """Caps how much of the initial grid a prior may displace, in `_tree`."""
        space = {"x": {"min": 0.0, "max": 1.0,
                       "prior": {"mean": 0.5, "std": 0.05},
                       "prior_strength": 40}}

        def prior_points(floor):
            tree_mod.MIN_GRID_POINTS = floor
            node = initialize_prob_tree(space)["x"]
            return sum(1 for s in node["scores"] if s != float("-inf"))

        assert prior_points(10) != prior_points(30)


class TestSweepCoversWhatItClaims:
    """`docs/api.md` names the constants `tune.py` sweeps; keep the two in step.

    Documenting a knob as swept and then not sweeping it is the same failure as
    a knob that does not move: the published default was never compared against
    its alternatives. `tune.py` is read rather than imported — importing it
    patches module constants as a side effect, and it lives outside the
    installed package.
    """

    @staticmethod
    def _dict_keys(name):
        import ast
        import os

        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "benchmarks", "tune.py")
        if not os.path.exists(path):
            pytest.skip("benchmarks/tune.py is not part of the installed package")
        with open(path) as handle:
            tree = ast.parse(handle.read())
        for node in tree.body:
            if (isinstance(node, ast.Assign)
                    and any(getattr(t, "id", None) == name for t in node.targets)):
                return {k.value for k in node.value.keys}
        raise AssertionError(f"{name} not found in benchmarks/tune.py")

    def test_documented_knobs_are_swept(self):
        swept = {"commitment", "discount", "neutral", "reservoir",
                 "min_bandwidth", "rank_sharpness"}
        assert swept <= self._dict_keys("GRID")

    def test_every_swept_knob_has_a_baseline(self):
        assert self._dict_keys("GRID") <= self._dict_keys("BASE")
