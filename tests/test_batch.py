"""Batch sampling and parallel dispatch.

The property that matters most is that a seeded run is reproducible whatever
the executor does with ordering or timing. Several tests below deliberately
evaluate out of order, or in reverse, to prove the answer does not move.
"""

import math
import os
import random

import pytest

from butchc import BUTChC_optimize
from butchc._parallel import evaluate_batch, resolve_batch
from butchc._sampling import traverse_sample
from butchc._tree import initialize_prob_tree

# ---------------------------------------------------------------------------
# Objectives. Module level so a process pool could pickle them, and so the
# same functions can be reused across tests.
# ---------------------------------------------------------------------------


def sphere(config):
    return -sum(v ** 2 for k, v in config.items() if isinstance(v, float))


def branchy(config):
    base = {"a": 0.0, "b": 1.0, "c": -1.0}[config["kind"]]
    return base - (config["x"] - 1.0) ** 2


SPACE = {"x": {"min": -5.0, "max": 5.0}, "y": {"min": -5.0, "max": 5.0}}
BRANCH_SPACE = {
    "kind": {"values": ["a", "b", "c"]},
    "x": {"min": -3.0, "max": 3.0},
}


def _run(**kw):
    kw.setdefault("verbose", False)
    kw.setdefault("seed", 0)
    return BUTChC_optimize(SPACE, sphere, kw.pop("budget", 60), **kw)


# ---------------------------------------------------------------------------
# Fake executors
# ---------------------------------------------------------------------------


class MapExecutor:
    """Order-preserving map, like concurrent.futures."""

    def __init__(self):
        self.batch_sizes = []

    def map(self, fn, iterable):
        items = list(iterable)
        self.batch_sizes.append(len(items))
        return [fn(item) for item in items]


class ReverseMapExecutor(MapExecutor):
    """Evaluates back to front but returns in input order.

    A real pool completes work in an arbitrary order and reassembles it. This
    makes that reordering visible to the test rather than incidental.
    """

    def map(self, fn, iterable):
        items = list(iterable)
        self.batch_sizes.append(len(items))
        results = [None] * len(items)
        for i in reversed(range(len(items))):
            results[i] = fn(items[i])
        return results


class _Future:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class SubmitExecutor:
    """Exposes only submit/Future, no map."""

    def __init__(self):
        self.calls = 0

    def submit(self, fn, arg):
        self.calls += 1
        return _Future(fn(arg))


class BrokenExecutor:
    def map(self, fn, iterable):
        raise TypeError("cannot pickle 'local object'")


class UselessExecutor:
    pass


# ---------------------------------------------------------------------------


class TestBatchSizing:
    def test_default_is_sequential(self):
        assert _run(budget=20)["n_batches"] == 20

    @pytest.mark.parametrize("batch", [2, 3, 5, 8])
    def test_batch_count_is_ceiling_of_budget(self, batch):
        result = _run(budget=20, batch=batch)
        assert result["n_batches"] == math.ceil(20 / batch)

    def test_budget_is_exact_when_batch_does_not_divide(self):
        result = _run(budget=17, batch=5)
        assert len(result["history"]) == 17
        assert result["n_batches"] == 4

    def test_batch_larger_than_budget_runs_one_batch(self):
        result = _run(budget=3, batch=50)
        assert len(result["history"]) == 3
        assert result["n_batches"] == 1

    def test_minus_one_resolves_to_cpu_count(self):
        assert resolve_batch(-1) == (os.cpu_count() or 1)

    def test_explicit_size_passes_through(self):
        assert resolve_batch(7) == 7

    @pytest.mark.parametrize("bad", [0, -2, 2.5, "4", True])
    def test_invalid_batch_rejected_before_any_trial(self, bad):
        calls = []

        def counting(config):
            calls.append(config)
            return 0.0

        with pytest.raises(ValueError):
            BUTChC_optimize(SPACE, counting, 10, batch=bad, verbose=False)
        assert calls == []


class TestBatchIndex:
    def test_history_records_which_batch_each_trial_belonged_to(self):
        result = _run(budget=10, batch=4)
        indices = [h["batch_index"] for h in result["history"]]
        assert indices == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]

    def test_sequential_gives_one_trial_per_batch(self):
        result = _run(budget=5)
        assert [h["batch_index"] for h in result["history"]] == [0, 1, 2, 3, 4]


class TestStaleness:
    """A batch must be drawn from one state of the tree, not k states."""

    def test_one_batch_equals_k_draws_from_the_untouched_tree(self):
        # With budget == batch there is exactly one batch, so every config must
        # be a draw from the initial tree. Reproducing them with a bare sampler
        # loop is the direct test of that: if any update leaked in between
        # draws, the sequences would diverge after the first.
        # explore is pinned on both sides rather than left at the default: a
        # non-zero explore draws an extra random number per node, so the two
        # RNG streams would diverge on the default alone and the test would
        # fail for a reason that has nothing to do with staleness.
        budget = 12
        result = BUTChC_optimize(SPACE, sphere, budget, batch=budget, seed=4,
                                 verbose=False, explore=0.0)

        tree = initialize_prob_tree(SPACE)
        rng = random.Random(4)
        expected = [traverse_sample(tree, 1.0, rng, explore=0.0)[0]
                    for _ in range(budget)]

        assert [h["params"] for h in result["history"]] == expected

    def test_a_whole_batch_ranks_against_the_pre_batch_history(self):
        # First batch has nothing to rank against, so every member is neutral
        # and none can clear the gate.
        result = _run(budget=12, batch=6)
        first_batch = result["history"][:6]
        assert all(h["quality"] == 0.0 for h in first_batch)
        assert all(not h["gated"] for h in first_batch)

    def test_batch_members_are_not_ranked_against_each_other(self):
        # A batch member's rank must not depend on its position in the batch.
        # Ranking within the batch would give the last member k-1 comparisons
        # the first one did not have.
        result = _run(budget=8, batch=8)
        assert {h["quality"] for h in result["history"]} == {0.0}

    def test_later_batches_do_clear_the_gate(self):
        result = _run(budget=200, batch=4)
        assert result["n_gated"] > 0


class TestReproducibility:
    def test_same_seed_same_result(self):
        a = _run(budget=40, batch=6)
        b = _run(budget=40, batch=6)
        assert a["best_value"] == b["best_value"]
        assert a["best_params"] == b["best_params"]
        assert a["loss_history"] == b["loss_history"]

    def test_result_does_not_depend_on_evaluation_order(self):
        forward = BUTChC_optimize(SPACE, sphere, 40, batch=6, seed=3,
                                  verbose=False, executor=MapExecutor())
        reverse = BUTChC_optimize(SPACE, sphere, 40, batch=6, seed=3,
                                  verbose=False, executor=ReverseMapExecutor())
        assert forward["loss_history"] == reverse["loss_history"]
        assert forward["best_params"] == reverse["best_params"]

    def test_executor_does_not_change_the_answer(self):
        inline = BUTChC_optimize(SPACE, sphere, 40, batch=6, seed=5,
                                 verbose=False)
        pooled = BUTChC_optimize(SPACE, sphere, 40, batch=6, seed=5,
                                 verbose=False, executor=MapExecutor())
        assert inline["loss_history"] == pooled["loss_history"]
        assert inline["best_value"] == pooled["best_value"]

    def test_submit_only_executor_agrees_with_map(self):
        mapped = BUTChC_optimize(SPACE, sphere, 30, batch=5, seed=2,
                                 verbose=False, executor=MapExecutor())
        submitted = BUTChC_optimize(SPACE, sphere, 30, batch=5, seed=2,
                                    verbose=False, executor=SubmitExecutor())
        assert mapped["loss_history"] == submitted["loss_history"]

    def test_different_batch_sizes_give_different_runs(self):
        # Not a correctness requirement so much as a guard against `batch`
        # being silently ignored.
        one = _run(budget=40, batch=1)
        eight = _run(budget=40, batch=8)
        assert one["loss_history"] != eight["loss_history"]


class TestExecutorDispatch:
    def test_executor_receives_whole_batches(self):
        ex = MapExecutor()
        BUTChC_optimize(SPACE, sphere, 20, batch=4, seed=0, verbose=False,
                        executor=ex)
        assert ex.batch_sizes == [4, 4, 4, 4, 4]

    def test_final_short_batch_is_dispatched_at_its_real_size(self):
        ex = MapExecutor()
        BUTChC_optimize(SPACE, sphere, 14, batch=4, seed=0, verbose=False,
                        executor=ex)
        assert ex.batch_sizes == [4, 4, 4, 2]

    def test_single_config_batches_skip_the_executor(self):
        # Dispatching one call costs latency and buys nothing.
        ex = MapExecutor()
        BUTChC_optimize(SPACE, sphere, 6, batch=1, seed=0, verbose=False,
                        executor=ex)
        assert ex.batch_sizes == []

    def test_kwargs_are_forwarded_through_the_executor(self):
        def scaled(config, factor):
            return sphere(config) * factor

        result = BUTChC_optimize(SPACE, scaled, 12, batch=4, seed=0,
                                 verbose=False, executor=MapExecutor(),
                                 factor=2.0)
        assert result["best_value"] == pytest.approx(
            2.0 * max(sphere(h["params"]) for h in result["history"])
        )

    def test_executor_without_map_or_submit_is_rejected(self):
        with pytest.raises(TypeError, match="map"):
            evaluate_batch(sphere, [{"x": 0.0}, {"x": 1.0}], UselessExecutor(), {})

    def test_pickling_failure_is_explained(self):
        with pytest.raises(RuntimeError, match="ThreadPoolExecutor"):
            evaluate_batch(sphere, [{"x": 0.0}, {"x": 1.0}], BrokenExecutor(), {})


class TestBatchStillOptimises:
    @pytest.mark.parametrize("batch", [1, 4, 16])
    def test_beats_random_search_on_5d_sphere(self, batch):
        space = {f"x{i}": {"min": -5.0, "max": 5.0} for i in range(5)}
        result = BUTChC_optimize(space, sphere, 400, batch=batch, seed=0,
                                 verbose=False)

        rng = random.Random(0)
        best_random = max(
            sphere({f"x{i}": rng.uniform(-5.0, 5.0) for i in range(5)})
            for _ in range(400)
        )
        assert result["best_value"] > best_random

    @pytest.mark.parametrize("batch", [2, 8])
    def test_categorical_still_concentrates(self, batch):
        result = BUTChC_optimize(BRANCH_SPACE, branchy, 300, batch=batch,
                                 seed=0, verbose=False)
        prob = result["prob_tree"]["kind"]["prob"]
        assert prob["b"] == max(prob.values())

    def test_warm_start_works_from_a_batched_run(self):
        first = BUTChC_optimize(SPACE, sphere, 40, batch=4, seed=0,
                                verbose=False)
        second = BUTChC_optimize(SPACE, sphere, 40, batch=4, seed=1,
                                 verbose=False,
                                 start_prob_tree=first["prob_tree"])
        assert second["best_value"] >= -25.0

    def test_warmup_interacts_with_batching(self):
        result = BUTChC_optimize(SPACE, sphere, 20, batch=4, n_warmup=8,
                                 seed=0, verbose=False)
        assert all(h["quality"] == 0.0 for h in result["history"][:8])

    def test_nan_objectives_survive_batching(self):
        def half_nan(config):
            return float("nan") if config["x"] > 0 else sphere(config)

        result = BUTChC_optimize(SPACE, half_nan, 40, batch=8, seed=0,
                                 verbose=False)
        assert result["best_params"]["x"] <= 0
