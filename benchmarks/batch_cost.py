"""What does batching cost in sample efficiency?

    python benchmarks/batch_cost.py [n_seeds] [--sizes 1,4,8,16]
                                    [--suite tuning|heldout|all]

A batch of ``k`` leaves the model stale for ``k - 1`` evaluations, so batching
must cost something at a fixed budget. The question is how much, because that
is the price of the wall-clock speedup an executor buys — accept it if the
objective is slow enough that ``k`` workers finish a batch in the time one
worker would take for a single evaluation.

This measures the price directly, in the only units that matter: best objective
found at a fixed number of evaluations. Every batch size sees the same seeds and
the same budget, so the comparison is paired.

The reported ``cost`` is the median regret of batch ``k`` relative to
sequential, as a percentage of the sequential median. Objectives here have a
known optimum of 0, so regret is just ``-median``. Negative cost means the
batch size beat sequential on that problem, which does happen — the ranking is
noisy at small seed counts and batching adds a little diversity.
"""

import argparse
import os
import random
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.abspath(os.path.join(_HERE, os.pardir))]

from butchc import BUTChC_optimize            # noqa: E402
from problems import HELDOUT, NOISY, TUNING   # noqa: E402


def run(problem, budget, seed, batch):
    """Best objective found, scored noise-free where the problem asks for it."""
    space, objective, report = _unpack(problem)
    result = BUTChC_optimize(space, objective, budget=budget, verbose=False,
                             seed=seed, batch=batch)
    if report is None:
        return result["best_value"]
    if result["best_params"] is None:
        return -float("inf")
    return report(result["best_params"])


def _unpack(problem):
    """Problems are (space, objective, budget) or (space, objective, budget, report)."""
    space, objective = problem[0], problem[1]
    report = problem[3] if len(problem) > 3 else None
    return space, objective, report


def _reset_noise(problem, seed, noise_seed_base=90000):
    """Give every batch size the identical noise sequence, not merely an
    equally distributed one.

    A noisy problem carries a mutable ``{'rng': ...}`` dict that its objective
    draws from, matching ``evaluate.py``. Reseeding it per seed is what makes
    the comparison paired rather than merely fair on average.
    """
    state = problem[4] if len(problem) > 4 else None
    if state is not None:
        state["rng"] = random.Random(noise_seed_base + seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("seeds", nargs="?", type=int, default=10)
    parser.add_argument("--sizes", default="1,4,8,16")
    parser.add_argument("--suite", default="all",
                        choices=["tuning", "heldout", "noisy", "all"])
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    if sizes[0] != 1:
        sizes.insert(0, 1)

    suites = {"tuning": [("tuning", TUNING)],
              "heldout": [("heldout", HELDOUT)],
              "noisy": [("noisy", NOISY)],
              "all": [("tuning", TUNING), ("heldout", HELDOUT), ("noisy", NOISY)]}

    header = f"{'problem':<22}{'budget':>8}" + "".join(f"{('k=' + str(k)):>12}" for k in sizes)
    started = time.perf_counter()

    all_costs = {k: [] for k in sizes[1:]}

    for suite_name, suite in suites[args.suite]:
        print(f"\n=== {suite_name} " + "=" * (len(header) - len(suite_name) - 5))
        print(header)
        print("-" * len(header))

        for name, problem in suite.items():
            budget = problem[2]
            medians = {}
            for k in sizes:
                scores = []
                for seed in range(args.seeds):
                    _reset_noise(problem, seed)
                    scores.append(run(problem, budget, seed, k))
                medians[k] = statistics.median(scores)

            row = f"{name:<22}{budget:>8}"
            for k in sizes:
                row += f"{medians[k]:>12.4f}"
            print(row)

            base = -medians[1]
            if base > 1e-12:
                for k in sizes[1:]:
                    all_costs[k].append(100.0 * ((-medians[k]) - base) / base)

    print("\n" + "=" * len(header))
    print("Median extra regret vs sequential, across all problems measured:")
    for k in sizes[1:]:
        costs = all_costs[k]
        if costs:
            print(f"  k={k:<4} {statistics.median(costs):+7.1f}%   "
                  f"(worst problem {max(costs):+.1f}%, best {min(costs):+.1f}%)")
    print("\nRead the median, not the extremes. A problem already solved to "
          "~1e-5\nturns any residual difference into a huge percentage, so the "
          "worst and best\ncolumns are dominated by problems where both batch "
          "sizes effectively won.")
    print(f"\n{args.seeds} seeds, {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
