"""Measure BUTChC against random search and TPE on matched budgets and seeds.

    python benchmarks/evaluate.py [n_seeds] [--suite tuning|heldout|all]
                                            [--methods random,tpe,butchc]

Every method sees the same seed for the same problem, so the comparison is
paired and the per-seed differences are meaningful. For each problem the
median is reported alongside a win rate and a two-sided sign-test p-value
against each baseline, because a median gap says nothing about whether the
method wins reliably or wins once and loses the rest.

Two suites are reported separately. ``TUNING`` is the set the shipped defaults
were selected on and should be read as optimistic; ``HELDOUT`` was not
consulted during tuning and is the honest read on whether those defaults
generalise.

Ratios of medians are deliberately not reported. Most of these objectives are
squared errors, where a ratio flatters: a 40x ratio on a sphere is about 6x in
distance.
"""

import argparse
import math
import os
import random
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.abspath(os.path.join(_HERE, os.pardir))]

from baselines import (flat_tpe_search, optuna_search, random_search,  # noqa: E402
                       tpe_search)
from butchc import BUTChC_optimize                       # noqa: E402
from problems import HELDOUT, NOISY, TUNING              # noqa: E402


def butchc_search(space, objective, budget, seed, report=None):
    """Adapter giving BUTChC the same call signature as the baselines."""
    result = BUTChC_optimize(space, objective, budget=budget, verbose=False,
                             seed=seed)
    if report is None:
        return result["best_value"]
    return report(result["best_params"]) if result["best_params"] else -float("inf")


#: ``optuna`` is the reference TPE and needs ``pip install optuna``; ``tpe``
#: is the dependency-free equivalent in baselines.py. Neither is in the
#: default method list twice, because running both doubles the wall clock to
#: measure nearly the same thing.
METHODS = {
    "random": random_search,
    "tpe": tpe_search,
    "flat": flat_tpe_search,
    "optuna": optuna_search,
    "butchc": butchc_search,
}


def sign_test(a, b):
    """Two-sided exact sign test on paired samples.

    Args:
        a, b: Equal-length paired results, higher being better.

    Returns:
        ``(wins, losses, p_value)``, ignoring exact ties. ``p_value`` is 1.0
        when every pair ties, which is the honest reading of no evidence
        either way.
    """
    wins = sum(1 for x, y in zip(a, b) if x > y)
    losses = sum(1 for x, y in zip(a, b) if x < y)
    n = wins + losses
    if n == 0:
        return wins, losses, 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return wins, losses, min(1.0, 2.0 * tail)


def run_problem(method, space, objective, budget, seeds, report=None,
                state=None, noise_seed_base=90000):
    """Run one method on one problem across every seed.

    ``state`` is the mutable dict a noisy problem draws its noise from. It is
    reset per seed so that every method meets the identical noise sequence,
    not merely an equally-distributed one.
    """
    results = []
    for seed in seeds:
        if state is not None:
            state["rng"] = random.Random(noise_seed_base + seed)
        results.append(method(space, objective, budget, seed, report=report))
    return results


def collect(suite, methods, seeds, noisy=False):
    """Return ``{problem: {method: [per-seed result]}}`` for one suite."""
    table = {}
    for name, entry in suite.items():
        if noisy:
            space, objective, budget, report, state = entry
        else:
            (space, objective, budget), report, state = entry, None, None
        table[name] = {
            label: run_problem(METHODS[label], space, objective, budget, seeds,
                               report=report, state=state)
            for label in methods
        }
        table[name]["_budget"] = budget
    return table


def report_table(title, table, methods, reference="butchc"):
    """Print medians, then the reference method's record against each baseline."""
    print(f"\n{title}")
    print("(higher is better, optimum 0; n = paired seeds)\n")

    head = f"{'problem':<20}{'budget':>7}" + "".join(f"{m:>13}" for m in methods)
    print(head)
    print("-" * len(head))
    for name, row in table.items():
        line = f"{name:<20}{row['_budget']:>7}"
        for m in methods:
            line += f"{statistics.median(row[m]):>13.4f}"
        print(line)

    baselines = [m for m in methods if m != reference]
    if reference not in methods or not baselines:
        return

    print(f"\n{reference} head-to-head (win-loss, two-sided sign test)\n")
    head = f"{'problem':<20}" + "".join(f"{'vs ' + b:>22}" for b in baselines)
    print(head)
    print("-" * len(head))
    for name, row in table.items():
        line = f"{name:<20}"
        for b in baselines:
            wins, losses, p = sign_test(row[reference], row[b])
            line += f"{f'{wins}-{losses}   p={p:.3f}':>22}"
        print(line)


# ------------------------------------------------------------------ anytime
#
# Final-best answers "how good, given the budget". It does not answer "how much
# budget to get good enough", which is the question when a trial costs an hour
# of GPU time. Reaching 95% of the achievable quality in a fifth of the trials
# is a different and often more valuable property than winning at the end.
#
# The measure is in trials, not seconds. Trials are what an expensive objective
# charges for, and unlike wall-clock they do not change with the machine.
# Sampler overhead is real but belongs in its own table: at a millisecond per
# trial it dominates, and at a minute per trial it is invisible.


def with_recorder(objective):
    """Wrap an objective so every value it returns is kept, in call order.

    This is how the anytime curve is obtained without touching any optimiser:
    each one calls the objective once per trial, whatever it does internally,
    so the call sequence is the trial sequence.
    """
    seen = []

    def wrapped(config, **kwargs):
        value = objective(config, **kwargs)
        seen.append(value)
        return value

    return wrapped, seen


def best_so_far(values):
    """Running maximum — the curve a user actually experiences."""
    curve, best = [], -float("inf")
    for value in values:
        best = max(best, value)
        curve.append(best)
    return curve


def trials_to_target(curve, target):
    """Trials until the curve first reaches ``target``, or None if it never does.

    None is reported rather than substituted with the budget, because a run
    that never arrived and a run that arrived on the last trial are different
    facts and averaging them together hides the difference.
    """
    for i, value in enumerate(curve, start=1):
        if value >= target:
            return i
    return None


def collect_curves(suite, methods, seeds):
    """Return ``{problem: {method: [curve per seed]}}``."""
    table = {}
    for name, (space, objective, budget) in suite.items():
        table[name] = {"_budget": budget}
        for label in methods:
            curves = []
            for seed in seeds:
                recorded, seen = with_recorder(objective)
                METHODS[label](space, recorded, budget, seed)
                curves.append(best_so_far(seen))
            table[name][label] = curves
    return table


def report_anytime(title, table, methods, fraction=0.95):
    """Median trials to reach ``fraction`` of the achievable range.

    The range is anchored per problem: 0% is random search's median final
    value, 100% is the best any method reached on any seed. Anchoring to the
    problem rather than to its nominal optimum keeps the target attainable —
    several of these optima are approached but never reached — and makes the
    number comparable across problems with wildly different scales.
    """
    print(f"\n{title}")
    print(f"(trials to reach {fraction:.0%} of the achievable range; "
          f"lower is better)\n")

    head = (f"{'problem':<20}{'budget':>7}"
            + "".join(f"{m:>16}" for m in methods))
    print(head)
    print("-" * len(head))

    for name, row in table.items():
        budget = row["_budget"]
        finals = {m: [c[-1] for c in row[m]] for m in methods}
        ceiling = max(max(v) for v in finals.values())
        floor = (statistics.median(finals["random"]) if "random" in finals
                 else min(min(v) for v in finals.values()))
        if ceiling <= floor:
            print(f"{name:<20}{budget:>7}" + "".join(f"{'—':>16}" for _ in methods))
            continue
        target = floor + fraction * (ceiling - floor)

        line = f"{name:<20}{budget:>7}"
        for m in methods:
            hits = [trials_to_target(c, target) for c in row[m]]
            reached = [h for h in hits if h is not None]
            if not reached:
                line += f"{'never':>16}"
            else:
                cell = f"{int(statistics.median(reached))}"
                if len(reached) < len(hits):
                    cell += f" ({len(reached)}/{len(hits)})"
                line += f"{cell:>16}"
        print(line)

    print("\n'n/m' means only n of m seeds ever reached the target; the median "
          "is over those that did.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("n_seeds", nargs="?", type=int, default=30)
    parser.add_argument("--suite", default="all",
                        choices=["tuning", "heldout", "all"])
    parser.add_argument("--methods", default="random,tpe,butchc")
    parser.add_argument("--anytime", action="store_true",
                        help="report trials-to-target instead of final best")
    parser.add_argument("--fraction", type=float, default=0.95,
                        help="target as a fraction of the achievable range")
    args = parser.parse_args(argv)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        parser.error(f"unknown method(s) {unknown}; choose from {sorted(METHODS)}")

    seeds = list(range(args.n_seeds))
    started = time.time()

    if args.anytime:
        if args.suite in ("tuning", "all"):
            report_anytime("TUNED ON — anytime",
                           collect_curves(TUNING, methods, seeds), methods,
                           args.fraction)
        if args.suite in ("heldout", "all"):
            report_anytime("HELD OUT — anytime",
                           collect_curves(HELDOUT, methods, seeds), methods,
                           args.fraction)
        print(f"\n{time.time() - started:.1f}s")
        return

    if args.suite in ("tuning", "all"):
        report_table("TUNED ON (optimistic)",
                     collect(TUNING, methods, seeds), methods)
        report_table("UNDER OBSERVATION NOISE — true value of the reported best",
                     collect(NOISY, methods, seeds, noisy=True), methods)
    if args.suite in ("heldout", "all"):
        report_table("HELD OUT (not consulted while tuning)",
                     collect(HELDOUT, methods, seeds), methods)

    print(f"\n{time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
