"""Run the real-world suite through the same harness as the synthetic one.

    python dev/eval/run_real.py [n_seeds] [--methods random,optuna,butchc]
                                          [--instances 1053,15]
                                          [--budget 400]

Deliberately thin. Everything that decides what a result *means* — paired
seeds, medians, win-loss records, sign tests — lives in
``benchmarks/evaluate.py`` and is imported, not reimplemented. A second copy of
the statistics is a second place for them to drift.

Read the output with one thing in mind: these are surrogate benchmarks, so the
landscape is a neural network's approximation of a real one. That is far closer
to reality than a Rastrigin function, and still not reality. A conclusion that
holds here and on the synthetic suite is worth acting on; one that holds only
here is worth investigating before believing.
"""

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
sys.path[:0] = [_HERE, _ROOT, os.path.join(_ROOT, "benchmarks")]

from evaluate import METHODS, collect, report_table   # noqa: E402
from yahpo_suite import DEFAULT_BUDGET, rbv2_super    # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("n_seeds", nargs="?", type=int, default=10)
    parser.add_argument("--methods", default="random,optuna,butchc",
                        help="optuna is the reference TPE; tpe is the "
                             "dependency-free equivalent")
    parser.add_argument("--instances", default="",
                        help="comma-separated YAHPO instance ids; "
                             "default is the published suite")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--target", default="acc")
    args = parser.parse_args(argv)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        parser.error(f"unknown method(s) {unknown}; choose from {sorted(METHODS)}")

    instances = [i.strip() for i in args.instances.split(",") if i.strip()] or None

    started = time.time()
    suite = rbv2_super(instances=instances, target=args.target,
                       budget=args.budget)
    report_table("rbv2_super — AutoML pipeline configuration (YAHPO Gym)",
                 collect(suite, methods, list(range(args.n_seeds))), methods)
    print(f"\n{time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
