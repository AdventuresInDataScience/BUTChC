"""What does the optimiser itself cost, separately from the objective?

Every other script here measures sample efficiency — how good an answer you get
for a fixed number of objective evaluations. This one measures the opposite
axis: the wall-clock the optimiser spends deciding what to try next, with the
objective stubbed out to a constant so nothing else is in the number.

It matters for two reasons. It is the axis on which a dependency-free pure
Python library would be expected to lose, so it is worth knowing whether it
does. And it decides whether the ~1 us objectives in `problems.py` make this
suite's runtime a statement about BUTChC or about its baselines.

    python benchmarks/overhead.py
    python benchmarks/overhead.py --trials 2000 --dims 5,20,50

`optuna` is used if installed and skipped otherwise, so this stays runnable on
a bare Python like everything else in this directory.
"""

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.abspath(os.path.join(_HERE, os.pardir))]

import baselines                                   # noqa: E402
from butchc import BUTChC_optimize                 # noqa: E402


def constant(config):
    """A free objective, so the measurement is all optimiser."""
    return 0.0


def continuous_space(dims):
    return {f"x{i}": {"min": 0.0, "max": 1.0} for i in range(dims)}


def time_one(fn, space, trials):
    started = time.perf_counter()
    fn(space, constant, trials)
    return time.perf_counter() - started


def butchc_runner(space, objective, trials):
    BUTChC_optimize(space, objective, trials, verbose=False, seed=0)


def tpe_runner(space, objective, trials):
    baselines.tpe_search(space, objective, trials, seed=0)


def optuna_runner(space, objective, trials):
    baselines.optuna_search(space, objective, trials, seed=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--dims", default="1,5,20")
    args = parser.parse_args(argv)

    dims = [int(d) for d in args.dims.split(",")]

    runners = [("BUTChC", butchc_runner), ("TPE (this repo)", tpe_runner)]
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        runners.append(("Optuna TPESampler", optuna_runner))
    except ImportError:
        print("optuna not installed — skipping that row\n")

    print(f"optimiser overhead: {args.trials} trials, free objective\n")
    head = f"{'optimiser':<22}" + "".join(f"{f'{d}D':>14}" for d in dims)
    print(head)
    print("-" * len(head))

    # Measured once and reused for both tables. Re-timing for the ratios would
    # divide two different samples of a noisy quantity, and BUTChC's own row
    # would come out as something other than 1.0x.
    per_trial = {}
    for label, fn in runners:
        row = f"{label:<22}"
        for d in dims:
            seconds = time_one(fn, continuous_space(d), args.trials)
            per_trial[(label, d)] = seconds / args.trials * 1e6
            row += f"{f'{per_trial[(label, d)]:.0f} us':>14}"
        print(row, flush=True)

    reference = runners[0][0]
    print(f"\nrelative to {reference} (higher = slower)\n")
    print(head)
    print("-" * len(head))
    for label, _ in runners:
        row = f"{label:<22}"
        for d in dims:
            ratio = per_trial[(label, d)] / per_trial[(reference, d)]
            row += f"{f'{ratio:.1f}x':>14}"
        print(row, flush=True)

    print("\nRead this alongside sample efficiency, not instead of it: a method "
          "that\nspends longer choosing may still need fewer evaluations. It "
          "matters when the\nobjective is cheap, and is close to irrelevant "
          "when a trial costs a model fit.")


if __name__ == "__main__":
    main()
