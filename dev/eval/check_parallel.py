"""Exercise ``batch`` and ``executor`` against real pools, and time the speedup.

The unit tests use fake executors so they stay fast and deterministic. This
runs the real thing — threads, processes, and a deliberately slow objective —
and checks the two properties that actually matter in production:

1. A seeded run gives the same answer inline, on threads and on processes.
2. Wall clock actually falls.

    python dev/eval/check_parallel.py
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                os.pardir, os.pardir)))

from butchc import BUTChC_optimize  # noqa: E402

SPACE = {
    "x": {"min": -5.0, "max": 5.0},
    "y": {"min": -5.0, "max": 5.0},
    "kind": {"values": ["a", "b", "c"]},
}

BUDGET = 48
BATCH = 8

#: Fixed rather than ``os.cpu_count()``: the objective below sleeps, so it is
#: waiting rather than computing, and the speedup is visible even on a
#: single-core box. A real CPU-bound objective should use ``batch=-1``.
WORKERS = 8


def objective(config):
    """Deliberately slow, so wall clock is dominated by the objective."""
    time.sleep(0.02)
    bonus = {"a": 0.0, "b": 0.5, "c": -0.5}[config["kind"]]
    return -(config["x"] ** 2 + config["y"] ** 2) + bonus


def run(executor=None, batch=BATCH):
    start = time.perf_counter()
    result = BUTChC_optimize(
        SPACE, objective, BUDGET, batch=batch, seed=0, verbose=False,
        executor=executor,
    )
    return result, time.perf_counter() - start


def main():
    print(f"budget={BUDGET} batch={BATCH} workers={WORKERS}\n")

    sequential, t_seq = run(batch=1)
    print(f"batch=1, inline          {t_seq:6.2f}s   best={sequential['best_value']:.4f}")

    inline, t_inline = run()
    print(f"batch={BATCH}, inline          {t_inline:6.2f}s   best={inline['best_value']:.4f}")

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        threaded, t_thread = run(pool)
    print(f"batch={BATCH}, threads         {t_thread:6.2f}s   best={threaded['best_value']:.4f}"
          f"   speedup {t_inline / t_thread:.1f}x")

    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        procs, t_proc = run(pool)
    print(f"batch={BATCH}, processes       {t_proc:6.2f}s   best={procs['best_value']:.4f}"
          f"   speedup {t_inline / t_proc:.1f}x")

    print()
    ok = True
    for label, other in (("threads", threaded), ("processes", procs)):
        same = (
            other["best_value"] == inline["best_value"]
            and other["best_params"] == inline["best_params"]
            and other["loss_history"] == inline["loss_history"]
        )
        ok &= same
        print(f"{label:<10} identical to inline: {same}")

    if sequential["loss_history"] == inline["loss_history"]:
        print("WARNING: batch=1 and batch=8 produced identical runs")
        ok = False

    print("\nOK" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
