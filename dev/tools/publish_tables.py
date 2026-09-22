"""Regenerate the published benchmark tables into files that survive a restart.

`benchmarks/evaluate.py` prints to stdout and collects every result before
printing anything, so a 30-seed run produces no output for over an hour and
then all of it at once. Run from a session that ends and the whole thing is
lost. This writes each table to `dev/tools/results/` as it completes, so the
work is recoverable and a later session can read the files instead of
re-measuring.

    python dev/tools/publish_tables.py            # 30 seeds, quality + speed
    python dev/tools/publish_tables.py 10         # faster, less reliable
    python dev/tools/publish_tables.py 30 quality # one section only

Runtime is dominated by TPE, not by BUTChC: TPE costs ~24 ms per suggestion
against BUTChC's ~0.2 ms, so one seed over TUNING+HELDOUT is ~95 s of TPE and
30 seeds is ~47 min of it. Budget accordingly.
"""
import contextlib
import io
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path[:0] = [os.path.join(ROOT, "benchmarks"), ROOT]

OUT_DIR = os.path.join(HERE, "results")


def write(name, text):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"  wrote {os.path.relpath(path, ROOT)} ({len(text)} bytes)", flush=True)


def captured(fn, *args, **kwargs):
    """Run fn, return its stdout as a string and also echo it."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*args, **kwargs)
    text = buf.getvalue()
    print(text, flush=True)
    return text


def main(argv):
    n_seeds = int(argv[0]) if argv else 30
    sections = argv[1:] or ["quality", "speed"]

    import evaluate
    from problems import HELDOUT, NOISY, TUNING

    methods = ["random", "tpe", "butchc"]
    seeds = list(range(n_seeds))
    started = time.time()

    if "quality" in sections:
        print(f"\n=== quality: final best after full budget, {n_seeds} seeds ===",
              flush=True)
        parts = []
        for title, suite, noisy in (
            ("TUNED ON (optimistic)", TUNING, False),
            ("UNDER OBSERVATION NOISE — true value of the reported best", NOISY, True),
            ("HELD OUT (not consulted while tuning)", HELDOUT, False),
        ):
            parts.append(captured(
                evaluate.report_table, title,
                evaluate.collect(suite, methods, seeds, noisy=noisy), methods))
            write(f"quality_{n_seeds}seed.txt", "\n".join(parts))

    if "speed" in sections:
        print(f"\n=== speed: trials to target, {n_seeds} seeds ===", flush=True)
        parts = []
        for title, suite in (("TUNED ON — anytime", TUNING),
                             ("HELD OUT — anytime", HELDOUT)):
            table = evaluate.collect_curves(suite, methods, seeds)
            for fraction in (0.50, 0.80, 0.95):
                parts.append(captured(evaluate.report_anytime,
                                      f"{title} @ {fraction:.0%}",
                                      table, methods, fraction))
            write(f"speed_{n_seeds}seed.txt", "\n".join(parts))

    print(f"\ntotal {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
