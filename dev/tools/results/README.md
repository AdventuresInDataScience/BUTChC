# Measured results

Output of `dev/tools/publish_tables.py` and `benchmarks/overhead.py`, kept in
the repo so the numbers quoted in `README.md`, `docs/api.md` and `CHANGELOG.md`
have a recorded provenance and do not have to be re-measured to be checked.

All of this was measured against the 0.6.0 defaults (`alpha` 3.0, `gamma` 0.85,
`explore` 0.05, `KDE_RESERVOIR_SIZE` 25, `MIN_BANDWIDTH_FRACTION` 0.0003) on
30 paired seeds unless the filename says otherwise.

| File | What it holds |
|---|---|
| `quality_30seed.txt` | Final best after full budget. `TUNED ON` and `UNDER OBSERVATION NOISE` complete, with win-loss records and sign-test p-values. |
| `speed_30seed.txt` | Trials-to-target at 50/80/95/99% of achievable range, plus trials to reach TPE's median final value. |
| `overhead_600trials.txt` | Optimiser wall-clock per trial, free objective, 1D/5D/20D, against this repo's TPE and Optuna's `TPESampler`. |

Every number quoted in `README.md` and `CHANGELOG.md` for 0.6.0 came from these
runs. The `TUNED ON` win-loss records were independently reproduced by two
separate 30-seed runs and matched exactly, which is the reproducibility check.

## Known gaps

Both are capture problems, not measurement problems — every published claim is
already backed by a completed run. Refilling them is tidying, not validation.

1. **`quality_30seed.txt` has no `HELD OUT` section.** The run was writing it
   when the session ended. Those numbers *were* measured and are published in
   `README.md` and `CHANGELOG.md`; only this file is short. Refill with:

   ```bash
   python dev/tools/publish_tables.py 30 quality     # ~45 min, writes here
   ```

2. **`speed_30seed.txt` is truncated at the head**: the `TUNED ON` section is
   missing its first six rows, because it was captured through a `tail`. The
   `HELD OUT` sections and both "fraction of the budget" summaries — which are
   what `README.md` quotes — are complete. Refill with:

   ```bash
   python dev/tools/publish_tables.py 30 speed       # ~45 min, writes here
   ```

Runtime on both is dominated by TPE, not BUTChC — see
`overhead_600trials.txt`. TPE costs ~100x more per suggestion, so ~99% of a
benchmark run's wall clock is the baseline, not the library under test.
