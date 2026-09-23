# Changelog

## 0.6.0 (unreleased)

**Seeded runs do not reproduce 0.5.1 results**, because the defaults changed.
Pass the old values explicitly to restore the previous behaviour.

### Added

- **`prune` and `prune_report`** reduce a search space using what a finished
  run learned. `prune(searchspace, result)` drops categorical branches the
  tree has abandoned and, with `narrow=k`, tightens continuous bounds to
  `mean ± k * std`. The output is an ordinary search space, so it feeds back
  into `BUTChC_optimize` or through `butchc.interop` to ConfigSpace, Optuna or
  SMAC. `return_tree=True` also returns a probability tree matching the pruned
  space, for use as `start_prob_tree`. Defaults are conservative: the branch
  that produced `best_params` is always kept, at least two choices survive per
  node, `threshold` is a fraction of the node's uniform share so it means the
  same thing at any arity, and narrowing is off. See
  [examples](docs/examples.md#pruning-a-space-and-handing-it-off).
- **Benchmark tooling.** `benchmarks/overhead.py` measures the optimiser's own
  per-trial cost with the objective stubbed out. `evaluate.py --anytime`
  reports trials needed to reach a target. `tune.py --regime NAME` tunes
  against problems sharing one property (`branched`, `highdim`, `multimodal`,
  `flat`, `noisy`) and prints a recommendation. `baselines.py` gains
  `flat_tpe_search`, TPE without knowledge of the conditional structure.

### Changed

Defaults re-selected by a full sweep (`benchmarks/tune.py 12 --rounds 2`),
confirmed on 30 seeds the selection never saw:

| Parameter | 0.5.1 | 0.6.0 |
|---|---|---|
| `alpha` | `10.0` | `3.0` |
| `gamma` | `0.7` | `0.85` |
| `explore` | `0.0` | `0.05` |
| `KDE_RESERVOIR_SIZE` | `50` | `25` |
| `MIN_BANDWIDTH_FRACTION` | `0.01` | `0.0003` |

Against the 0.5.1 defaults: 13 problems better, 4 worse, 8-1 on statistically
significant results, and 5 of 6 held-out problems improved.

| Problem | 0.5.1 defaults | 0.6.0 defaults | record |
|---|---|---|---|
| `Nested pipeline` | -0.0295 | **-0.0084** | 24-6, p=0.001 |
| `Optimiser choice` | -0.0005 | **-0.0001** | 26-4, p=0.000 |
| `Branch trap` | -0.0209 | **-0.0016** | 24-6, p=0.001 |
| `Categorical mix` | -0.0025 | **-0.0005** | 22-8, p=0.016 |
| `Rastrigin 8D` | -30.0803 | **-26.8328** | 16-14 |
| `20D sphere` | -0.0021 | -0.0746 | 4-26, p=0.000 |

`20D sphere` is the one significant regression. It costs no win against TPE —
BUTChC is still 253× nearer the optimum than TPE's -18.90 — and a smooth
high-dimensional space can recover it by setting `KDE_RESERVOIR_SIZE = 50` and
`MIN_BANDWIDTH_FRACTION = 0.001`. `KDE_RESERVOIR_SIZE` and
`MIN_BANDWIDTH_FRACTION` interact and should be retuned as a pair; see
[api.md](docs/api.md#how-the-bandwidth-floor-was-chosen-and-why-it-moved-five-other-defaults).

### Fixed

- **Categorical commitment depended on dict key order.** The sub-space size
  that slows a node's commitment could differ for the same space written with
  its keys in a different order. It now counts the most parameters one
  configuration can carry. No benchmark result changes.
- **ConfigSpace 1.x emitted a `DeprecationWarning` on every conversion.** The
  1.x accessors are now tried first.

### Documentation

- **`n_warmup`** was documented as updating nothing. Warm-up trials hold
  `quality` at 0, so no continuous archive moves, but categorical nodes still
  record their visit and rank. Warm-started runs do not skip warm-up; the
  default of 0 simply means there is none. Both behaviours are now tested.
- **`lambda_` saturates.** Weights depend on `lambda_` only through
  `min(RANK_SHARPNESS * lambda_, MAX_SHARPNESS)`, so every `lambda_` at or
  above 3.33 behaves identically. `MAX_SHARPNESS` was measured at the same
  time: raising it above 10 degrades results monotonically, so it stays.
- **Benchmarks re-measured** at 30 paired seeds against the new defaults:
  across 18 problems, 13 significant wins over TPE, 5 ties, and no significant
  loss. Anytime and per-trial cost results are new. Raw output is kept in
  `dev/tools/results/`.

---

## 0.5.1

A correctness release for the parts of the project that measure the library
rather than the library itself. Default search behaviour is unchanged: the
fingerprint over 17 seeded runs is byte-identical to 0.5.0.

### `NEUTRAL_QUALITY` was documented as patchable and did nothing

`optimizer` passed `neutral=0.5` as a literal to `update_tree`, which shadowed
`_update.NEUTRAL_QUALITY` on every call. Setting the constant to `0.0` or to
`0.9` produced byte-identical runs. It is documented in `docs/api.md` as a knob
and is now genuinely one; `benchmarks/tune.py` sweeps it over `[0.3, 0.5, 0.7]`.

The literal and the constant were both `0.5`, so nothing about a default run
changes. Any sweep that believed it had varied this knob was measuring noise.

This is the second instance of the same defect — `KDE_RESERVOIR_SIZE` was
half-patched in 0.4.0 — so `tests/test_constants.py` is new and asserts that
every constant in the `docs/api.md` table still reaches the search. Nine
constants, one test each, plus a check that what `tune.py` sweeps and what the
docs claim it sweeps are the same set.

### The benchmark tables did not reproduce

`README.md` published medians measured before 0.4.0's first-observation rank
fix (see below), on the four problems that contain a categorical node. Running
`evaluate.py 30` against the shipped code returned different numbers for
`Branch trap`, `Categorical mix`, `Nested pipeline` and `Optimiser choice`, and
reverting the rank fix reproduced the published values exactly, which is what
identifies the cause.

All tables are re-measured against 0.5.1 at 30 paired seeds. The `v0.3` column
is replaced by `TPE`, which is the comparator a user choosing against BUTChC
would actually reach for; the separate 5-seed TPE table it supersedes is gone,
and every median and win-loss record now carries a sign-test p-value.

Running TPE on the held-out suite for the first time changed one of the
README's claims. BUTChC loses to TPE on `Nested pipeline` (10-20) and
`Optimiser choice` (12-18) — both conditional problems, which is the shape the
method is meant to suit. Neither is significant at 30 seeds, and `Branch trap`
is still a 23-7 win, but "conditional structure is BUTChC's category" is no
longer a claim the benchmarks support, and the README no longer makes it.

### ConfigSpace is now tested against a real install

`tests/test_interop.py` ran entirely against stubs, so it could not catch a
misreading of the ConfigSpace API — only a misreading of the tree assembly.
Seven tests now exercise a real `ConfigurationSpace`: conditions to branches,
constants to `fixed`, forbidden-clause refusal, name qualifying, round-tripping
and sampling. They skip when ConfigSpace is absent, so the suite stays
dependency-free, and run under `pip install .[configspace]`.

Detection of a `ConfigurationSpace` passed to `BUTChC_optimize` no longer hinges
on `get_default_configuration` surviving a major version: the type's defining
module is checked first, with the accessor checks kept as fallbacks for
subclasses and wrappers. This only affects objects that previously raised
`TypeError`.

### Docs

- The module-constants table in `docs/api.md` gains **Patch in** and **Swept**
  columns. `MIN_GRID_POINTS` has the same by-name import hazard as
  `KDE_RESERVOIR_SIZE` and its second binding is now named.
- `docs/examples.md` and `dev/eval/README.md` no longer describe the
  ConfigSpace tests as stub-only.

### Still outstanding

The full tuning sweep has not been re-run since the first-observation rank fix,
so the shipped defaults were selected under the older rule. `benchmarks/tune.py
12 --rounds 2` is the command; expect hours. The sweep now includes `neutral`,
which no previous sweep can have covered.

---

## 0.5.0

Additive. No behavioural change to existing calls: a fingerprint over 17 seeded
runs spanning categorical, continuous, log, integer, nested-conditional, prior,
NaN and warm-start paths is byte-identical to 0.4.0.

### Batch and parallel evaluation

Two new keyword arguments on `BUTChC_optimize`:

| Argument | Default | Meaning |
|---|---|---|
| `batch` | `1` | Configurations drawn per model update. `-1` resolves to `os.cpu_count()`. |
| `executor` | `None` | Anything with an order-preserving `map(fn, iterable)`, or `submit(fn, arg)` returning futures. |

`batch=1` is unchanged from 0.4 in every respect. BUTChC never creates a pool;
you supply the executor, so the package stays dependency-free and the
picklability decision stays with whoever wrote the objective.

Three rules make the batched path correct:

- Every batch member is ranked against the observations that existed when it
  was **sampled**, never against its batch-mates.
- Updates are applied in **sample order**, never completion order, so a seeded
  run gives the same answer whichever worker finishes first.
- The final batch is **truncated**, so a budget is exactly a budget.

Measured cost at fixed budget across 18 problems and 20 seeds: `k=8` costs
about 2.6% in median regret, `k=16` about 21%, `k=32` about 43%. Reproduce with
`benchmarks/batch_cost.py`.

### `butchc.interop`

Optional conversions, not imported by `butchc`, so ConfigSpace never becomes a
dependency of the core package. `pip install butchc[configspace]`.

- `from_configspace(space, drop=())` — promoted from `dev/eval/configspace_bridge.py`.
  Refuses conjunctions, multiple parents, forbidden clauses and non-categorical
  parents with `UnsupportedSpace` naming the parameter.
- `to_configspace(searchspace)` — new, the reverse direction. Returns
  `(space, names)`; BUTChC permits a name to repeat across sibling branches and
  ConfigSpace does not, so collisions are qualified as `parent:value:name` and
  `names` maps back.
- `wrap_objective(objective, fixed, casts)` — restores constants and applies
  casts. Now a picklable class rather than a closure, so a wrapped objective
  works with a process pool.
- `to_json` / `from_json` — round-trip a search space. `from_json` re-keys
  `next_level` against its node's `values`, because JSON stringifies object keys
  and a space branching on integers would otherwise come back with its branches
  silently unreachable.

`BUTChC_optimize` accepts a `ConfigurationSpace` directly wherever a dict is
expected, detected by duck-typing, and wraps the objective for you.

### Return value

- **New:** `n_batches`, the number of model updates performed. Equals `budget`
  when `batch=1`.
- **New:** `batch_index` on every `history` entry.

Code asserting on the exact key set of the result dict needs updating. Nothing
else changed.

### Docs

`docs/butchc_docs_detailed.md` is replaced by three focused documents:
[api.md](docs/api.md), [examples.md](docs/examples.md) and
[design.md](docs/design.md). Several statements in the old document had gone
stale — the categorical probability formula, the claim that priors are folded
into pseudo-counts, the claim that `lambda_` is annealed, and the claim that
`loss` is zero on trials failing the gate — and are corrected rather than
carried across.

---

## 0.4.0

### Defaults changed

Selected by a benchmark sweep (`benchmarks/tune.py`). **Seeded runs do not reproduce 0.3 results.** Pass the old values explicitly to restore the previous behaviour.

| Parameter | 0.3 | 0.4 |
|---|---|---|
| `lambda_` | `1.0` | `2.0` |
| `alpha` | `1.0` | `10.0` |
| `gamma` | `0.5` | `0.7` |
| `explore` | `0.1` | `0.0` |
| `n_warmup` | `budget // 10` | `0` |

### The first trial no longer scores a perfect rank

`_quantile_rank` returned `1.0` when there was nothing to compare against, so
whichever branch trial 1 happened to draw was credited with the best possible
score before any comparison existed. The continuous archive ignored it, but
categorical nodes consume every rank, so the bias was real and largest on the
smallest budgets. It now returns `0.5`, the rank of a configuration drawn at
random.

**Seeded runs before this change do not reproduce**, on any space containing a
categorical node. The benchmark tables published for 0.4 were measured before
it and were corrected in 0.5.1.

### `lambda_` and `gamma` changed meaning

Both now apply to continuous nodes only. `lambda_` sets the archive's rank sharpness rather than a categorical learning rate; `gamma` gates the archive rather than the whole update. Categorical nodes score every trial by rank.

### Warm starting

Probability trees from 0.3 and earlier **cannot be loaded** — categorical nodes now carry `visits`, `prior` and `prior_strength`. `BUTChC_optimize` reports this rather than silently producing nonsense. Re-run instead.

Warm starting also verifies that the tree's continuous bounds and scales match the search space, so a mismatch that previously passed silently is now an error.

### Reporting

`n_gated` is new, alongside `n_updates`; `history` entries carry a matching `gated` flag. `n_updates` counts trials that shifted some distribution, `n_gated` counts trials that cleared the quality gate and so reached a continuous archive. Previously categorical movement was recorded only on trials that cleared the gate, which under-reported movement on categorical-heavy spaces.

---

## 0.2.0

The public signature stayed compatible — existing 0.1.x calls keep working, and `lambda_`/`alpha` gained defaults. Behavioural changes:

- **The objective now affects the model.** In 0.1.x the update rule never received the objective value, so the tree reinforced every sampled configuration equally: optimizing `f` and `-f` produced identical trees, and results were worse than random search. Expect substantially different, and better, behaviour on the same settings.
- **`start_prob_tree` from 0.1.x is not loadable.** Continuous nodes carry a `scores` field. Re-run rather than migrate.
- **Malformed search spaces raise** instead of failing partway through a run.
- **Duplicate parameter names raise.** In 0.1.x these silently corrupted the tree or crashed with a bare `KeyError`.
- **Extra objective arguments must be keyword.** The old `*args` slot collided with `temp` positionally.
- **Seeding** is via `seed=`; the library no longer depends on global `random` state.
- New: `prior`, `prior_strength`, `log`, `int`, `gamma`, `explore`, `n_warmup`, `n_updates`, `reservoir_summary`.
