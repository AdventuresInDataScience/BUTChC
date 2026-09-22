# Changelog

## 0.6.0 (unreleased)

### The package directory was unimportable on a case-insensitive filesystem

The working tree carried the package as `BUTChC/` while git tracks it as
`butchc/`. Python's import machinery compares filenames case-sensitively even
where the filesystem does not, so `import butchc` failed and the entire test
suite ended in nine collection errors. Renamed back to `butchc/`, which is what
`pyproject.toml`, every import in `tests/`, `benchmarks/` and `dev/`, and every
document already said. `core.ignorecase` is true here, so git records no change.

Nothing was wrong with the committed tree — only the checkout. Worth knowing
because the reverse is not harmless: committing new files while the directory
was misnamed could have put a second casing into the index, which is invisible
on Windows and fatal on Linux.

### `_subspace_size` depended on dict ordering

It accumulated a running count into the same variable a `max` also wrote to, so
the same logical space scored differently depending on which key its dict
yielded first — `{A, B, C -> {S}}` gave 3 or 4. The value divides the
commitment exponent (`max(sharpen, 1) / (1 + sub_size)`), so two identical
spaces written in a different key order committed to branches at different
rates.

It now counts the most parameters one configuration can carry: every parameter
at a level, plus the largest branch of each branching parameter. Siblings that
both branch both contribute; branches of a single choice do not, since they
never co-occur.

Latent on everything measured. All three conditional benchmark problems hold
flat sub-spaces, where the accumulator and the `max` coincide, and the
fingerprint over 17 seeded runs is byte-identical either way. No published
number changes. `tests/test_tree.py` gains `TestSubspaceSize`.

### `n_warmup` was documented as updating nothing, and never did that

Three documents and the docstring said warm-up trials update nothing. They
update categorical nodes, deliberately: choices are scored by mean rank over
every visit, and warm-up is the unbiased sample that stops a branch being
written off before it has been tried — which `optimizer.py` explains at the
call site while the docs claimed the opposite. A run that is entirely warm-up
moves `prob` from uniform to `{a: 0.008, b: 0.048, c: 0.944}` on an ordered
three-way choice and reports 29 of 30 trials as updates.

What warm-up actually suppresses is the continuous archive, by holding
`quality` at 0. Corrected in `butchc/optimizer.py`, `docs/api.md` and
`docs/design.md`.

`docs/examples.md` separately claimed warm-started runs skip warm-up. There is
no such detection; `n_warmup` is honoured as given, and the claim was only
vacuously true because the default is 0. Both behaviours are now pinned by
tests in `tests/test_optimizer.py`.

### `dev/tools/codemap_gen.py` could not run on Windows

Two defects, both platform-specific, and the second silently produced a wrong
document rather than failing:

- Source files were opened at the locale encoding. On Windows that is cp1252,
  which cannot decode the proportional sign in `_update.py`'s docstring, so the
  generator `docs/codemap.md` tells the reader to run crashed outright. Now
  opened as UTF-8, and stdout is reconfigured to UTF-8 so the em-dashes survive
  the documented `>` redirect.
- `all_files()` built paths with `os.path.join`, but every path test in the
  module (`startswith("butchc/interop/")`, `split("/")`, `replace("/", ".")`)
  treats `/` as the separator. On Windows all of them missed, and the tables
  came out with the cross-module call graph — the part the document exists for
  — almost entirely empty.

`docs/codemap.md`'s line numbers are re-synced against the source.

### ConfigSpace accessors probed in the wrong order

`_hyperparameters`, `_conditions` and `_forbiddens` tried the ConfigSpace 0.x
spelling first. On 1.x those still exist and merely emit a
`DeprecationWarning`, so the fallback never fired and every conversion warned
— fourteen warnings across the suite, attributed to the caller's code, for a
branch that was never taken. The 1.x spelling is now tried first, which also
fails in the safe direction when 2.x removes the old names.

### Measured: 13 significant wins against TPE, zero significant losses

Re-measured at 30 paired seeds against the new defaults. Across all 18
problems, BUTChC takes 13 statistically significant wins over TPE, 5 ties, and
no significant loss. Both results 0.5.1 conceded are closed: `Optimiser choice`
12-18 to **25-5**, `Nested pipeline` 10-20 to a **15-15 tie** with equal
medians. `README.md` carries the tables.

Speed of arrival was measured for the first time as more than a footnote.
BUTChC matches TPE's *final* answer partway through its own budget on every
problem it reaches — `20D sphere` in 148 trials of 1500 (1/10.1), `10D sphere`
1/7.3, median around a third of budget. Against TPE's own arrival time it is
much quicker in high dimensions (148 trials against 1133 on `20D sphere`, on
30 of 30 seeds against TPE's 15) and still slower on the conditional problems
(`Nested pipeline` 324 against 188). The 0.5.1 README's blanket "arrives more
reliably and later" was half wrong and is replaced.

New: `benchmarks/overhead.py` measures what the optimiser itself costs, with
the objective stubbed to a constant. BUTChC is **73-145x cheaper per trial**
than either TPE — 193 us against Optuna's 27,864 us at 20 dimensions. Zero
dependencies costs nothing here; the numpy-backed implementation pays ~145x
more per suggestion because TPE refits Parzen estimators over the whole
history. It also means ~99% of this suite's wall clock is the baselines, not
the library under test.

Raw output is kept in `dev/tools/results/`, regenerated by
`dev/tools/publish_tables.py`, so every number above has a recorded provenance.

### Defaults changed — re-selected by a full sweep

**Seeded runs do not reproduce 0.5.1 results.** Pass the old values explicitly
to restore the previous behaviour.

| Parameter | 0.5.1 | 0.6.0 |
|---|---|---|
| `alpha` | `10.0` | `3.0` |
| `gamma` | `0.7` | `0.85` |
| `explore` | `0.0` | `0.05` |
| `KDE_RESERVOIR_SIZE` | `50` | `25` |
| `MIN_BANDWIDTH_FRACTION` | `0.01` | `0.0003` |

Selected by `benchmarks/tune.py 12 --rounds 2`, confirmed on 30 seeds the
selection never saw, and measured against the previous defaults on both suites:
13 problems better, 4 worse, 8-1 on statistically significant results, and 5 of
6 held-out problems improved.

The two results that matter most are the two the 0.5.1 README conceded to TPE,
both held out, both now won:

| Problem | 0.5.1 defaults | 0.6.0 defaults | record |
|---|---|---|---|
| `Nested pipeline` | -0.0295 | **-0.0084** | 24-6, p=0.001 |
| `Optimiser choice` | -0.0005 | **-0.0001** | 26-4, p=0.000 |
| `Branch trap` | -0.0209 | **-0.0016** | 24-6, p=0.001 |
| `Categorical mix` | -0.0025 | **-0.0005** | 22-8, p=0.016 |
| `Rastrigin 8D` | -30.0803 | **-26.8328** | 16-14 |
| `20D sphere` | -0.0021 | -0.0746 | 4-26, p=0.000 |

`20D sphere` is the one significant regression and it is reported here as
measured. It costs no win: at -0.0746 BUTChC is still 253x nearer the optimum
than TPE's -18.90 and 925x nearer than random search, so the problem remains a
30-0 sweep. The trade — a still-dominant margin on a problem already won, for
two genuine losses becoming wins — is the one the sweep's aggregate record
reflects. A smooth high-dimensional space can recover the difference by
reverting `KDE_RESERVOIR_SIZE` to 50 and `MIN_BANDWIDTH_FRACTION` to 0.001;
`docs/api.md` says so where the constants are listed.

Five reverts were tested individually against the chosen configuration before
accepting it, and none removed the `20D sphere` cost without giving up more
elsewhere. `explore=0.05` turned out to be *protecting* that problem: removing
it takes the median to -2.95, because the sharper archive the other defaults
create needs a floor under it in high dimensions.

### `lambda_` saturates, and nothing said so

Archive weights depend on `lambda_` only through
`min(RANK_SHARPNESS * lambda_, MAX_SHARPNESS)`, so at the shipped
`RANK_SHARPNESS = 3.0` every `lambda_` at or above 3.33 is clipped to the same
value. `lambda_=4.0` and `lambda_=8.0` return medians identical to four
decimals; so do `lambda_=1.0` and `rank_sharpness=1.5`, which have the same
product. The two constants are one degree of freedom, and `docs/api.md`
documented `lambda_` as an open-ended `float > 0` with no mention of a ceiling
— so "raise `lambda_` to exploit harder" was advice that silently stopped
working above 3.33.

Documented in `docs/api.md` with the measurement, and added to
`docs/design.md`'s open questions beside the `temp`/`COMMITMENT` collapse it
mirrors. No behaviour change.

`MAX_SHARPNESS` was measured at the same time to check whether the ceiling was
itself the binding constraint. It is not: across all 18 problems at 20 paired
seeds, raising it degrades monotonically — 14.0 loses 64-288, 20.0 loses 9-349,
30.0 loses 6-352. The shipped 10.0 stays.

### The bandwidth floor looks too high, and it is the biggest result here

Every one of the five problem-shape sweeps picked `MIN_BANDWIDTH_FRACTION =
0.003` over the shipped `0.01`. A knob selected by every regime is not a regime
finding. Measured directly across all 17 problems at 20 paired seeds, `0.003`
beats `0.01` **185-38**, including 20-0 sweeps on `5D sphere`, `10D sphere`,
`Ackley 5D`, `Styblinski 4D` and `20D sphere` — the last two held out. `Ackley
5D` moves from -1.7597 to -0.3745, and to -0.1159 at `0.001`.

The default is unchanged in this release. Every other default was selected with
the floor at `0.01`, so changing it invalidates that selection; the fix is a
full re-sweep and a regeneration of every published table, not a one-line edit.
`docs/api.md` carries the measurements and the one-line override.

### Tuning by problem shape

`benchmarks/tune.py --regime NAME` tunes against problems sharing one property
— `branched`, `highdim`, `multimodal`, `flat`, `noisy` — and prints a
recommendation rather than a proposed default. Recommendations for all five are
in `docs/api.md`, measured at 6 selection seeds and one round, which is enough
to indicate a direction and not enough to settle a default.

### A flat baseline, and what it failed to show

`benchmarks/baselines.py` gains `flat_tpe_search`: the same TPE denied any
knowledge of the conditional structure, searching the union of every branch's
parameters on every trial. It is what an optimiser without a conditional schema
costs you, and the difference against `tpe_search` isolates that cost.

On the current problems it isolates almost nothing. `Branch trap` medians are
-1.0005 flat against -1.0004 hierarchical; `Categorical mix` is -0.0680 for
both. Only `Nested pipeline` separates them, -0.0570 against -0.0096. These
spaces are too narrow for the waste to matter — the union is only a few
parameters wider than a branch. Demonstrating the effect needs a genuinely wide
space, which the suite does not yet contain. The baseline is shipped anyway, so
the comparison is available once such a problem exists.

### Anytime measurement

`evaluate.py --anytime` reports median trials to reach 95% of the achievable
range, anchored per problem between random search's median and the best any
method reached. Trials rather than seconds, because trials are what an
expensive objective charges for.

The result complicates the case for BUTChC on small budgets rather than
supporting it. On `Categorical mix` it reaches the target on 26 of 30 seeds
against 8 of 30 for both TPE variants — but takes 200 trials to their 132. On
`Optimiser choice`, 239 against 93. BUTChC arrives more reliably and later.
Reliability of arrival is a real property; "90% of the result in 10% of the
trials" is not supported by these measurements.

### `prune` — reduce a space using what a run learned

New: `prune(searchspace, result)` returns a smaller search space, dropping
categorical branches the tree has abandoned and optionally narrowing continuous
bounds to the learned distribution. `prune_report` describes what changed.
Output is an ordinary search space, so it feeds back into `BUTChC_optimize` or
through `butchc.interop` to ConfigSpace, Optuna or SMAC.

The motivation is in the benchmarks rather than in theory. BUTChC wins
decisively where a branch must be *rejected* (`Branch trap` 10-2 at budget 100,
`Categorical mix` 9-3) and loses where a branch must be *tuned well once
chosen* (`Optimiser choice` 3-9, `Nested pipeline`). Scouting with BUTChC and
handing the survivors to a refiner uses the half that measures well.

Defaults are conservative because the failure is one-directional — pruning too
little wastes budget, pruning away the branch holding the best result loses the
answer:

- the branch that produced `best_params` is kept whatever its probability,
- at least two choices survive per node,
- `threshold` is a fraction of the node's uniform share rather than a bare
  probability, so it means the same thing at any arity,
- continuous narrowing is off by default, because a dropped branch is visible
  in `prune_report` and a bound narrowed past the optimum is not.

Two limits found while testing and documented rather than papered over.
Pruning needs budget *per branch*: on a twelve-way node with 200 trials the
useless options are still near uniform and nothing is dropped at all. And a
branch's probability is not a claim that the branch is empty — `Branch trap`
exists precisely because a poor average can hide an excellent best.

**Fixed before release:** the documented warm-start pattern —
`start_prob_tree=scout['prob_tree']` against the pruned space — raised
`ValueError` on every call that actually pruned anything. `scout['prob_tree']`
still describes the *original* space, and `BUTChC_optimize` requires an exact
match between a warm-start tree and the space it is given. `prune` now takes
`return_tree=False`; passing `True` returns `(searchspace, prob_tree)`, where
`prob_tree` carries the learned statistics for only the survivors and is
guaranteed to match. Default behaviour and return type are unchanged for
existing callers. `docs/api.md` and `docs/examples.md` are corrected to match;
`tests/test_prune.py` gained `TestReturnTree`.

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
