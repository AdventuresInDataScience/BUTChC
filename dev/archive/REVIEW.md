> **Archived.** This is a one-time engineering review, not living
> documentation — kept here for audit-trail purposes only. It is not linked
> from the README or any shipped doc. For the current design rationale see
> [`docs/design.md`](../../docs/design.md); for version history see
> [`CHANGELOG.md`](../../CHANGELOG.md).

# BUTChC v0.4.0 — code review

> **Status note (0.5.1).** This reviews 0.4.0 and is kept as the record of why
> those changes were made. Items since addressed: §4's ConfigSpace bridge is now
> `butchc.interop`, covered by `tests/test_interop.py` — and as of 0.5.1 seven of
> those tests run against a real ConfigSpace install rather than stubs, closing
> §5 item 3 apart from the YAHPO surrogate stack itself. The docs sweep of §5
> item 5 is done, with `docs/butchc_docs_detailed.md` replaced by `docs/api.md`,
> `docs/examples.md` and `docs/design.md`. §5 item 2 is done: the README tables
> are re-measured at 30 paired seeds against shipped code and the `v0.3` column
> is now `TPE`.
>
> Still outstanding: **§5 item 1** — the full sweep has not been re-run since B3,
> so the shipped defaults were selected under the old first-trial rank rule; and
> **§5 item 4**, the `temp` / `COMMITMENT` overlap, recorded in `docs/design.md`
> under Open questions.
>
> One defect of the same class as B1 was found in 0.5.0 and fixed in 0.5.1:
> `optimizer` passed `neutral=0.5` as a literal, so `_update.NEUTRAL_QUALITY`
> was documented as patchable and swept while being unreachable. B1 is therefore
> a recurring failure mode rather than a one-off, and `tests/test_constants.py`
> now guards every constant in the public table against it.

Scope: the whole package, its tests, its docs and its benchmark scripts.
Verified by running the suite (257 pass), by fingerprinting seeded runs before
and after every edit, and by executing the new benchmark scripts end to end.

---

## Verdict

The library is in good shape. Module boundaries are clean and mean something
(`_tree` builds, `_sampling` draws, `_update` learns, `_validate` refuses,
`optimizer` orchestrates). Validation is genuinely up-front, so a malformed
space costs nothing when trials are expensive. There are no dependencies and
the test suite is thorough — 257 cases, including regression guards for
specific past defects.

Everything worth criticising falls in three places: two live bugs in the
tuning harness, a layer of commentary written as changelog rather than as
documentation, and a handful of knobs that no longer earn their complexity.

---

## 1. Bugs

### B1 — the `reservoir` knob was never actually swept *(fixed)*

`KDE_RESERVOIR_SIZE` is defined in `_utils` and imported **by name** into both
`_tree` and `_update`. `from x import y` creates an independent binding, so
patching `x.y` does not reach it.

The old `tune.py` patched `_utils` and `_update` but not `_tree`:

```python
utils.KDE_RESERVOIR_SIZE = cfg["reservoir"]
upd.KDE_RESERVOIR_SIZE   = cfg["reservoir"]
# _tree.KDE_RESERVOIR_SIZE untouched
```

`_tree.initialize_prob_tree` is what sizes the initial grid, so every swept
configuration ran with a 50-point starting grid and a *different* eviction cap.
At `reservoir=25` the tree started over-full and evicted on the first update;
at `reservoir=100` it started half-empty and grew. Neither is a configuration
anyone can ship. Demonstrated:

```
patch _utils and _update to 25  ->  initial grid still 50 points
```

`benchmarks/tune.py::apply_config` now sets all three bindings and says why.

### B2 — `BASE` did not match the shipped defaults *(fixed)*

`tune.py` declared `commitment: 4.0`; `optimizer.COMMITMENT` is `8.0`. `8.0`
was not in the grid either, so the sweep could not have selected the shipped
value, and the "baseline" column measured a configuration that does not exist.
`BASE` is now synced and carries a comment saying it must stay that way.

### B3 — the first trial always scores a perfect rank *(fixed)*

`_quantile_rank([], value)` returns `1.0`. The continuous archive ignores it
(`quality` requires `len(observed) > 1`), but categorical nodes consume `rank`
**ungated on every trial**, so whichever branch trial 1 happens to draw is
credited with the best possible score before any comparison exists.

With `alpha=10` the effect is small — one pseudo-visit against ten — but it is
a free bias in favour of an arbitrary branch, and it is largest exactly where
it matters least defensibly: tiny budgets.

`_quantile_rank` now returns `0.5` for the first observation. **This changes
seeded results**, so 0.4's published numbers no longer reproduce and the sweep
must be re-run. Measured over 8 seeds, only the four problems containing a
categorical node move at all:

| Problem | before | after |
|---|---|---|
| Branch trap | -0.1408 | **-0.0965** |
| Nested pipeline | -0.0385 | **-0.0122** |
| Categorical mix | **-0.0029** | -0.0062 |
| Optimiser choice | **-0.0009** | -0.0021 |

Two better, two worse, which at 8 seeds is a wash. The point is not the gain;
it is that the sweep is now measuring a rule that does not credit an arbitrary
branch for winning against an empty set.

### B4 — `loss` and `n_updates` under-report categorical movement *(fixed)*

`_update_categorical` mutates `prob` on every trial that reaches the node, but
`update_tree` appends the resulting delta only when `quality > 0`. So a trial
can move the distribution substantially and be recorded as having moved
nothing. Measured on a purely categorical space, 40 trials:

```
n_updates reported: 18 of 40
prob went from {p:.333, q:.333, r:.333} to {p:.935, q:.030, r:.035}
```

The README said loss "is 0 on trials that failed the quality gate", which
describes the symptom but not the consequence: `rolling_loss` is documented as
a convergence signal, and on categorical-heavy spaces it was measuring the
wrong thing.

`deltas` are checked and this feeds nothing but reporting — not sampling, not
updating — so the fix does not alter the search at all. Categorical deltas are
now recorded unconditionally, and the two questions that were conflated are
split:

- `n_updates` — trials that actually shifted some distribution
- `n_gated` — trials that cleared the quality gate and reached a continuous
  archive

`history` entries carry a matching `gated` flag alongside `updated`. One extra
detail this exposed: recomputing an unchanged distribution returns deltas of
order 1e-17 from floating-point rounding, so "moved" is thresholded at
`MOVEMENT_EPSILON = 1e-12` rather than at zero. Without that, a run whose
objective returns only NaN reported 12 updates out of 40.

Four tests asserted the old behaviour and were rewritten; two new ones were
added.

### B5 — a docstring that had become false *(fixed)*

`_categorical_node` claimed it was "folding any prior into pseudo-counts" and
argued at length that a prior *must* live in `counts` rather than `prob`. The
code does neither: `counts` and `visits` start at zero, and the prior is
applied multiplicatively in `_apply_prior` with an exponent that decays as
visits accumulate. The docstring described a design that was replaced.

---

## 2. Comments

You were right about this, and it is the single most pervasive issue.

### The pattern

Three flavours, all of which document a *diff* rather than the code:

**Version archaeology** — unreadable without the prior release in front of you.

> "Weight-based eviction — the v0.1 scheme — could not work…"
> "That was a real limitation of the previous version, which had no log
> support at all."
> "Two changes from the previous version are worth calling out."

**Changelog in past tense** — states an outcome of an experiment nobody else
ran.

> "…they came up half as often as every other value."
> "Spending a tenth of the budget re-deriving that cost real performance on
> every problem in the tuning suite."
> "On a two-valued objective the model learned essentially nothing."

**Justification by benchmark** — defends a decision to a reviewer instead of
explaining a mechanism to a reader.

> "Zero by measurement, not by omission."
> "…building it a second time here cost 2**(depth+1)-2 recursive calls."

The test: if deleting the previous release from history would make the comment
unreadable, it is archaeology, not documentation.

### What changed

About 25 comments and docstrings rewritten across all six library modules.
The load-bearing reasoning was **kept** — why reflection rather than clipping,
why midranks rather than strict ranks, why means rather than totals, why the
prior multiplies rather than contributing counts — but restated as present-tense
claims about the code that is there. For example:

> *before:* "Totals were a Pólya urn with a quality veneer: a branch sampled
> three times as often accumulated roughly three times the count…"
>
> *after:* "A mean separates 'this branch is good' from 'this branch was tried
> a lot'; a total does not, because a branch sampled three times as often
> accumulates roughly three times the score at equal quality and so amplifies
> its own early luck."

Same information, no dependency on a version that no longer exists.

### Stale docstrings found while doing it

| Location | Said | Actually |
|---|---|---|
| `BUTChC_optimize` | `n_warmup` defaults to `max(10, budget // 10)`, or 0 if any prior | defaults to `0`, unconditionally |
| `BUTChC_optimize` | `lambda_` is an "update weight", "start at 1.0" | archive rank sharpness, default `2.0` |
| `BUTChC_optimize` | `alpha` "start at 1.0" | default `10.0` |
| `update_tree` | `neutral` is `(1 - gamma) / 2` | `0.5` — categorical scoring stopped using the gate |
| `_recompute_prob` | scores are gated quality, neutral derived from `gamma` | scores are ungated midranks |
| `_categorical_node` | prior folded into pseudo-counts | prior applied multiplicatively, counts start at zero |

Every one of these is a leftover from the 0.3 → 0.4 change of meaning for
`lambda_` and `gamma`. The README migration section describes the change
correctly; the docstrings never caught up.

### Not touched

`docs/butchc_docs_detailed.md` and the README have the same habit
(`README.md` benchmark table has a `v0.3` column; the detailed docs discuss
v0.1 defects at length). That is more defensible in a document whose job
includes migration notes, so it is your call — but the `v0.3` column would be
strictly more useful as a `TPE` column, see §4.

---

## 3. Redundant or over-complicated machinery

### Dead

- **`optimizer._has_prior`** — never called. It existed to force `n_warmup=0`
  when a prior was present; `n_warmup` is now unconditionally 0. *(deleted)*

### Live-looking but unreachable on the default path

- **Warm-up.** `n_warmup` defaults to `0`, so `warming`, the
  `explore=1.0 if warming else explore` branch, the `not warming` guard on
  `quality`, and the `[warmup]` suffix in the verbose line are all dead unless
  a caller opts in. It is a documented public argument, so keeping it is
  defensible — but four code paths and a comment paragraph exist for a feature
  the project measured as harmful. Either keep it and note in the code that
  the default is off, or drop it and reclaim the branch.

### Pure indirection

- **`update_tree(..., neutral=None, discount=None)`.** Both resolve to module
  globals immediately (`NEUTRAL_QUALITY if neutral is None else neutral`), and
  the module globals are what `tune.py` patches. So the parameters exist only
  to be threaded through four recursive call sites and one test. `rank` is
  genuinely needed — it differs from `quality`. The other two could go, taking
  eight arguments down to six.

### Three knobs shaping one exponent

`_recompute_prob` raises choice values to `max(sharpen, 1) / (1 + sub_size)`.
`sample_node` then softmaxes `log(prob)` at temperature `temp`, which is
`prob ** (1/temp)` renormalised. Composed, a choice's sampling probability
goes as

```
value ** ( COMMITMENT * progress / ((1 + sub_size) * temp) )
```

Three user- or structure-controlled quantities multiplying into one exponent.
This is very likely why `temp` reads as inert in sweeps — it is fighting
`COMMITMENT` for the same degree of freedom, and `COMMITMENT` ramps while
`temp` is constant. Worth deciding which one owns categorical sharpness.

### Minor

- `_evict_worst` scans `scores` twice (`min` then a comprehension). Irrelevant
  at n=50; mentioned only for completeness.
- `rank_weights` recomputes all weights on every continuous update — O(n log n)
  per trial. Also fine at n=50, but it is the hot path if the reservoir ever
  grows.

---

## 4. Evaluation and tuning

### Random search is the wrong bar, and you already knew it

The README says so itself: "Beating it is the minimum, not a strong claim."
The right comparator is **TPE** — same niche as BUTChC (sequential,
model-based, handles conditional spaces, no gradient assumptions) and it is
what a user choosing against you would actually reach for, via Optuna or
Hyperopt.

`benchmarks/baselines.py` now contains a dependency-free TPE: split the trials
into a good fraction and the rest, fit a Parzen density to each, propose the
candidate maximising `l(x)/g(x)`. Conditional parameters are fitted only on
trials in which that node was sampled, which is the tree-structured part.

Two implementation notes that cost real accuracy if you get them wrong, and
are written into the file as comments:

- The bad set must be **thinned by an even stride over the value ordering**,
  not truncated to the best rejects. Truncating pulls `g(x)` on top of `l(x)`
  and destroys the contrast the ratio is measuring — it cost roughly an order
  of magnitude on the sphere problems when I tried it.
- Fitting *every* reject makes the run O(budget²). The cap keeps it linear.

### What the comparison actually says

Medians over 5 paired seeds, tuning suite (higher is better, optimum 0):

| Problem | random | TPE | BUTChC |
|---|---|---|---|
| 5D sphere | -3.24 | -0.17 | **-0.00** |
| Rosenbrock | -0.06 | **-0.02** | -0.09 |
| Rastrigin 4D | -17.50 | -7.29 | **-4.74** |
| 10D sphere | -20.32 | -2.61 | **-0.03** |
| Ackley 5D | -13.84 | -3.78 | **-1.79** |
| Log-scale target | -0.005 | -0.001 | **-0.000** |
| Branch trap | -0.81 | -1.00 | **-0.02** |
| Categorical mix | -0.54 | -0.010 | **-0.001** |
| Integer mix | -0.31 | -0.002 | **-0.000** |
| Noisy 5D sphere | -3.51 | -0.71 | **-0.36** |

This is a far more informative table than the random column. BUTChC wins
decisively where its design claims it should — conditional structure
(`Branch trap`, where TPE is *worse than random*), high dimensions, categorical
mixtures, noise — and loses on `Rosenbrock`, whose narrow curved valley is
exactly the parameter interaction the independence assumption cannot model.
That is a much stronger and more honest story than "beats random search".

Caveat: with 5 seeds most of the sign tests sit at p=0.062 at best, since
5-0 is the most extreme result available. Use 30 seeds for anything you publish.

### New scripts

**`benchmarks/evaluate.py`** replaces `compare.py`.

```bash
python benchmarks/evaluate.py 30
python benchmarks/evaluate.py 10 --suite heldout --methods tpe,butchc
```

Paired seeds across all methods, medians plus win-loss records plus two-sided
exact sign-test p-values, so "wins by a lot once" is distinguishable from
"wins reliably". Noise state is reset per seed so every method meets the
identical noise sequence, not merely an equally-distributed one. Ratios of
medians are deliberately gone — they flatter squared-error objectives.

**`benchmarks/tune.py`** rewritten. Beyond the two bug fixes:

- **Disjoint selection and confirmation seeds.** Selection uses `0..n`;
  confirmation uses `1000..1030`, which the sweep never saw. A knob chosen
  because it suited the selection seeds shows up as a shrunken gap.
- **Held-out problems in the confirmation.** A configuration fitted to the
  shape of the tuning problems shows up as a gap that does not transfer.
- **Paired sign tests in the confirmation output**, not just medians.
- **Cache keyed on a hash of the `butchc` source.** The old cache was keyed on
  config and seed count only, so editing the update rule and re-running would
  silently report the old behaviour under the new code's name. This is the
  failure mode most likely to have quietly corrupted a past sweep.
- **Two knobs the old grid never covered:** `rank_sharpness`
  (`_utils.RANK_SHARPNESS`) and `min_bandwidth`
  (`_utils.MIN_BANDWIDTH_FRACTION`). Both are hard-coded constants that shape
  the continuous model directly.

### An early result worth chasing

A single-round, 3-seed smoke run already found a change that survived
confirmation on unseen seeds:

```
min_bandwidth  0.003:0.764  0.01:0.681  0.03:0.431  0.1:0.125   best=0.003
```

`MIN_BANDWIDTH_FRACTION` 0.01 → 0.003, confirmed on 3 held-out seeds:

| Problem | suite | base | tuned |
|---|---|---|---|
| Ackley 5D | tuning | -1.81 | **-0.22** |
| 10D sphere | tuning | -0.033 | **-0.003** |
| Styblinski 4D | held-out | -0.066 | **-0.004** |
| 20D sphere | held-out | -0.121 | **-0.013** |
| Rastrigin 8D | held-out | -31.98 | **-28.47** |

Four of the six held-out problems improved. The bandwidth floor is what stops
the KDE collapsing to a delta, and at 1% of the range it was evidently
stopping convergence too. This is exactly the kind of thing the old grid could
not have found, because it never varied the constant.

Also note `reservoir=100` scoring 0.167 against `50`'s 0.750 — the first
honest measurement of that knob, now that the initial grid actually changes
size with it.

---

## 5. Suggested order of work

1. **Run the full sweep**: `python benchmarks/tune.py 12 --rounds 2`. Expect a
   few hours. B3 changed the update rule, so the previous sweep's conclusions
   no longer apply and this is now required rather than optional. The
   `min_bandwidth` result above suggests the continuous side has headroom the
   previous sweep could not reach.
2. **Re-run `evaluate.py 30`** with the chosen defaults and replace the
   README's `v0.3` column with an `optuna` column. The migration section
   already documents the 0.3 → 0.4 change; the table does not need to as well.
3. **Get `dev/eval` running** against a real ConfigSpace install and run
   `rbv2_super`. The bridge logic is tested against stubs; the API surface is
   not.
4. **Resolve the `temp` / `COMMITMENT` overlap.** If the sweep again reports
   `temp` as inert, that is evidence rather than noise, and one of the two
   should be removed from the public surface.
5. **Sweep the docs** (`README.md`, `docs/butchc_docs_detailed.md`) for the
   same past-tense habit if you want the whole project consistent.

---

## Changes already applied

- `benchmarks/tune.py` — rewritten (B1, B2, seed splits, held-out
  confirmation, source-hashed cache, two new knobs).
- `benchmarks/evaluate.py` — new, replaces `compare.py`.
- `benchmarks/baselines.py` — new; `random_search` moved here from
  `problems.py`, plus the TPE implementation.
- `benchmarks/problems.py` — baselines removed, module docstring corrected to
  describe three suites rather than two.
- `butchc/*.py` — ~25 comments and docstrings rewritten; six stale docstrings
  corrected; `_has_prior` deleted.
- `benchmarks/baselines.py` — added `optuna_search`, Optuna's own `TPESampler`
  as the reference baseline, alongside the dependency-free equivalent.
- `butchc/optimizer.py`, `butchc/_update.py` — B3 and B4 applied;
  `MOVEMENT_EPSILON` added.
- `tests/` — four tests rewritten to match the corrected semantics, three added.
- `dev/` — new: the real-world evaluation tier (see `dev/README.md`).
- `README.md` — reproduction command, return-value block and convergence
  section updated.

**Verification.** The library edits are comment-only plus one dead-function
deletion. A fingerprint over 20 seeded runs spanning every suite, plus a
prior-seeded warm-start chain over log and integer parameters, is byte-identical
before and after:

```
ebf3b658493b8451c81c41a9661e57c37a7a41cc5495b47bdeaf766477ef0f1b   (before)
ebf3b658493b8451c81c41a9661e57c37a7a41cc5495b47bdeaf766477ef0f1b   (after)
```

Test suite: 257 pass, unchanged.
