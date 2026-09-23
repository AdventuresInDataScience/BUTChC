# Codemap

What every file in this repository does, and — for every function, method and
class in the Python source — where it lives, what it does, what it calls, and
what calls it.

For *why* the library works the way it does, see [design.md](design.md). For
the parameter-by-parameter API, see [api.md](api.md). This document is a
map of the code, not an argument for it.

---

## How the tables are generated

The tables below come from static analysis of the source with Python's `ast`
module. Call resolution is import-aware — it follows `from x import y`, module
aliases like `import butchc._tree as tree_mod`, `self.method()`, and package
re-exports through `__init__.py` — rather than matching bare names globally,
since several files reuse generic names like `objective`, `run` and `main`.

Four things it cannot see:

- **Module-level calls.** A call made directly in a module's top-level code —
  `benchmarks/problems.py` builds its `TUNING`/`HELDOUT`/`NOISY` dicts by
  calling its problem constructors at import time, outside any function — is
  invisible to this analysis. Those functions correctly show no callers below
  even though they are used; the wiring happens at import time via the suite
  dicts, not via calls inside a function body.
- **Dispatch through data.** A function referenced only as a value — stored
  in a dict, passed as a callback — doesn't produce a `Call` node at the
  point of reference, so it won't show up as "called by" there.
- **Nested closures.** A function defined *inside* another function (e.g.
  `evaluate.py`'s `with_recorder` returns one) is invisible as its own
  entity; only module-level and class-level functions are tracked.
- **Tests.** `tests/` is deliberately excluded from the tables. Its functions
  are pytest entry points, not part of the library's internal call graph;
  including them would mostly add "every function is called by some test",
  which doesn't help build a mental model of how the library is wired
  together. `tests/` gets a one-line-per-file summary instead, below.

This goes stale the moment the source changes shape. Regenerate the tables
with:

```bash
python dev/tools/codemap_gen.py > /tmp/codemap_tables.md
```

and splice the output back into the "Detailed reference" sections below. The
directory map and the tests summary are maintained by hand and aren't
touched by that script.

---

## Repository layout

```
BUTChC/
├── butchc/              the shipped library — the only thing in the wheel
│   ├── __init__.py       public API surface
│   ├── optimizer.py      the optimisation loop (BUTChC_optimize)
│   ├── _tree.py          builds the probability tree; internal/external coordinate transforms
│   ├── _sampling.py      draws one configuration by walking the tree
│   ├── _update.py        the learning rule — categorical mean-rank scoring, continuous elite archive
│   ├── _prune.py         reduces a finished run's search space (dropped branches, narrowed bounds)
│   ├── _utils.py         shared numeric primitives (softmax, bandwidth, rank weights, reflection)
│   ├── _validate.py      up-front validation of search spaces, hyperparameters, warm-start trees
│   ├── _parallel.py      batch evaluation through a user-supplied executor
│   └── interop/          optional conversion to/from other formats; not imported by butchc itself
│       ├── __init__.py    re-exports the conversion functions
│       ├── _configspace.py   ConfigSpace <-> BUTChC search space
│       └── _json.py          JSON <-> BUTChC search space
├── benchmarks/           dev tooling, zero external deps, safe to run anywhere
│   ├── problems.py        synthetic objective functions + the TUNING/HELDOUT/NOISY suites
│   ├── baselines.py       random search, TPE (+ a "flat" variant with structure hidden), Optuna wrapper
│   ├── evaluate.py        runs BUTChC + baselines across seeds/problems; final-best or --anytime reporting
│   ├── tune.py            selects BUTChC's shipped defaults; --regime tunes for one problem shape
│   ├── batch_cost.py      measures the accuracy cost of batch > 1 at fixed budget
│   └── overhead.py        optimiser wall-clock per trial, objective stubbed out
├── dev/                  development-only; nothing here ships or is importable from butchc
│   ├── README.md          explains this split and why benchmarks/ stays at the top level
│   ├── eval/              real-world evaluation tier — needs optuna, yahpo-gym, ConfigSpace
│   │   ├── README.md          stub vs. real ConfigSpace testing, and what still isn't covered
│   │   ├── fingerprint.py     hashes seeded runs, to prove a refactor is behaviour-neutral
│   │   ├── yahpo_suite.py     wraps YAHPO Gym's rbv2_super scenario as a BUTChC problem
│   │   ├── run_real.py        runs the real-world suite through evaluate.py's harness
│   │   ├── check_parallel.py  exercises batch/executor against real pools, times the speed-up
│   │   └── requirements.txt   the extra dependencies this tier needs
│   ├── archive/           retired process documents — past reviews, planning notes
│   │   └── REVIEW.md          the v0.4.0 code review (see the note at its top)
│   └── tools/             one-off scripts that help maintain the repo itself
│       ├── codemap_gen.py     regenerates this document's tables
│       ├── publish_tables.py  regenerates the published benchmark tables into results/
│       └── results/           recorded benchmark output, so published numbers have provenance
├── tests/                pytest suite
│   ├── test_optimizer.py      integration tests, incl. a 0.4.0 regression class
│   ├── test_tree.py           tree construction, coordinate transforms, prior seeding
│   ├── test_update.py         the update rule
│   ├── test_sampling.py       sampling, reflection, exploration floor
│   ├── test_utils.py          the shared numeric primitives
│   ├── test_batch.py          batch/executor semantics
│   ├── test_interop.py        ConfigSpace + JSON conversion, incl. tests against a real ConfigSpace install
│   ├── test_constants.py      every documented module constant actually reaches the search
│   └── test_prune.py          `prune`'s refusal-to-remove guarantees, narrowing, reports
├── docs/
│   ├── api.md             parameter-by-parameter reference, search-space format, return value
│   ├── design.md          why it's built this way — the loop, key decisions, limitations, open questions
│   ├── examples.md        task-oriented recipes
│   └── codemap.md         this document
├── README.md              quick start, search-space format, how it works, tuning guide, benchmarks
├── CHANGELOG.md           living version history, one section per release
├── LICENSE                MIT
└── pyproject.toml         packaging config — wheel includes only butchc*, so nothing else ships
```

### Tests, briefly

Full call-graph detail isn't included for `tests/` (see "How this was built"
above); here's what each file covers instead:

| File | Covers |
|---|---|
| `test_optimizer.py` | Integration tests: optimisation quality vs. random search, return-value shape, best-tracking, loss properties, warm start, hierarchical spaces, priors, and `TestRegressions040` — a dedicated regression class for the 0.4.0 bugs. |
| `test_tree.py` | Tree construction, internal/external coordinate transforms, prior seeding. |
| `test_update.py` | The update rule: categorical scoring, continuous archive eviction and reweighting. |
| `test_sampling.py` | Sampling, reflection, the exploration floor. |
| `test_utils.py` | The shared numeric primitives. |
| `test_batch.py` | Batch/executor semantics: sample-order updates, the truncated final batch, `batch=-1`. |
| `test_interop.py` | ConfigSpace and JSON conversion, including the refusal cases (conjunctions, multiple parents, forbidden clauses), plus `TestAgainstRealConfigSpace` — seven tests against a real `ConfigurationSpace` install, skipped when it's absent. |
| `test_constants.py` | New in 0.5.1. Every constant `docs/api.md` documents as patchable is asserted to actually move the search — the failure mode this guards against (a half-bound constant that silently does nothing) has occurred twice. |
| `test_prune.py` | New in 0.6.0. `prune`'s one-directional safety guarantees (never emptying a node, protecting the best branch), continuous narrowing, input immutability, and `prune_report`'s output. |

---

## Detailed reference: `butchc/` — the shipped library

Everything below is importable as `butchc.*` or used internally by it. This
is the part of the codebase most worth a new contributor's attention first.

### `butchc/__init__.py`

BUTChC: Bayesian Update Tree Chained Conditionally

| Name | Line | Kind | Purpose | Calls | Called by |
|---|---|---|---|---|---|

### `butchc/_parallel.py`

Evaluating a batch of configurations, optionally in parallel.

| Name | Line | Kind | Purpose | Calls | Called by |
|---|---|---|---|---|---|
| `resolve_batch` | 26 | function | Turn the user-facing ``batch`` argument into a concrete size. | — | `butchc.optimizer.BUTChC_optimize` |
| `evaluate_batch` | 38 | function | Evaluate ``configs``, returning results in the same order. | `_explain` | `butchc.optimizer.BUTChC_optimize` |
| `_explain` | 65 | function | Attach the diagnosis that a raw pickling error does not carry. | — | `evaluate_batch` |

### `butchc/_prune.py`

Reduce a search space using what a finished run learned about it.

| Name | Line | Kind | Purpose | Calls | Called by |
|---|---|---|---|---|---|
| `prune` | 66 | function | Return a smaller search space, keeping the branches worth searching. | `_prune_level`, `butchc._validate.SearchSpaceError`, `butchc._validate.check_tree_matches_searchspace`, `butchc._validate.validate_searchspace` | — |
| `_prune_level` | 144 | function |  | `_copy_spec`, `_narrow_continuous`, `_prune_categorical`, `butchc._tree.initialize_prob_tree` | `_prune_categorical`, `prune` |
| `_prune_categorical` | 165 | function |  | `_copy_spec`, `_prune_level`, `_pruned_categorical_node`, `butchc._validate.SearchSpaceError` | `_prune_level` |
| `_pruned_categorical_node` | 212 | function | Build a categorical tree node restricted to `kept`, renormalised. | `butchc._tree._subspace_size` | `_prune_categorical` |
| `_narrow_continuous` | 252 | function |  | `_copied_continuous_node`, `_copy_spec`, `_narrowed_continuous_node`, `butchc._tree.reservoir_summary` | `_prune_level` |
| `_copied_continuous_node` | 283 | function | Deep-copy a continuous tree node's mutable fields. | — | `_narrow_continuous` |
| `_narrowed_continuous_node` | 297 | function | Build a continuous tree node matching a narrowed `[lo_ext, hi_ext]`. | `butchc._tree.to_internal` | `_narrow_continuous` |
| `_copy_spec` | 337 | function | Deep-copy a spec node without importing copy for the nested case. | *(itself, recursively)* | `_narrow_continuous`, `_prune_categorical`, `_prune_level` |
| `prune_report` | 353 | function | Describe what `prune` removed, for printing before you trust it. | `_report_level` | — |
| `_report_level` | 364 | function |  | *(itself, recursively)* | `prune_report` |

### `butchc/_sampling.py`

Drawing configurations from the probability tree.

| Name | Line | Kind | Purpose | Calls | Called by |
|---|---|---|---|---|---|
| `sample_node` | 22 | function | Draw a single value from a probability-tree node. | `butchc._tree.snap_internal`, `butchc._tree.to_external`, `butchc._utils.reflect`, `butchc._utils.silverman_bandwidth`, `butchc._utils.softmax` | `traverse_sample` |
| `traverse_sample` | 78 | function | Recursively sample a full configuration from the probability tree. | `sample_node`, *(itself, recursively)* | `butchc.optimizer.BUTChC_optimize` |

### `butchc/_tree.py`

Probability-tree construction and coordinate transforms.

| Name | Line | Kind | Purpose | Calls | Called by |
|---|---|---|---|---|---|
| `to_internal` | 26 | function | Map an external parameter value into the node's internal space. | — | `butchc._prune._narrowed_continuous_node`, `_seed_continuous_prior`, `initialize_prob_tree` |
| `to_external` | 33 | function | Map an internal coordinate back to the user-facing value. | — | `butchc._sampling.sample_node`, `reservoir_summary` |
| `snap_internal` | 49 | function | Internal coordinate of the value the objective will actually receive. | — | `butchc._sampling.sample_node`, `_seed_continuous_prior` |
| `initialize_prob_tree` | 62 | function | Recursively build a probability tree from a validated search space. | `_categorical_node`, `_seed_continuous_prior`, `to_internal`, `butchc._utils.rank_weights` | `butchc._prune._prune_level`, `_categorical_node`, `butchc.optimizer.BUTChC_optimize` |
| `reservoir_summary` | 123 | function | Human-readable summary of a continuous node. | `to_external` | `butchc._prune._narrow_continuous` |
| `_categorical_node` | 154 | function | Build a categorical node, storing any prior alongside the counts. | `_subspace_size`, `initialize_prob_tree` | `initialize_prob_tree` |
| `_subspace_size` | 202 | function | Parameters reachable under a choice, counting the deepest path. | *(itself, recursively)* | `butchc._prune._pruned_categorical_node`, `_categorical_node` |
| `_seed_continuous_prior` | 229 | function | Replace part of the uniform grid with points drawn from the user's prior. | `_inverse_normal_cdf`, `_thin`, `snap_internal`, `to_internal` | `initialize_prob_tree` |
| `_thin` | 282 | function | Drop ``n_drop`` points from ``grid``, spreading the loss evenly. | — | `_seed_continuous_prior` |
| `_inverse_normal_cdf` | 295 | function | Standard-normal quantile via Acklam's rational approximation. | — | `_seed_continuous_prior` |

### `butchc/_update.py`

The update rule — where the objective value enters the model.

| Name | Line | Kind | Purpose | Calls | Called by |
|---|---|---|---|---|---|
| `quality_weight` | 35 | function | Convert a trial's quantile rank into an update weight in ``[0, 1]``. | — | `butchc.optimizer.BUTChC_optimize` |
| `update_tree` | 56 | function | Update the probability tree in-place from one evaluated trial. | `_update_categorical`, `_update_continuous`, *(itself, recursively)* | `butchc.optimizer.BUTChC_optimize` |
| `_update_categorical` | 152 | function | Record one visit to ``value`` and the rank it earned. | `_recompute_prob` | `update_tree` |
| `_recompute_prob` | 187 | function | Recompute a categorical node's probabilities from its counts and visits. | `_apply_prior` | `_update_categorical` |
| `_apply_prior` | 230 | function | Bias the values by the user's stated prior, fading as evidence arrives. | — | `_recompute_prob` |
| `_update_continuous` | 259 | function |  | `_evict_worst`, `butchc._utils.rank_weights`, `butchc._utils.weighted_mean` | `update_tree` |
| `_evict_worst` | 276 | function | Remove the worst-scoring archive entry, breaking ties uniformly. | — | `_update_continuous` |

### `butchc/_utils.py`

Numeric helpers shared across the package.

| Name | Line | Kind | Purpose | Calls | Called by |
|---|---|---|---|---|---|
| `softmax` | 36 | function | Temperature-scaled softmax over log-space scores. | — | `butchc._sampling.sample_node` |
| `weighted_mean` | 63 | function | Weighted mean of ``values``. Assumes ``weights`` sums to 1. | — | `butchc._update._update_continuous`, `silverman_bandwidth` |
| `effective_sample_size` | 68 | function | Kish effective sample size, ``1 / sum(w^2)``, for normalised weights. | — | `silverman_bandwidth` |
| `silverman_bandwidth` | 79 | function | Silverman's rule-of-thumb KDE bandwidth from a weighted reservoir. | `effective_sample_size`, `weighted_mean` | `butchc._sampling.sample_node` |
| `reflect` | 113 | function | Fold ``x`` back into ``[lo, hi]`` by reflection. | — | `butchc._sampling.sample_node` |
| `compute_trial_loss` | 132 | function | Aggregate per-node deltas from one trial into a scalar. | — | `butchc.optimizer.BUTChC_optimize` |
| `is_finite_number` | 148 | function | True if ``value`` is a real number that is neither NaN nor infinite. | — | `butchc.optimizer.BUTChC_optimize` |
| `rank_weights` | 173 | function | Geometric weights over within-archive score rank. | — | `butchc._tree.initialize_prob_tree`, `butchc._update._update_continuous` |

### `butchc/_validate.py`

Up-front validation of search spaces, hyperparameters and warm-start trees.

| Name | Line | Kind | Purpose | Calls | Called by |
|---|---|---|---|---|---|
| `SearchSpaceError` | 14 | class | Raised when a search-space definition is malformed. | — | `butchc._prune._prune_categorical`, `butchc._prune.prune`, `_validate_categorical`, `_validate_continuous`, `_validate_continuous_prior`, `_validate_prior_strength`, `validate_searchspace` |
| `_path` | 18 | function |  | — | `check_tree_matches_searchspace`, `validate_prob_tree`, `validate_searchspace` |
| `integer_span` | 22 | function | Count of integers reachable in ``[lo, hi]`` after rounding to nearest. | — | `_validate_continuous` |
| `validate_searchspace` | 27 | function | Recursively validate a search-space definition. | `SearchSpaceError`, `_path`, `_validate_categorical`, `_validate_continuous` | `butchc._prune.prune`, `_validate_categorical`, `butchc.optimizer.BUTChC_optimize` |
| `_validate_categorical` | 99 | function |  | `SearchSpaceError`, `_validate_prior_strength`, `validate_searchspace` | `validate_searchspace` |
| `_validate_continuous` | 170 | function |  | `SearchSpaceError`, `_validate_continuous_prior`, `_validate_prior_strength`, `integer_span` | `validate_searchspace` |
| `_validate_prior_strength` | 203 | function |  | `SearchSpaceError` | `_validate_categorical`, `_validate_continuous` |
| `_validate_continuous_prior` | 221 | function |  | `SearchSpaceError` | `_validate_continuous` |
| `_is_integer` | 274 | function | True for ints and integer-like scalars such as ``numpy.int64``. | — | `validate_hyperparameters` |
| `validate_hyperparameters` | 287 | function | Validate the optimiser's scalar arguments, failing before trial 1. | `_is_integer` | `butchc.optimizer.BUTChC_optimize` |
| `validate_prob_tree` | 312 | function | Structurally validate a warm-start probability tree. | `_path`, *(itself, recursively)* | `butchc.optimizer.BUTChC_optimize` |
| `check_tree_matches_searchspace` | 370 | function | Verify a warm-start tree describes the same parameters as ``searchspace``. | `_check_continuous_matches`, `_path`, *(itself, recursively)* | `butchc._prune.prune`, `butchc.optimizer.BUTChC_optimize` |
| `_check_continuous_matches` | 416 | function | Verify a warm-started continuous node describes the same interval. | — | `check_tree_matches_searchspace` |

### `butchc/optimizer.py`

The BUTChC optimisation loop.

| Name | Line | Kind | Purpose | Calls | Called by |
|---|---|---|---|---|---|
| `BUTChC_optimize` | 38 | function | Run the BUTChC optimisation loop (maximisation). | `butchc._parallel.evaluate_batch`, `butchc._parallel.resolve_batch`, `butchc._sampling.traverse_sample`, `butchc._tree.initialize_prob_tree`, `butchc._update.quality_weight`, `butchc._update.update_tree`, `butchc._utils.compute_trial_loss`, `butchc._utils.is_finite_number`, `butchc._validate.check_tree_matches_searchspace`, `butchc._validate.validate_hyperparameters`, `butchc._validate.validate_prob_tree`, `butchc._validate.validate_searchspace`, `_as_float`, `_coerce_searchspace`, `_quantile_rank` | `benchmarks.batch_cost.run`, `benchmarks.evaluate.butchc_search`, `benchmarks.tune.run_one`, `dev.eval.check_parallel.run`, `dev.eval.fingerprint.main` |
| `_coerce_searchspace` | 299 | function | Accept a ConfigSpace ``ConfigurationSpace`` wherever a dict is expected. | — | `BUTChC_optimize` |
| `_as_float` | 340 | function | Coerce an objective's return to a float, or explain why it cannot be. | — | `BUTChC_optimize` |
| `_quantile_rank` | 363 | function | Midrank of ``value`` among previously observed objectives. | — | `BUTChC_optimize` |

---

## Detailed reference: `butchc/interop/` — optional format conversion

Not imported by `butchc/__init__.py`, so nothing here makes ConfigSpace a
dependency of the core package. Only reached if the caller passes a
`ConfigurationSpace` to `BUTChC_optimize` or imports `butchc.interop`
directly.

### `butchc/interop/__init__.py`

Conversion between BUTChC search spaces and other libraries' formats.

| Name | Line | Kind | Purpose | Calls | Called by |
|---|---|---|---|---|---|

### `butchc/interop/_configspace.py`

Conversion between ``ConfigSpace.ConfigurationSpace`` and BUTChC dicts.

| Name | Line | Kind | Purpose | Calls | Called by |
|---|---|---|---|---|---|
| `UnsupportedSpace` | 45 | class | Raised when a ConfigSpace object has no faithful tree representation. | — | `_assemble`, `_convert_one`, `_index_conditions`, `from_configspace` |
| `_hyperparameters` | 61 | function |  | — | `from_configspace` |
| `_conditions` | 67 | function |  | — | `_index_conditions` |
| `_forbiddens` | 73 | function |  | — | `from_configspace` |
| `_class_name` | 81 | function |  | — | `_convert_one`, `_index_conditions` |
| `from_configspace` | 87 | function | Convert a ``ConfigurationSpace`` to a BUTChC search space. | `UnsupportedSpace`, `_assemble`, `_convert_one`, `_forbiddens`, `_hyperparameters`, `_index_conditions` | `dev.eval.yahpo_suite.make_problem` |
| `_convert_one` | 135 | function | One hyperparameter to a spec dict, or None if it is a constant. | `UnsupportedSpace`, `_class_name` | `from_configspace` |
| `_round_to_int` | 165 | function | Module-level so a wrapped objective stays picklable for process pools. | — | — |
| `_index_conditions` | 170 | function | Map each conditional parameter to ``(parent_name, [values])``. | `UnsupportedSpace`, `_class_name`, `_conditions` | `from_configspace` |
| `_assemble` | 222 | function | Nest each conditional parameter under its parent's chosen branches. | `UnsupportedSpace` | `from_configspace` |
| `_WrappedObjective` | 250 | class | Restores constants and applies casts before calling the real objective. | — | `wrap_objective` |
| `_WrappedObjective.__init__` | 257 | method |  | — | — |
| `_WrappedObjective.__call__` | 262 | method |  | — | — |
| `wrap_objective` | 270 | function | Wrap an objective so it receives complete, correctly typed configs. | `_WrappedObjective` | `dev.eval.yahpo_suite.make_problem` |
| `to_configspace` | 285 | function | Convert a BUTChC search space to a ``ConfigurationSpace``. | `_Emitter`, `_Emitter.emit`, `_add` | — |
| `_add` | 334 | function | Add to a ConfigurationSpace across the 0.x and 1.x APIs. | — | `to_configspace` |
| `_Emitter` | 344 | class | Walks a BUTChC tree, collecting hyperparameters and their conditions. | — | `to_configspace` |
| `_Emitter.__init__` | 352 | method |  | — | — |
| `_Emitter.emit` | 360 | method |  | `_Emitter._unique`, `_condition`, `_to_hyperparameter`, *(itself, recursively)* | `to_configspace` |
| `_Emitter._unique` | 374 | method | Qualify only on collision, so unambiguous names stay readable. | — | `_Emitter.emit` |
| `_condition` | 388 | function |  | — | `_Emitter.emit` |
| `_to_hyperparameter` | 394 | function |  | — | `_Emitter.emit` |

### `butchc/interop/_json.py`

Serialising a search space to JSON and back.

| Name | Line | Kind | Purpose | Calls | Called by |
|---|---|---|---|---|---|
| `to_json` | 18 | function | Serialise a search space to a JSON string. | — | — |
| `from_json` | 36 | function | Parse a search space from a JSON string, restoring ``next_level`` keys. | `_restore` | — |
| `_restore` | 48 | function |  | *(itself, recursively)* | `from_json` |

---

## Detailed reference: `benchmarks/` — synthetic benchmark harness

Dev tooling with zero external dependencies (per `pyproject.toml`, none of
this ships in the wheel — only `butchc*` does). This is the regression loop
run on every change; see `dev/README.md` for why it lives at the top level
rather than under `dev/`.

### `benchmarks/baselines.py`

Baseline optimisers to measure BUTChC against.

| Name | Line | Kind | Purpose | Calls | Called by |
|---|---|---|---|---|---|
| `random_config` | 73 | function | Draw one configuration uniformly, respecting log scales and nesting. | *(itself, recursively)* | `random_search` |
| `random_search` | 94 | function | Uniform random search on a fixed budget. | `random_config` | — |
| `tpe_search` | 124 | function | Tree-structured Parzen estimator on a fixed budget. | `_suggest` | — |
| `_suggest` | 151 | function | Propose one configuration, descending into whichever branch is chosen. | `_suggest_categorical`, `_suggest_continuous`, *(itself, recursively)* | `flat_tpe_search`, `tpe_search` |
| `_split` | 174 | function | Partition observations into the good fraction and the rest. | — | `_suggest_categorical`, `_suggest_continuous` |
| `_suggest_categorical` | 186 | function | Pick the choice maximising ``l/g``, or draw uniformly during startup. | `_counts`, `_split` | `_suggest` |
| `_counts` | 202 | function | Laplace-smoothed frequency of each choice, normalised to sum to 1. | — | `_suggest_categorical` |
| `_suggest_continuous` | 212 | function | Draw from the good density and keep the candidate maximising ``l/g``. | `_log_density`, `_parzen`, `_split` | `_suggest` |
| `_parzen` | 253 | function | Place a Gaussian on each observation, plus one flat prior component. | — | `_suggest_continuous` |
| `flatten_space` | 281 | function | Collapse a conditional space into one flat space of qualified names. | *(itself, recursively)* | `flat_tpe_search` |
| `unflatten_config` | 313 | function | Keep only the parameters the chosen branches actually use. | *(itself, recursively)* | `flat_tpe_search` |
| `flat_tpe_search` | 336 | function | TPE with the conditional structure taken away. | `_suggest`, `flatten_space`, `unflatten_config` | — |
| `optuna_search` | 371 | function | Optuna's ``TPESampler`` on the same budget, as a reference baseline. | — | — |
| `_log_density` | 432 | function | Log density of an equally-weighted mixture, truncated to ``[lo, hi]``. | — | `_suggest_continuous` |

### `benchmarks/batch_cost.py`

What does batching cost in sample efficiency?

| Name | Line | Kind | Purpose | Calls | Called by |
|---|---|---|---|---|---|
| `run` | 37 | function | Best objective found, scored noise-free where the problem asks for it. | `_unpack`, `butchc.optimizer.BUTChC_optimize` | `main` |
| `_unpack` | 49 | function | Problems are (space, objective, budget) or (space, objective, budget, report). | — | `run` |
| `_reset_noise` | 56 | function | Give every batch size the identical noise sequence, not merely an equally distributed one. | — | `main` |
| `main` | 69 | function |  | `_reset_noise`, `run` | — |

### `benchmarks/overhead.py`

What does the optimiser itself cost, separately from the objective?

| Name | Line | Kind | Purpose | Calls | Called by |
|---|---|---|---|---|---|
| `constant` | 32 | function | A free objective, so the measurement is all optimiser. | — | — |
| `continuous_space` | 37 | function |  | — | `main` |
| `time_one` | 41 | function |  | — | `main` |
| `butchc_runner` | 47 | function |  | `butchc.optimizer.BUTChC_optimize` | — |
| `tpe_runner` | 51 | function |  | `benchmarks.baselines.tpe_search` | — |
| `optuna_runner` | 55 | function |  | `benchmarks.baselines.optuna_search` | — |
| `main` | 59 | function |  | `continuous_space`, `time_one` | — |

### `benchmarks/evaluate.py`

Measure BUTChC against random search and TPE on matched budgets and seeds.

| Name | Line | Kind | Purpose | Calls | Called by |
|---|---|---|---|---|---|
| `butchc_search` | 39 | function | Adapter giving BUTChC the same call signature as the baselines. | `butchc.optimizer.BUTChC_optimize` | — |
| `sign_test` | 61 | function | Two-sided exact sign test on paired samples. | — | `report_table`, `benchmarks.tune.confirm` |
| `run_problem` | 82 | function | Run one method on one problem across every seed. | — | `collect` |
| `collect` | 98 | function | Return ``{problem: {method: [per-seed result]}}`` for one suite. | `run_problem` | `main`, `dev.eval.run_real.main` |
| `report_table` | 115 | function | Print medians, then the reference method's record against each baseline. | `sign_test` | `main`, `dev.eval.run_real.main` |
| `with_recorder` | 158 | function | Wrap an objective so every value it returns is kept, in call order. | — | `collect_curves` |
| `best_so_far` | 175 | function | Running maximum — the curve a user actually experiences. | — | `collect_curves` |
| `trials_to_target` | 184 | function | Trials until the curve first reaches ``target``, or None if it never does. | — | `report_anytime` |
| `collect_curves` | 197 | function | Return ``{problem: {method: [curve per seed]}}``. | `best_so_far`, `with_recorder` | `main` |
| `report_anytime` | 212 | function | Median trials to reach ``fraction`` of the achievable range. | `trials_to_target` | `main` |
| `main` | 258 | function |  | `collect`, `collect_curves`, `report_anytime`, `report_table` | — |

### `benchmarks/problems.py`

Benchmark problems.

| Name | Line | Kind | Purpose | Calls | Called by |
|---|---|---|---|---|---|
| `_sphere` | 19 | function |  | — | `_noisy_sphere` |
| `_quad2d` | 29 | function |  | — | — |
| `_rosenbrock` | 34 | function |  | — | — |
| `_rastrigin` | 39 | function |  | — | — |
| `_ackley` | 47 | function |  | — | — |
| `_log_target` | 58 | function |  | — | — |
| `_branch_trap` | 63 | function | The branch whose optimum is best has the worst average. | — | — |
| `_categorical_mix` | 94 | function |  | — | — |
| `_integer_mix` | 113 | function |  | — | — |
| `_plateau` | 128 | function | A discrete objective with heavy ties, like accuracy on a small set. | — | — |
| `_noisy_sphere` | 139 | function |  | `_sphere` | — |
| `_griewank` | 173 | function |  | — | — |
| `_styblinski` | 188 | function |  | — | — |
| `_nested_pipeline` | 200 | function | Three levels of conditioning, the structure the library exists for. | — | — |
| `_mixed_conditional_wide` | 239 | function |  | — | — |

*Every function above with an empty "called by" is in fact used — each is
called once, at import time, while building the `TUNING` / `HELDOUT` /
`NOISY` dicts at the bottom of this file (e.g. `"Rosenbrock": (*_rosenbrock(),
500)`). That's a module-level call, outside any function body, which is the
one blind spot this analysis has by design — see "How this was built" above.*

### `benchmarks/tune.py`

Select BUTChC's defaults by measurement rather than by taste.

| Name | Line | Kind | Purpose | Calls | Called by |
|---|---|---|---|---|---|
| `source_fingerprint` | 107 | function | Hash every source file in the ``butchc`` package. | — | `main` |
| `apply_config` | 121 | function | Patch the module constants a candidate configuration overrides. | — | `run_one` |
| `run_one` | 140 | function | One seeded run of one problem under one candidate configuration. | `apply_config`, `butchc.optimizer.BUTChC_optimize` | `evaluate` |
| `evaluate` | 154 | function | Per-seed results for one configuration. | `run_one` | `Measurer.__call__` |
| `regime_suites` | 199 | function | Return ``(suites, include_noisy)`` for one regime name. | — | `main` |
| `score_candidates` | 217 | function | Rank candidates within each problem, then average the ranks. | — | `sweep` |
| `Measurer` | 255 | class | Caches problem results by configuration, seeds and library source. | — | `main` |
| `Measurer.__init__` | 258 | method |  | `Measurer._load` | — |
| `Measurer._load` | 262 | method |  | — | `Measurer.__init__` |
| `Measurer._save` | 270 | method |  | — | `Measurer.__call__` |
| `Measurer.__call__` | 277 | method |  | `Measurer._save`, `evaluate` | — |
| `sweep` | 287 | function | Coordinate descent over ``GRID``, starting from ``BASE``. | `score_candidates` | `main` |
| `confirm` | 322 | function | Compare the chosen configuration against ``BASE`` on unseen seeds. | `benchmarks.evaluate.sign_test` | `main` |
| `main` | 353 | function |  | `Measurer`, `confirm`, `regime_suites`, `source_fingerprint`, `sweep` | — |

---

## Detailed reference: `dev/eval/` — real-world evaluation tier

Needs `optuna`, `yahpo-gym` and `ConfigSpace` — run before a release, not on
every commit. `run_real.py` deliberately reimplements none of the statistics;
everything that decides what a result *means* lives in `benchmarks/evaluate.py`
and is imported from there.

### `dev/eval/check_parallel.py`

Exercise ``batch`` and ``executor`` against real pools, and time the speedup.

| Name | Line | Kind | Purpose | Calls | Called by |
|---|---|---|---|---|---|
| `objective` | 38 | function | Deliberately slow, so wall clock is dominated by the objective. | — | — |
| `run` | 45 | function |  | `butchc.optimizer.BUTChC_optimize` | `main` |
| `main` | 54 | function |  | `run` | — |

### `dev/eval/fingerprint.py`

Hash a spread of seeded runs, so a refactor can be proved behaviour-neutral.

| Name | Line | Kind | Purpose | Calls | Called by |
|---|---|---|---|---|---|
| `f_flat` | 61 | function |  | — | — |
| `f_nested` | 66 | function |  | — | — |
| `f_prior` | 74 | function |  | — | — |
| `f_nan` | 79 | function |  | — | — |
| `summarise` | 83 | function |  | — | `main` |
| `main` | 98 | function |  | `butchc.optimizer.BUTChC_optimize`, `summarise` | — |

### `dev/eval/run_real.py`

Run the real-world suite through the same harness as the synthetic one.

| Name | Line | Kind | Purpose | Calls | Called by |
|---|---|---|---|---|---|
| `main` | 32 | function |  | `benchmarks.evaluate.collect`, `benchmarks.evaluate.report_table`, `dev.eval.yahpo_suite.rbv2_super` | — |

### `dev/eval/yahpo_suite.py`

YAHPO Gym problems, exposed in the shape ``benchmarks/evaluate.py`` expects.

| Name | Line | Kind | Purpose | Calls | Called by |
|---|---|---|---|---|---|
| `_benchmark_set` | 61 | function | Construct a YAHPO ``BenchmarkSet`` pinned to one instance. | — | `make_problem` |
| `make_problem` | 73 | function | Build one ``(searchspace, objective, budget)`` triple. | `butchc.interop._configspace.from_configspace`, `butchc.interop._configspace.wrap_objective`, `_benchmark_set` | `rbv2_super` |
| `rbv2_super` | 109 | function | The ``rbv2_super`` suite, keyed by name like ``problems.TUNING``. | `make_problem` | `dev.eval.run_real.main` |

---

## Keeping this current

- **Source changed shape** (new file, renamed function, new import)? Re-run
  `python dev/tools/codemap_gen.py` and replace the four "Detailed reference"
  sections above with its output.
- **Directory added or removed**? Update the tree and the tests table by
  hand — those aren't generated.
- This document is normal shipped documentation (`docs/`), not a process
  record, so unlike `dev/archive/`, it's expected to be kept accurate rather
  than left as a dated snapshot.
