# API reference

Everything the package exposes. For task-shaped guidance see
[examples.md](examples.md); for why the algorithm is built the way it is see
[design.md](design.md).

- [`BUTChC_optimize`](#butchc_optimize)
- [Search space format](#search-space-format)
- [Return value](#return-value)
- [Errors](#errors)
- [`reservoir_summary`](#reservoir_summary)
- [`butchc.interop`](#butchcinterop)
- [`prune` and `prune_report`](#prune-and-prune_report)
- [Tuning by problem shape](#tuning-by-problem-shape)
- [Module constants](#module-constants)

---

## `BUTChC_optimize`

```python
from butchc import BUTChC_optimize

results = BUTChC_optimize(
    searchspace,
    objective,
    budget,
    lambda_         = 2.0,
    alpha           = 3.0,
    temp            = 1.0,
    start_prob_tree = None,
    verbose         = True,
    *,
    gamma           = 0.85,
    explore         = 0.05,
    n_warmup        = None,
    seed            = None,
    batch           = 1,
    executor        = None,
    **kwargs,
)
```

Maximizes `objective` over `searchspace` in `budget` evaluations. Negate to
minimize.

Arguments after `*` are keyword-only. `**kwargs` is forwarded verbatim to every
`objective` call, so any extra data your objective needs goes there.

### Problem definition

| Parameter | Type | Default | Description |
|---|---|---|---|
| `searchspace` | `dict` \| `ConfigurationSpace` | — | See [format](#search-space-format). A `ConfigSpace.ConfigurationSpace` is converted on entry. |
| `objective` | `callable` | — | `(config, **kwargs) → float`. Higher is better. Non-finite returns are recorded but excluded from ranking and from best-tracking. |
| `budget` | `int` | — | Number of objective evaluations. Exact — a `batch` that does not divide it truncates the final batch rather than overrunning. |

### Model shape

| Parameter | Type | Default | Description |
|---|---|---|---|
| `lambda_` | `float > 0` | `2.0` | Sharpness of the continuous archive's rank weighting. Higher concentrates the KDE on the best archived points. Categorical nodes ignore it. **Saturates** — see below. |
| `alpha` | `float > 0` | `3.0` | Smoothing for categorical nodes, in pseudo-visits. Higher keeps probabilities nearer uniform for longer. |
| `temp` | `float > 0` | `1.0` | Softmax temperature for categorical sampling. `>1` explores, `<1` exploits. |
| `gamma` | `float` in `[0, 1)` | `0.85` | Quantile gate on the *continuous* archive. Only trials ranking above this quantile are archived. Categorical nodes score every trial by rank and ignore it. |
| `explore` | `float` in `[0, 1]` | `0.05` | Per-node probability of a uniform draw instead of a modelled one. Non-zero by default, because the archive concentrates sharply enough to need a floor under it. |
| `n_warmup` | `int >= 0` \| `None` | `None` → `0` | Trials drawn uniformly from every node before modelling begins. They reach no continuous archive, but categorical nodes still record their visit and rank — see below. |

`explore` is a floor **per node**, not per configuration: a config in which all
`d` nodes explore has probability `explore ** d`. It bounds how far any single
parameter's marginal can collapse; it does not make BUTChC a superset of random
search in high dimensions.

`lambda_` has a hard ceiling, and it is closer than the table suggests. It acts
only through

```
sharpness = min(RANK_SHARPNESS * lambda_, MAX_SHARPNESS)      # 3.0, 10.0
```

so at the shipped `RANK_SHARPNESS` **any `lambda_` at or above 3.33 is clipped
and behaves identically**. Measured on `Nested pipeline` at 20 seeds,
`lambda_=4.0` and `lambda_=8.0` give the same median to four decimals, as do
`lambda_=1.0` and `rank_sharpness=1.5` — equal products give equal results,
because the product is the only thing either one touches.

Two consequences. Raising `lambda_` past 3.33 to "exploit harder" does nothing;
lower `RANK_SHARPNESS` first if you want room above. And `lambda_` and
`RANK_SHARPNESS` are one degree of freedom, not two, so sweeping both is
sweeping the same axis twice. This is the continuous-side twin of the
`temp`/`COMMITMENT` collapse in
[limitations.md](limitations.md#open-design-questions).

The ceiling itself is not the limitation — it is close to optimal. Raising
`MAX_SHARPNESS` above 10 degrades monotonically across all 18 problems at 20
seeds: 14.0 loses 64-288, 20.0 loses 9-349, 30.0 loses 6-352.

`n_warmup` does not freeze the whole model. Warm-up trials sample uniformly and
their `quality` is held at 0, so no continuous archive moves — but categorical
nodes are scored by mean rank over *every* visit, and warm-up trials are
included deliberately: they are the unbiased sample that stops a branch being
written off before it has been tried. Expect `prob` to have moved and
`n_updates` to be non-zero after a run that was entirely warm-up.

`n_warmup` is also **not** skipped when `start_prob_tree` is supplied. If a
warm-started run should go straight to modelling, pass `n_warmup=0` — which is
the default.

### Run control

| Parameter | Type | Default | Description |
|---|---|---|---|
| `start_prob_tree` | `dict` \| `None` | `None` | A previous run's `prob_tree`. Validated against `searchspace`, deep-copied, never mutated. |
| `verbose` | `bool` | `True` | Print one progress line per trial to stdout. |
| `seed` | `int` \| `None` | `None` | Seed for this run's private RNG. Global `random` state is never read or disturbed. |

### Parallel evaluation

| Parameter | Type | Default | Description |
|---|---|---|---|
| `batch` | `int >= 1` or `-1` | `1` | Configurations drawn per model update. `-1` resolves to `os.cpu_count()`. |
| `executor` | object \| `None` | `None` | Anything with an order-preserving `map(fn, iterable)`, or a `submit(fn, arg)` returning futures. |

`batch=1` is sequential and is what earns model-based search its sample
efficiency. `batch=k` draws `k` configurations from one state of the tree, so
the model is stale for the other `k-1` — pay that to keep an `executor` busy,
not for smoother convergence. Measured cost at fixed budget across 18 problems
and 20 seeds:

| `batch` | Median extra regret vs sequential |
|---|---|
| 2 | −3.3% |
| 4 | −0.1% |
| 8 | +2.6% |
| 16 | +21.3% |
| 32 | +43.1% |

Reproduce with `python benchmarks/batch_cost.py 20 --sizes 1,2,4,8,16,32`.
Up to `k=8` the cost is within seed noise; past `k=16` it is real. Batch sizes
above the number of workers you actually have are paying the cost for nothing.

**BUTChC never creates a pool.** You supply the executor, which keeps the
package dependency-free, keeps process lifecycle out of a library, and leaves
the picklability decision with the person who wrote the objective. Known-good:
`concurrent.futures.ThreadPoolExecutor` and `ProcessPoolExecutor`,
`multiprocessing.Pool`, joblib, dask, `mpi4py.futures`.

Order preservation is required. Updates are applied in *sample* order, so a
seeded run gives the same answer regardless of which worker finishes first. An
executor returning results as they complete would silently break reproducibility.

`batch>1` with `executor=None` evaluates the batch inline. That is a real mode —
it is how `benchmarks/batch_cost.py` measures the cost without any parallelism
confounding it — but it buys no speed. If you set `batch` for wall clock, set
`executor` too.

---

## Search space format

A plain dict, so it is JSON-serializable and needs no imports to write.

### Categorical

```python
'activation': {'values': ['relu', 'tanh', 'elu']}
```

Values must be hashable and unique.

### Continuous

```python
'dropout':       {'min': 0.0,  'max': 0.5}                  # linear
'learning_rate': {'min': 1e-5, 'max': 1e-1, 'log': True}    # log10-uniform
'n_layers':      {'min': 1,    'max': 6,    'int': True}    # integer-valued
```

| Key | Meaning |
|---|---|
| `min`, `max` | Inclusive bounds. Required together. |
| `log` | Model in log10 space. Requires `min > 0`. Cannot combine with `int`. |
| `int` | Objective receives a real Python `int`; the tree learns from the rounded value it actually saw. |

Use `log: True` whenever the range spans more than about one order of
magnitude. Sampled linearly, `[1e-5, 1e-1]` puts 99.99% of its mass above
`1e-3` — the bottom three decades are effectively unreachable.

### Conditional parameters — `next_level`

A categorical node maps each choice to a sub-searchspace. Those parameters are
sampled and updated only when their parent value is chosen.

```python
'optimizer': {
    'values': ['adam', 'sgd', 'lbfgs'],
    'next_level': {
        'adam': {'lr': {'min': 1e-4, 'max': 1e-2, 'log': True}},
        'sgd':  {'lr':       {'min': 1e-3, 'max': 1e-1, 'log': True},
                 'momentum': {'min': 0.0,  'max': 0.99}},
        # 'lbfgs' takes no sub-params — omitting it is fine
    },
}
```

Each branch keeps its own model, so learning the best `lr` for adam does not
interfere with learning it for sgd.

#### Nesting deeper

`next_level` is the only nesting mechanism, and it composes: a sub-searchspace
is an ordinary searchspace, so any categorical inside one can carry its own
`next_level`, to any depth.

```python
searchspace = {
    'task': {
        'values': ['vision', 'text'],
        'next_level': {
            'vision': {
                'backbone': {                          # level 2 choice
                    'values': ['resnet', 'vit'],
                    'next_level': {
                        'resnet': {'depth': {'values': [18, 50]},
                                   'lr': {'min': 1e-4, 'max': 1e-1, 'log': True}},
                        'vit':    {'patch': {'values': [8, 16]},
                                   'lr': {'min': 1e-5, 'max': 1e-2, 'log': True}},
                    },
                },
            },
            'text': {'lr': {'min': 1e-5, 'max': 1e-3, 'log': True}},
        },
    },
    'seed_pool': {'values': [0, 1]},                   # always present
}
```

The objective receives only the parameters on the path that was sampled:

```python
{'task': 'vision', 'backbone': 'resnet', 'depth': 50, 'lr': 0.003, 'seed_pool': 1}
{'task': 'vision', 'backbone': 'vit',    'patch': 16, 'lr': 2e-4,  'seed_pool': 0}
{'task': 'text',                                      'lr': 4e-5,  'seed_pool': 1}
```

Note `lr` appearing three times with three different ranges. That is allowed
and is the point: a name may repeat across branches that cannot co-occur, and
each occurrence is a separate node learning from only its own trials. See
[names must be unique across the tree](#names-must-be-unique-across-the-tree)
for the rule that governs when a repeat is legal.

#### There is no implicit nesting

A sub-space must hang off `next_level`. Hanging it directly off the choice
name looks natural — the whole search space is JSON-like, so it reads as
though structure alone should work — but it is rejected:

```python
# WRONG — no 'next_level'
{'model': {'values': ['svm', 'rf'],
           'svm': {'C': {'min': 0.1, 'max': 10.0}}}}
```
```
SearchSpaceError: model: unexpected keys ['svm'] on a categorical node.
Allowed: ['next_level', 'prior', 'prior_strength', 'values']
```

A categorical node accepts exactly those four keys, so this fails at
validation, before the first objective call, rather than silently dropping the
sub-space.

### Priors

```python
'optimizer': {
    'values': ['adam', 'sgd', 'lbfgs'],
    'prior': {'adam': 0.7, 'sgd': 0.25, 'lbfgs': 0.05},   # must sum to 1
    'prior_strength': 30,
},
'learning_rate': {
    'min': 1e-5, 'max': 1e-1, 'log': True,
    'prior': {'mean': 1e-3, 'std': 5e-4},   # or a list: [1e-3, 3e-3]
    'prior_strength': 20,
},
```

`prior_strength` is denominated in **pseudo-trials**: how many real
observations your belief is worth. Default 10.

| Value | Reading |
|---|---|
| 5 | a hint; a dozen trials will overturn it |
| 50 | a strong belief; needs sustained contrary evidence |
| 500 | near-commitment; use when you are confident |

That scale applies to **categorical** priors, which decay as
`strength / (strength + visits)` and have no ceiling.

**Continuous priors saturate, and much sooner than the scale suggests.** The
prior is seeded as reservoir points, and `MIN_GRID_POINTS` of flat grid always
survive, so the most it can ever occupy is

```
KDE_RESERVOIR_SIZE - MIN_GRID_POINTS      # 25 - 10 = 15
```

Above that, `prior_strength` is silently clamped: on a continuous parameter,
`50` and `500` place exactly the same 15 points and are indistinguishable. If
you want a continuous prior to dominate for longer, raise
`KDE_RESERVOIR_SIZE` — raising `prior_strength` past 15 does nothing. The
ceiling tracks the reservoir size, so it moves if you retune that.

Continuous priors are given in the units you defined, so a `log` parameter takes
its prior on the original scale. Values outside `[min, max]` are rejected.

A prior is a starting point, not a constraint — the surviving grid still spans
the full range, so no `prior_strength` can make part of it unreachable.

### Names must be unique across the tree

Configs are flattened into one dict, so two parameters that can appear together
must not share a name. Names *may* repeat across sibling branches of the same
`next_level` — those never co-occur, and that is the point of `next_level`.
Violations are rejected at definition time with a message naming both.

---

## Return value

```python
{
    'best_params':  dict | None,  # highest-scoring config; None if all trials were non-finite
    'best_value':   float,        # its objective value; -inf if all trials were non-finite
    'prob_tree':    dict,         # final probability tree — pass to start_prob_tree
    'history':      list[dict],   # one entry per trial, in trial order
    'loss_history': list[float],  # per-trial loss
    'rolling_loss': list[float],  # per-trial rolling mean loss
    'n_updates':    int,          # trials that moved some distribution
    'n_gated':      int,          # trials that cleared the quality gate
    'n_batches':    int,          # model updates performed; equals budget when batch=1
}
```

Each `history` entry:

| Key | Type | Meaning |
|---|---|---|
| `params` | `dict` | The configuration evaluated |
| `objective` | `float` | What the objective returned |
| `loss` | `float` | How far the tree moved on this trial |
| `rolling_loss` | `float` | Rolling mean of `loss` over `max(10, budget // 20)` trials |
| `quality` | `float` | Gated rank weight in `[0, 1]` that drove the continuous archive |
| `updated` | `bool` | Some distribution actually shifted |
| `gated` | `bool` | Cleared the `gamma` gate and reached a continuous archive |
| `batch_index` | `int` | Which batch this trial belonged to; `0, 1, 2, …` |

`loss` measures tree movement, not configuration quality. Read `n_updates` and
`n_gated` together to separate the two reasons a rolling loss can fall: the
distribution has settled, or few trials are clearing the gate.

`best_params` and `prob_tree` answer different questions — the best single point
seen, versus where the model would spend the next trial. A large disagreement
means the budget was too small for the tree to settle.

---

## Errors

| Raised | When |
|---|---|
| `SearchSpaceError` | The search space is malformed. The message names the offending path. Subclasses `ValueError`; import with `from butchc import SearchSpaceError`. |
| `ValueError` | A hyperparameter is out of range, or `start_prob_tree` does not match `searchspace`. |
| `TypeError` | `searchspace` is neither a dict nor a `ConfigurationSpace`; `objective` is not callable or returned a non-numeric value; `executor` has neither `map` nor `submit`. |
| `RuntimeError` | The objective could not be sent to the executor. The message explains the pickling constraint. |
| `UnsupportedSpace` | A `ConfigurationSpace` has no faithful tree representation. From `butchc.interop`. |

Every search-space and hyperparameter check runs **before the first objective
evaluation**, so a typo costs nothing when trials are expensive.

---

## `reservoir_summary`

```python
from butchc import reservoir_summary

reservoir_summary(node) -> {'mean': float, 'std': float, 'ess': float, 'n': int}
```

Summarises one continuous node of a probability tree. `mean` and `std` come back
in the units you defined, so a `log` parameter reports on its original scale.

`ess` is the Kish effective sample size — how many independent observations the
reservoir is worth. An `ess` far below `n` means the distribution has collapsed
onto a handful of points.

---

## `butchc.interop`

Optional conversions. Not imported by `butchc`, so ConfigSpace never becomes a
dependency of the core package.

```bash
pip install butchc[configspace]
```

```python
from butchc.interop import (
    from_configspace, to_configspace, wrap_objective,
    to_json, from_json, UnsupportedSpace,
)
```

### `from_configspace(space, drop=()) -> (searchspace, fixed, casts)`

Converts a `ConfigurationSpace`. `drop` names parameters to exclude entirely —
fidelity knobs (`trainsize`, `repl`, `epoch`) and task selectors (`task_id`)
belong there.

`fixed` holds constants to merge into every config; `casts` holds per-parameter
callables to apply. `wrap_objective` applies both, and `BUTChC_optimize` calls
it for you when handed a `ConfigurationSpace` directly.

| Converts | To |
|---|---|
| `CategoricalHyperparameter` | `{'values': [...]}` |
| `OrdinalHyperparameter` | `{'values': [...]}` — **order is discarded** |
| `UniformFloatHyperparameter` | `{'min', 'max', 'log'}` |
| `UniformIntegerHyperparameter` | `{'min', 'max', 'int'}` |
| log-scaled integer | log continuous + a rounding cast |
| `Constant`, `UnParametrizedHyperparameter` | an entry in `fixed` |
| `EqualsCondition` | child under `parent.next_level[value]` |
| `InCondition` | child under each named branch |

Refused, with `UnsupportedSpace` naming the parameter:

| Refused | Why |
|---|---|
| `AndConjunction`, `OrConjunction` | a child gated on two parents has no single place in a tree |
| multiple parents for one child | same |
| forbidden clauses | BUTChC cannot express "invalid combination"; it would sample them and waste budget |
| non-categorical parent | only a categorical node can carry a `next_level` |

ConfigSpace expresses conditionality as a DAG, BUTChC as a tree; the DAG is
strictly more general, so an approximation would mean silently optimising a
different problem.

### `wrap_objective(objective, fixed, casts) -> callable`

Wraps an objective so every config it receives has `fixed` merged in and
`casts` applied, using the two values `from_configspace` returns. The wrapper
is picklable, so it works with a process pool. Returns `objective` unchanged
when both are empty.

### `to_configspace(searchspace, name='butchc', seed=None) -> (space, names)`

The reverse direction, for handing your space to a benchmark harness or to SMAC
or Optuna for a like-for-like comparison. Never refuses.

One reconciliation: BUTChC allows a name to repeat across sibling branches,
ConfigSpace requires global uniqueness. Collisions are qualified as
`parent:value:name` and `names` maps each ConfigSpace name back to its BUTChC
one — the identity for anything that did not need qualifying.

Priors are **not** carried across. ConfigSpace's prior support does not share
BUTChC's pseudo-trial semantics, and a silently reinterpreted prior is worse
than an absent one.

### `to_json(searchspace, **kwargs) -> str` / `from_json(text) -> dict`

Round-trips a search space through JSON. `**kwargs` goes to `json.dumps`.

`from_json` is not just `json.loads`: JSON object keys are always strings, so a
space branching on `{'values': [1, 2, 3]}` would come back with `next_level`
keyed by `"1"`, `"2"`, `"3"`, silently unreachable. `from_json` re-keys each
`next_level` against its own node's `values`.

---

## `prune` and `prune_report`

```python
from butchc import prune, prune_report

prune(searchspace, result, threshold=1/3, keep_min=2, protect_best=True,
      narrow=None, return_tree=False)
```

Reduce a search space using what a finished run learned, for handing off to
another optimiser or for a second, narrower run. Returns a new search space;
the input is not modified.

| Parameter | Default | Meaning |
|---|---|---|
| `searchspace` | — | The space the run used |
| `result` | — | The dict returned by `BUTChC_optimize` |
| `threshold` | `1/3` | Fraction of uniform at or below which a choice is dropped. `0.0` drops nothing |
| `keep_min` | `2` | Choices kept per node regardless of probability |
| `protect_best` | `True` | Keep the branch that produced `best_params`, whatever its probability |
| `narrow` | `None` | If set, also narrow continuous bounds to `mean ± narrow * std`, clipped to the originals |
| `return_tree` | `False` | If `True`, also return a probability tree matching the pruned space, for `start_prob_tree` |

Raises `SearchSpaceError` if `result` is not an optimizer result, if
`keep_min < 1`, or if `narrow <= 0`.

`prune_report(searchspace, pruned)` returns a list of human-readable lines
describing what changed, and an empty list when nothing did.

A branch's probability says where the model would spend the next trial, not
that the branch is empty of good configurations — see
[examples](examples.md#pruning-a-space-and-handing-it-off) for what the
defaults protect against and when pruning is too early to be meaningful.

**Warm-starting a follow-up run needs `return_tree=True`.** `result['prob_tree']`
describes the *original* space — every dropped choice and the original,
wider continuous bounds are still in it — so passing it as `start_prob_tree`
for the pruned space raises `ValueError`, because a warm-start tree must match
its search space exactly. `prune(..., return_tree=True)` returns
`(searchspace, prob_tree)` instead, where `prob_tree` carries the learned
statistics for only the choices and archive points that survived pruning and
is guaranteed to match the returned `searchspace` exactly.

---

## Tuning by problem shape

The shipped defaults are an average over problem shapes. You have one shape.
`benchmarks/tune.py --regime NAME`, in the repository, sweeps against a subset
of problems sharing one property and prints what that shape wants instead:

```bash
python benchmarks/tune.py 6 --rounds 1 --regime branched
```

Measured at 6 selection seeds and one round, against the 0.5.1 defaults —
enough to indicate a direction, not enough to settle a default. Treat as a
starting point and confirm on your own space. Names are `tune.py`'s:
`commitment`, `discount`, `neutral` and `reservoir` are the
[module constants](#module-constants) `COMMITMENT`, `DISCOUNT`,
`NEUTRAL_QUALITY` and `KDE_RESERVOIR_SIZE`; `warmup_frac` is `n_warmup` as a
fraction of `budget`.

| Regime | Problems it covers | Recommendation |
|---|---|---|
| `branched` | conditional structure | `lambda_ 4.0`, `alpha 0.3`, `commitment 6.0`, `discount 0.95`, `neutral 0.3` |
| `highdim` | many continuous parameters | no change |
| `multimodal` | dense local optima | `warmup_frac 0.1`, `reservoir 25` (now the default) |
| `flat` | plateaus and ties | `gamma 0.5` |
| `noisy` | observation noise | `explore 0.1` |

The `branched` recommendation is the one to read with most suspicion: `alpha
0.3` against the default of `3.0` is a tenfold swing selected on four
problems, which is exactly the shape of an overfitted result. It is also
directionally consistent with the low-budget measurements, where faster
commitment helped medians on conditional problems, so it is worth trying rather
than dismissing.

### How the bandwidth floor was chosen, and why it moved five other defaults

This is a worked example of retuning the library: one knob that wins
everywhere invalidates the rest.

Every regime sweep selected `min_bandwidth` below the then-shipped `0.01`. A
knob that wins in every regime is not a regime finding — it is evidence the
default is wrong. Measured directly across all 18 problems at 20 paired seeds:

| `min_bandwidth` | 0.01 | 0.003 | 0.001 | 0.0003 |
|---|---|---|---|---|
| Win-loss vs `0.01` | — | 192-41 | **200-34** | 195-40 |
| Mean rank | 0.458 | 0.653 | **0.722** | 0.667 |
| Ackley 5D | -1.7597 | -0.3745 | **-0.1159** | -0.0337 |
| 20D sphere | -0.1196 | -0.0135 | **-0.0024** | -0.0106 |
| Styblinski 4D | -0.0325 | -0.0061 | -0.0001 | **0.0006** |

The reversal below `0.001` is what makes this an interior optimum rather than
the edge of the grid searched.

Changing it invalidated the selection of every other default, since all of them
had been chosen with the floor at `0.01`. The re-sweep that followed moved five:
`alpha` 10.0→3.0, `gamma` 0.7→0.85, `explore` 0.0→0.05, `KDE_RESERVOIR_SIZE`
50→25, and `min_bandwidth` itself to `0.0003` — which it only preferred *after*
the reservoir shrank. Those two interact strongly and should be retuned as a
pair; at reservoir 50 the ordering reverses and `0.001` is better.

`KDE_RESERVOIR_SIZE` turned out to carry more of the gain than anything else.
Reverting that one value while keeping every other new default turns a
13-better/4-worse result into 5-better/12-worse.

---

## Module constants

Patchable for experiments. Changing them changes results, so pin what you ship.

**Patch in** is the module whose binding is actually read at call time. Where
two are listed, the constant is imported by name into a second module and
`from x import y` creates an independent binding, so patching only the
defining module leaves half the code on the old value — a knob in that state
reports a winner that means nothing. **Swept** marks the knobs
`benchmarks/tune.py` varies; the rest are fixed by argument or measured once.

| Constant | Defined in | Patch in | Swept | Default | Controls |
|---|---|---|---|---|---|
| `COMMITMENT` | `butchc.optimizer` | `optimizer` | yes | `8.0` | Categorical sharpening exponent at end of run; ramps from 0 |
| `MOVEMENT_EPSILON` | `butchc.optimizer` | `optimizer` | no | `1e-12` | Smallest delta counted as real movement. Reporting only — it cannot change the search |
| `DISCOUNT` | `butchc._update` | `_update` | yes | `0.98` | Per-visit decay on categorical statistics |
| `NEUTRAL_QUALITY` | `butchc._update` | `_update` | yes | `0.5` | Score credited to an untried choice |
| `KDE_RESERVOIR_SIZE` | `butchc._utils` | `_utils`, `_tree`, `_update` | yes | `25` | Points retained per continuous node |
| `MIN_BANDWIDTH_FRACTION` | `butchc._utils` | `_utils` | yes | `0.0003` | Bandwidth floor as a fraction of range |
| `MIN_GRID_POINTS` | `butchc._utils` | `_utils`, `_tree` | no | `10` | Grid points a prior can never displace |
| `RANK_SHARPNESS` | `butchc._utils` | `_utils` | yes | `3.0` | Archive weight ratio at `lambda_ = 1` |
| `MAX_SHARPNESS` | `butchc._utils` | `_utils` | no | `10.0` | Ceiling, so large `lambda_` cannot collapse the KDE |

`benchmarks/tune.py::apply_config` sets every binding in the **Patch in**
column and is the reference implementation. A constant patched in only its
defining module will appear to have no effect.
