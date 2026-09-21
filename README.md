# BUTChC

**B**ayesian **U**pdate **T**ree **Ch**ained **C**onditionally — a dependency-free, probabilistic black-box hyperparameter optimizer.

BUTChC maintains a probability distribution over a **hierarchical, conditional search space** and refines it from observed objective values. Parameters can depend on choices made higher up the tree: `momentum` exists only when `optimizer=sgd`, `kernel_size` only when `model=cnn`. No trial is spent on irrelevant combinations, and each branch learns its own parameters from only the trials that used it.

No gradients, no differentiability, no assumptions about the objective's internals.

- **Zero dependencies** — Python ≥ 3.8 standard library only
- **Conditional search spaces** — nested arbitrarily deep via `next_level`
- **Non-parametric continuous model** — a weighted KDE elite archive, no distributional assumptions
- **Scale-invariant** — updates use rank, not raw objective, so an objective in the millions behaves like one in `[0, 1]`
- **Parallel evaluation** — bring your own executor; threads, processes, joblib, dask
- **Reproducible** — `seed` gives a private RNG, and the answer does not depend on which worker finishes first
- **Fails fast** — search spaces and hyperparameters are validated before the first objective call
- **Prunable** — `prune` turns a finished run into a smaller space, to search again or hand to another optimiser

📖 **[API reference](docs/api.md)** · **[Examples](docs/examples.md)** · **[Design notes](docs/design.md)** · **[Codemap](docs/codemap.md)** · **[Changelog](CHANGELOG.md)**

---

## Install

```bash
pip install .                      # from a checkout
pip install .[configspace]         # + ConfigSpace interop (optional)
```

Run the tests:

```bash
pip install pytest && pytest
```

---

## Quick start

```python
from butchc import BUTChC_optimize

searchspace = {
    # Top-level choice of model family. Each choice unlocks its own
    # sub-parameters via 'next_level'.
    'model_type': {
        'values': ['svm', 'random_forest', 'neural_net'],
        'next_level': {
            'svm': {
                'kernel': {'values': ['rbf', 'linear', 'poly']},
                'C':      {'min': 0.01, 'max': 100.0, 'log': True},
                'gamma':  {'min': 1e-4, 'max': 10.0,  'log': True},
            },
            'random_forest': {
                'n_estimators': {'values': [50, 100, 200, 500]},
                'max_depth':    {'min': 2, 'max': 30, 'int': True},
                'max_features': {'values': ['sqrt', 'log2']},
            },
            'neural_net': {
                'learning_rate': {'min': 1e-4, 'max': 1e-1, 'log': True},
                'hidden_units':  {'values': [64, 128, 256, 512]},
                'dropout':       {'min': 0.0, 'max': 0.5},
            },
        },
    },
    # Always present, whatever model_type is chosen
    'preprocessing': {'values': ['standard_scaler', 'min_max', 'none']},
}

def objective(config):
    # config carries 'model_type' and 'preprocessing', plus only the params
    # of the chosen branch, e.g.
    #   {'model_type': 'svm', 'kernel': 'rbf', 'C': 4.2, 'gamma': 0.01, ...}
    # 'dropout' never appears in an svm config; 'C' never in a neural_net one.
    return cross_val_score(build_model(config), X, y).mean()   # higher is better

results = BUTChC_optimize(
    searchspace = searchspace,
    objective   = objective,
    budget      = 150,
    seed        = 0,
)

print(results['best_params'])
print(f"Best score: {results['best_value']:.4f}")
```

BUTChC always **maximizes**. Negate to minimize.

### Slow objective? Use an executor

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=8) as pool:
    results = BUTChC_optimize(searchspace, objective, budget=200,
                              batch=8, executor=pool, seed=0)
```

### Already have a ConfigSpace?

```python
results = BUTChC_optimize(configuration_space, objective, budget=200, seed=0)
```

---

## Defining a search space

A search space is a plain dict — JSON-serializable, and no imports needed to write one.

```python
'activation':    {'values': ['relu', 'tanh', 'elu']}        # categorical
'dropout':       {'min': 0.0,  'max': 0.5}                  # linear
'learning_rate': {'min': 1e-5, 'max': 1e-1, 'log': True}    # log10-uniform
'n_layers':      {'min': 1,    'max': 6,    'int': True}    # integer-valued
```

Use `log: True` whenever the range spans more than about one order of magnitude. Sampled linearly, `[1e-5, 1e-1]` places 99.99% of its mass above `1e-3`, leaving the bottom three decades effectively unreachable.

A categorical node can map each of its choices to a sub-searchspace via `next_level`. Those parameters are sampled and updated only when their parent value is chosen:

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

Each branch keeps its own model, so learning the best `lr` for adam does not interfere with learning the best `lr` for sgd. Nesting is arbitrarily deep.

Any node can carry a `prior` and a `prior_strength` measured in pseudo-trials. See the [API reference](docs/api.md#search-space-format) for the full format, priors, and the uniqueness rule for names.

---

## How it works

1. **Initialize** a probability tree from the search space. Categorical nodes start uniform; continuous nodes start with an evenly spaced reservoir covering the range.

2. **Sample** by traversing the tree. Categorical nodes draw from a temperature-scaled softmax. Continuous nodes pick a reservoir point weighted by its rank, add Silverman-bandwidth Gaussian jitter, and *reflect* back into range.

3. **Evaluate** the objective.

4. **Rank** the result against every finite objective seen so far, counting ties as half. Continuous nodes apply the `gamma` gate and convert what survives into a `quality` in `[0, 1]`; categorical nodes use the raw rank, ungated.

5. **Update**. Categorical nodes score each choice by its recency-weighted mean rank, smoothed by `alpha` pseudo-visits. Continuous nodes append to an elite archive, evict the worst-scoring entry, and reweight geometrically by rank.

6. **Repeat**, with categorical commitment ramping up over the budget.

Two properties are load-bearing. **Reflection rather than clipping**: jitter clipped to `[min, max]` deposits probability mass on each bound and biases every search toward interval edges. **Rank rather than raw objective**: raw-value weighting makes behaviour depend on units, and one catastrophic outlier could dominate the archive permanently.

Full reasoning in [design notes](docs/design.md).

---

## Tuning

| Parameter | Start | Increase if… | Decrease if… |
|---|---|---|---|
| `lambda_` | `2.0` | tree adapts too slowly | converging too fast to a suboptimal region |
| `alpha` | `3.0` | many categorical options, small budget | want faster commitment to early evidence |
| `temp` | `1.0` | categorical exploration too greedy | budget spent on clearly bad choices |
| `gamma` | `0.85` | objective is noisy; want only strong trials to count | want more trials contributing signal |
| `explore` | `0.05` | search collapses to a local optimum early | objective is expensive and smooth |
| `batch` | `1` | you have idle workers | you have no executor |
| `budget` | 10× param count | rolling loss has not plateaued | rolling loss flat after 20% of the run |

`lambda_` and `alpha` act on different node types and do not interact: `lambda_` controls how sharply continuous archives concentrate, `alpha` how slowly categorical nodes commit. Note `lambda_` saturates — it acts only through `min(RANK_SHARPNESS * lambda_, MAX_SHARPNESS)`, so any value at or above 3.33 is clipped and does nothing. See [api.md](docs/api.md#butchc_optimize).

The two knobs most worth reaching for are not in this table. `KDE_RESERVOIR_SIZE` (default 25) and `MIN_BANDWIDTH_FRACTION` (default 0.0003) between them decide how hard the continuous model concentrates, and they carry more of the 0.6.0 gain than anything else. They interact, so retune them together: a **smooth, high-dimensional** space wants the gentler pair (`50` and `0.001`), while conditional and multimodal spaces want the sharp defaults.

---

## Benchmarks

Median best objective over 30 paired seeds — same budget, same seeds, every
method. Every problem is a maximization with optimum 0, so nearer zero is
better. Measured against the shipped 0.6.0 defaults; re-running the command
below reproduces the tables.

```bash
python benchmarks/evaluate.py 30
python benchmarks/evaluate.py 10 --suite heldout --methods tpe,butchc
```

Random search is a low bar, so the comparator carried throughout is TPE — same
niche, and what a user choosing against BUTChC would actually reach for.
`benchmarks/baselines.py` carries a dependency-free TPE and an Optuna
`TPESampler` wrapper. `evaluate.py` also reports paired win-loss records and
two-sided exact sign-test p-values; those records, not the medians, are what
the claims below rest on.

**Tuned on** — the defaults were selected against these, so read them as
optimistic. Win-loss is BUTChC against TPE.

| Problem | Budget | Random | TPE | BUTChC | vs TPE |
|---|---|---|---|---|---|
| 2D quadratic | 200 | -0.0425 | -0.0004 | **-0.0000** | 30-0, p=0.000 |
| 5D sphere | 500 | -3.7393 | -0.1312 | **-0.0000** | 29-1, p=0.000 |
| Rosenbrock | 500 | -0.1036 | **-0.0202** | -0.0368 | 13-17, p=0.585 |
| Rastrigin 4D | 600 | -17.5005 | -8.6510 | **-5.1169** | 26-4, p=0.000 |
| 10D sphere | 1000 | -19.0750 | -2.2176 | **-0.0036** | 30-0, p=0.000 |
| Ackley 5D | 600 | -14.1027 | -4.3008 | **-0.0323** | 30-0, p=0.000 |
| Log-scale target | 200 | -0.0085 | -0.0006 | **-0.0000** | 29-1, p=0.000 |
| Branch trap | 300 | -0.6219 | -1.0004 | **-0.0189** | 25-5, p=0.000 |
| Categorical mix | 300 | -0.3226 | -0.0680 | **-0.0004** | 28-2, p=0.000 |
| Integer mix | 300 | -0.2727 | -0.0016 | **-0.0000** | 26-4, p=0.000 |
| Plateau (ties) | 300 | -0.5000 | -0.5000 | -0.5000 | 0-0, p=1.000 |

**Held out** — never consulted while tuning, so this is the honest read on
whether the defaults generalise.

| Problem | Budget | Random | TPE | BUTChC | vs TPE |
|---|---|---|---|---|---|
| Griewank 6D | 800 | -1.0183 | -0.4915 | **-0.4383** | 18-12, p=0.362 |
| Styblinski 4D | 600 | -20.0007 | -4.0751 | **+0.0007** | 30-0, p=0.000 |
| Nested pipeline | 400 | -0.2970 | -0.0096 | **-0.0089** | 15-15, p=1.000 |
| Optimiser choice | 300 | -0.0277 | -0.0007 | **-0.0001** | 25-5, p=0.000 |
| Rastrigin 8D | 1000 | -62.4076 | -39.0921 | **-21.5439** | 28-2, p=0.000 |
| 20D sphere | 1500 | -69.0434 | -18.8961 | **-0.1255** | 30-0, p=0.000 |

Under observation noise, scoring the *reported* best on the noise-free function
(5D sphere, N(0,1) noise, budget 400): random -4.8604, TPE -0.8099, BUTChC
**-0.4490**, 19-11 against TPE at p=0.200 — better on the median, not
separable at 30 seeds.

### What the records say

Against TPE across all 18 problems: **13 statistically significant wins, zero
significant losses, 5 ties.** The margins are large where they are large —
`Styblinski 4D` reaches the optimum outright (+0.0007 against -4.0751, 30-0),
`20D sphere` lands 150× nearer it, `Ackley 5D` 133× nearer, and `Branch trap`
25-5 on a problem where TPE does *worse than random*, because the trap is
precisely the conditional structure a flat model cannot see.

The 0.6.0 retune closed both results the previous release conceded. `Optimiser
choice` went from a 12-18 loss to a **25-5 win**, and `Nested pipeline` from
10-20 to a **15-15 tie** with the medians now effectively equal (-0.0089
against -0.0096). Both are held out, so neither was available to fit against.

Three things still worth weighing, none of them a loss:

- **`Rosenbrock` 13-17, p=0.585.** Expected and structural: a narrow curved
  valley is parameter interaction, which the independence assumption cannot
  model. Read it as "no advantage", not a loss — and as the honest cost of
  modelling siblings independently.
- **`Griewank 6D` 18-12, p=0.362 and `Noisy 5D sphere` 19-11, p=0.200.** Better
  on the median, not separable at 30 seeds. More seeds would settle them.
- **`Nested pipeline` is a tie, not a win.** Conditional structure is where
  BUTChC is *designed* to win, and on this problem it now draws rather than
  leads. It also still arrives later than TPE there (324 trials against 188)
  even while matching the final answer.

`Plateau` is saturated — all three methods reach the discretised optimum, so it
is a regression guard for tie handling, not a discriminator.

**What these benchmarks do not show.** Every problem here is synthetic and was
written in this repository. The claim BUTChC is built on — that a pre-specified
conditional space avoids the budget a flat optimiser wastes on inactive
parameters — has no published-benchmark evidence behind it yet. The suite's own
`flat_tpe_search` exists to isolate exactly that and finds almost nothing,
because these spaces are too narrow for the waste to matter. The candidate that
would settle it, YAHPO Gym's `rbv2_super` (41 parameters, 75.6% inactive per
configuration, 103 real datasets), is currently unusable for a reason recorded
in [dev/eval/README.md](dev/eval/README.md).

### Tuning for your problem's shape

The defaults are an average over problem shapes. `benchmarks/tune.py --regime`
sweeps against problems sharing one property and prints what that shape wants:
`branched` spaces want faster commitment, `noisy` ones want `explore 0.1`,
`multimodal` ones want a smaller reservoir. The table is in
[docs/api.md](docs/api.md#tuning-by-problem-shape).

That mechanism is what produced 0.6.0's defaults. Every regime sweep preferred
a smaller `min_bandwidth` than the then-shipped `0.01`; measured directly,
`0.001` wins **200-34** across all 18 problems at 20 paired seeds. Because every
other default had been selected with the old floor, fixing it required a full
re-sweep, which moved five defaults in total — `alpha`, `gamma`, `explore`,
`KDE_RESERVOIR_SIZE` and `min_bandwidth` itself. The tables above are measured
against the result.

### How long a result takes to arrive

Final quality says where a method ends up, not when. `evaluate.py --anytime`
reports the other axis. Measured on the 0.6.0 defaults, 30 paired seeds, the
clearest framing is **how much budget BUTChC needs to match TPE's final
answer**:

| Problem | Budget | Trials BUTChC needed | Fraction of budget |
|---|---|---|---|
| 20D sphere | 1500 | 148 | **1/10.1** |
| 10D sphere | 1000 | 138 | **1/7.3** |
| Styblinski 4D | 600 | 146 | 1/4.1 |
| Ackley 5D | 600 | 155 | 1/3.9 |
| 5D sphere | 500 | 128 | 1/3.9 |
| Rastrigin 8D | 1000 | 325 | 1/3.1 |
| Integer mix | 300 | 100 | 1/3.0 |
| Categorical mix | 300 | 107 | 1/2.8 |
| Branch trap | 300 | 111 | 1/2.7 |
| Nested pipeline | 400 | 324 | 1/1.2 |

On every problem it reaches, BUTChC matches TPE's *final* result partway
through its own budget — median around a third of it.

Against TPE's own arrival time the picture is split, and the split is
informative. On high-dimensional problems BUTChC is far quicker: `20D sphere`
in 148 trials against TPE's 1133, on 30 of 30 seeds against TPE's 15. On the
conditional problems it is still slower to arrive even though it now matches or
beats TPE's final quality — `Nested pipeline` 324 against 188, `Optimiser
choice` 206 against 115, `Griewank 6D` 364 against 193.

Reliability is consistently BUTChC's. At 50% of the achievable range it reaches
the target on 27–30 of 30 seeds where TPE manages 10–29; at 99% on `20D sphere`
it arrives on 29 of 30 seeds while TPE never arrives at all.

So: quicker and more reliable on dimension, more reliable but later on
conditional structure. Full tables in
[`dev/tools/results/`](dev/tools/results/).

### What the optimiser itself costs

Sample efficiency is one axis; the wall clock the optimiser spends choosing is
another. With the objective stubbed to a constant, so the number is all
optimiser:

| Optimiser | 1D | 5D | 20D |
|---|---|---|---|
| **BUTChC** (pure Python, 0 deps) | **14 µs** | **52 µs** | **193 µs** |
| TPE (this repo, pure Python) | 1037 µs | 5219 µs | 21270 µs |
| Optuna `TPESampler` (numpy-backed) | 1433 µs | 6789 µs | 27864 µs |

BUTChC is **73–145× cheaper per trial** than either TPE. Zero dependencies is
not costing speed here: the numpy-backed implementation pays ~145× more per
suggestion, because TPE refits Parzen estimators over the whole history and
scores candidates, while BUTChC draws a reservoir point and jitters it —
`O(d·K)` with `K = 25`.

Read it alongside sample efficiency rather than instead of it, and note it is
close to irrelevant when a trial costs a model fit: against a 100 ms objective,
193 µs is 0.2%. Reproduce with `python benchmarks/overhead.py`.

### What batching costs

A batch of `k` leaves the model stale for `k-1` evaluations. Median extra regret against sequential, 18 problems × 20 seeds at matched budgets:

| `batch` | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| Extra regret | −3.3% | −0.1% | +2.6% | +21.3% | +43.1% |

```bash
python benchmarks/batch_cost.py 20 --sizes 1,2,4,8,16,32
```

Up to `k=8` the cost sits inside seed noise, which makes an 8× wall-clock speedup close to free. Past `k=16` it is real. Batch sizes above your worker count pay the cost for nothing.

### Reading the tables

- **Ratios flatter.** Most of these are squared errors, where an 800× ratio is about 28× in distance. The raw columns are the honest ones.
- **Medians hide the spread.** A median gap of 2× on one problem and a 30-0 record on another are different kinds of evidence. Prefer the win-loss column.
- **Rastrigin is the hard case.** Dense local optima limit how much any method modelling parameters independently can gain.
- **These are synthetic.** Every problem here was written alongside the optimiser, held-out or not. `dev/eval/` runs the same harness against YAHPO surrogates for a landscape nobody involved designed.

---

## Limitations

- **Independence assumption** — sibling nodes are modelled independently. Interactions between parameters in the same branch are not captured; encode known ones via `next_level`.
- **Maximization only** — negate to minimize.
- **No formal uncertainty bounds** — `rolling_loss` is a heuristic, not a posterior. It says the model stopped moving, not that the optimum was found.
- **Categorical ties commit arbitrarily** — when two branches reach the same optimum, budget concentrates on whichever wins first. Raise `alpha` or `temp` if you need both explored.
- **`best_params` and `prob_tree` can disagree** — the best configuration found may sit in a branch the tree assigns low probability. They answer different questions: one is the best single point seen, the other is where the model would spend the next trial. A large disagreement means the budget was too small for the tree to settle.
- **Branches are judged on average** — a branch whose best configuration is excellent but whose typical configuration is poor is at a disadvantage while its sub-parameters are untuned. Three mechanisms push back: a node commits more slowly the more sub-space its choices open, its statistics are recency-weighted so an early verdict decays, and the commitment exponent ramps from zero rather than from one. This is mitigation, not a guarantee — raise `alpha` or lower `COMMITMENT` if a branch you believe in is being abandoned. The `Branch trap` benchmark isolates this case.
- **No ordinal node type** — ordered discrete parameters are modelled as unordered choices, losing neighbourhood structure.
- **Not a Gaussian-process method** — on very small budgets (under ~50 trials) with cheap-to-model smooth objectives, a GP-based optimizer will typically do better. BUTChC's measured strengths are dimension, categorical mixtures, noise and objective scale, plus zero dependencies. Conditional structure is what it is *designed* for, and the benchmarks are mixed there: a decisive win on `Branch trap`, losses to TPE on `Nested pipeline` and `Optimiser choice`. Benchmark your own space before assuming the structure alone is a reason to choose it.

---

## Versioning

Defaults and internals change between minor versions, so **seeded runs do not reproduce across them**. See [CHANGELOG.md](CHANGELOG.md).

## License

MIT — see [LICENSE](LICENSE).
