# Benchmarks

The full results behind the headline numbers in the README. Every table here
can be reproduced from a clone of the repository with the command shown next
to it.

- [How the comparison is run](#how-the-comparison-is-run)
- [Final quality](#final-quality)
- [Under observation noise](#under-observation-noise)
- [Summary of the win-loss records](#summary-of-the-win-loss-records)
- [How quickly a good result arrives](#how-quickly-a-good-result-arrives)
- [What the optimizer itself costs](#what-the-optimizer-itself-costs)
- [What batching costs](#what-batching-costs)
- [How conditional real search spaces are](#how-conditional-real-search-spaces-are)
- [Caveats](#caveats)

---

## How the comparison is run

The main comparison is against **TPE** (Tree-structured Parzen Estimator), the
algorithm behind Optuna's default sampler and Hyperopt. TPE fills the same
niche as BUTChC, so it is what most users would choose instead. Random search
is included as a floor.

- **Same budget, same seeds.** Every method gets the same number of
  evaluations on the same 30 seeds, and results are compared seed by seed.
- **The claims rest on win-loss records, not medians.** For each problem,
  "30-0" means BUTChC finished ahead on all 30 seeds. The p-value is a
  two-sided exact sign test on that record. We call a result significant when
  p < 0.05.
- **Two TPE implementations.** `benchmarks/baselines.py` has a dependency-free
  TPE and a wrapper around Optuna's `TPESampler`.
- **Tuned-on and held-out problems.** BUTChC's defaults were chosen by
  sweeping against the *tuned-on* problems only. The *held-out* problems were
  never looked at during that sweep, so they are the fairer test of whether
  the defaults carry over to new problems.

```bash
python benchmarks/evaluate.py 30
python benchmarks/evaluate.py 10 --suite heldout --methods tpe,butchc
```

---

## Final quality

Median best objective over 30 seeds. Every problem is set up as a maximization
with its optimum at 0, so **closer to zero is better**. (Styblinski's offset is
rounded, so it can read slightly above 0.)

### Tuned on

| Problem | Budget | Random | TPE | BUTChC | BUTChC vs TPE |
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

### Held out

| Problem | Budget | Random | TPE | BUTChC | BUTChC vs TPE |
|---|---|---|---|---|---|
| Griewank 6D | 800 | -1.0183 | -0.4915 | **-0.4383** | 18-12, p=0.362 |
| Styblinski 4D | 600 | -20.0007 | -4.0751 | **+0.0007** | 30-0, p=0.000 |
| Nested pipeline | 400 | -0.2970 | -0.0096 | **-0.0089** | 15-15, p=1.000 |
| Optimiser choice | 300 | -0.0277 | -0.0007 | **-0.0001** | 25-5, p=0.000 |
| Rastrigin 8D | 1000 | -62.4076 | -39.0921 | **-21.5439** | 28-2, p=0.000 |
| 20D sphere | 1500 | -69.0434 | -18.8961 | **-0.1255** | 30-0, p=0.000 |

Four of the six held-out problems are significant wins, using defaults that
were never tuned on them.

---

## Under observation noise

On a 5D sphere with N(0, 1) noise added to every evaluation (budget 400), each
method's *reported* best configuration is re-scored on the noise-free function:

| Random | TPE | BUTChC |
|---|---|---|
| -4.8604 | -0.8099 | **-0.4490** |

BUTChC has the better median, but the record against TPE is 19-11 (p=0.200),
which is not significant at 30 seeds.

---

## Summary of the win-loss records

Across all 18 problems: **13 significant wins, zero significant losses, 5
ties.**

Where BUTChC wins, it usually wins by a lot. It reaches the optimum outright
on `Styblinski 4D` (+0.0007 against TPE's -4.0751, 30-0). Its final objective
is 150× closer to the optimum on `20D sphere` and 133× closer on `Ackley 5D`.
(Most of these objectives are squared errors, so a ratio like 150× overstates
the distance in parameter space. Read the raw columns too.) On `Branch trap`,
TPE does *worse than random search*, because the trap is built from exactly
the conditional structure a flat model cannot see. BUTChC wins it 25-5.

The five ties are `Rosenbrock`, `Griewank 6D`, the noisy sphere, `Nested
pipeline` and `Plateau`. None is a significant loss. `Rosenbrock` rewards
modelling how two parameters interact, which BUTChC does not do. `Griewank`
and the noisy sphere favour BUTChC on the median but are not separable at 30
seeds. On `Plateau`, all three methods reach the optimum. `Nested pipeline` is
a genuine draw on a conditional problem. Details are in
[limitations](limitations.md#the-non-results-in-full).

---

## How quickly a good result arrives

Final quality shows where a method ends up, not how long it took to get there.
`evaluate.py --anytime` measures that. The clearest way to read it is **how
many trials BUTChC needs to match TPE's final answer**:

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

On every problem where it gets there, BUTChC matches TPE's *final* result
partway through its own budget. The median is about a third of the budget.

Comparing how fast each method *itself* reaches a given level gives a more
mixed picture:

- **High-dimensional problems: BUTChC is much faster.** On `20D sphere` it
  gets there in 148 trials against TPE's 1133, and does so on 30 of 30 seeds
  against TPE's 15.
- **Conditional problems: BUTChC is slower to get there**, even where it
  matches or beats TPE's final result. `Nested pipeline` takes 324 trials
  against 188, and `Optimiser choice` 206 against 115. This is deliberate.
  BUTChC holds off committing to a branch early on, so that a branch that
  needs tuning before it looks good is not abandoned too soon.
- **Reliability favours BUTChC.** For a target of 50% of the achievable
  improvement, BUTChC reaches it on 27–30 of 30 seeds, where TPE manages
  10–29. For a target of 99% on `20D sphere`, BUTChC reaches it on 29 of 30
  seeds and TPE never does.

Full tables are in
[`dev/tools/results/`](https://github.com/AdventuresInDataScience/BUTChC/tree/main/dev/tools/results).

---

## What the optimizer itself costs

This is the time the optimizer spends choosing the next configuration, with
the objective replaced by a constant so that nothing else is measured:

| Optimizer | 1D | 5D | 20D |
|---|---|---|---|
| **BUTChC** (pure Python, no dependencies) | **14 µs** | **52 µs** | **193 µs** |
| TPE (this repo, pure Python) | 1037 µs | 5219 µs | 21270 µs |
| Optuna `TPESampler` (numpy-backed) | 1433 µs | 6789 µs | 27864 µs |

BUTChC is **73–145× cheaper per trial** than either TPE. Pure Python is not
the bottleneck: TPE refits its density models over the whole history on every
trial, while BUTChC samples from a small fixed-size list of good values per
parameter. The gap grows with the number of parameters, because BUTChC only
touches the parameters that are active in the current configuration.

For most real objectives, like training a model, this overhead is negligible
either way. It matters when the objective is cheap or the budget runs to
thousands of trials.

```bash
python benchmarks/overhead.py
```

---

## What batching costs

With `batch=k`, BUTChC proposes `k` configurations at once so they can be
evaluated in parallel. The trade-off is that the model is not updated between
them. The table shows the median extra regret compared with running one at a
time, over 18 problems × 20 seeds at the same total budget:

| `batch` | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| Extra regret | −3.3% | −0.1% | +2.6% | +21.3% | +43.1% |

Up to `batch=8`, the cost is within seed-to-seed noise, so an 8× wall-clock
speedup is close to free. From 16 upwards the cost is real. There is no point
setting `batch` above the number of workers you have.

```bash
python benchmarks/batch_cost.py 20 --sizes 1,2,4,8,16,32
```

---

## How conditional real search spaces are

The 18 problems above are synthetic and were written in this repository. To
check that real search spaces really are as conditional as BUTChC assumes,
[`dev/eval/pcs_stats.py`](https://github.com/AdventuresInDataScience/BUTChC/blob/main/dev/eval/pcs_stats.py)
parses eleven published configuration spaces, written by other people years
before this library existed. It reports what fraction of each space's
parameters are inactive in a typical configuration. The five most conditional:

| Space | Params | Median active | Inactive | Depth |
|---|---|---|---|---|
| AutoWEKA | 786 | 14 | **98.2%** | 4 |
| auto-sklearn (2017) | 138 | 16 | **88.4%** | 2 |
| SparrowToRiss | 222 | 67 | 69.8% | 4 |
| SATenstein | 54 | 26 | 51.9% | 4 |
| clasp 3.1.4 | 98 | 59 | 39.8% | 3 |

```bash
python dev/eval/pcs_stats.py --download
```

In AutoWEKA, 98.2% of the parameters are inactive in any given configuration.
A flat optimizer still has to model all of them, while a conditional one skips
them. All 174 of AutoWEKA's multi-parent conditions are chain-shaped, so the
space can be written as a BUTChC tree exactly.

This shows that real spaces have the structure BUTChC is built for. It is
**not** a head-to-head result, because BUTChC and TPE have not yet been run
against each other on these spaces.

---

## Caveats

The most important one: **the 18 benchmark problems are synthetic and were
written alongside the optimizer.** The held-out split reduces the risk of
overfitting to them, but does not remove it. Benchmark on your own problem
before relying on BUTChC for it. The full list of what these benchmarks do and
do not establish is in
[limitations](limitations.md#what-the-benchmarks-establish-and-what-they-do-not).
