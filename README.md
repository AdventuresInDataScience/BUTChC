# BUTChC

**B**ayesian **U**pdate **T**ree **Ch**ained **C**onditionally: a hyperparameter optimizer for search spaces where some settings only exist when others are chosen.

```bash
pip install butchc
```

Pure Python, no dependencies, Python 3.8+.

---

## Why BUTChC

Real search spaces are rarely flat. For example, in ML model selection and tuning, `momentum` only matters if `optimizer='sgd'`. `kernel` only matters if `model='svm'`. Most optimizers see every parameter all the time, so they spend trials exploring settings that have no effect.

BUTChC lets you write the search space as a tree. Each option can have its own sub-parameters, and BUTChC only samples and learns them when that option is picked. Invalid combinations cannot be sampled at all.

**Headline results** against TPE (the algorithm behind Optuna's default sampler), over 18 benchmark problems with 30 seeds each:

| | |
|---|---|
| **13 wins, 0 losses, 5 ties** | statistically significant, seed-by-seed |
| **~3× fewer trials** | to match TPE's final result, median across problems (up to 10× on the widest) |
| **4 of 6 wins on held-out problems** | problems that were never used to tune BUTChC's defaults |
| **73–145× less overhead per trial** | time the optimizer spends choosing the next config |

These problems are synthetic and were written alongside the optimizer, so benchmark on your own problem too. Full tables, methodology and caveats are in **[benchmarks](https://github.com/AdventuresInDataScience/BUTChC/blob/main/docs/benchmarks.md)**.

---

## Quick start

```python
from butchc import BUTChC_optimize

searchspace = {
    'model_type': {
        'values': ['svm', 'random_forest', 'neural_net'],
        'next_level': {                      # sub-parameters for each choice
            'svm': {
                'kernel': {'values': ['rbf', 'linear', 'poly']},
                'C':      {'min': 0.01, 'max': 100.0, 'log': True},
            },
            'random_forest': {
                'n_estimators': {'values': [50, 100, 200, 500]},
                'max_depth':    {'min': 2, 'max': 30, 'int': True},
            },
            'neural_net': {
                'learning_rate': {'min': 1e-4, 'max': 1e-1, 'log': True},
                'dropout':       {'min': 0.0, 'max': 0.5},
            },
        },
    },
    'scaler': {'values': ['standard', 'min_max', 'none']},   # always present
}

def objective(config):
    # config holds only the parameters that apply, e.g.
    #   {'model_type': 'svm', 'kernel': 'rbf', 'C': 4.2, 'scaler': 'none'}
    return cross_val_score(build_model(config), X, y).mean()

results = BUTChC_optimize(searchspace, objective, budget=150, seed=0)

print(results['best_params'])
print(results['best_value'])
```

BUTChC **maximizes** by default. For a loss or error, pass `direction='minimize'`:

```python
results = BUTChC_optimize(searchspace, validation_loss, budget=150,
                          direction='minimize')
```

---

## How it works

BUTChC builds a model with the same tree shape as your search space. Every parameter in the tree learns on its own:

- **A categorical parameter** (like `model_type` in the above example) keeps a probability for each option. All options start equal.
- **A numeric parameter** (like `C` in the above example) keeps a list of 25 good values it has seen. At the start, these are 25 evenly spaced points across the range.

Each trial then runs through the same steps:

1. **Sample a configuration.** BUTChC walks the tree from the top.
   - For a categorical parameter, it draws an option according to the current probabilities. If that option has sub-parameters, it goes into them. Every other branch is skipped for this trial.
   - For a numeric parameter, it picks one of the 25 stored values, favouring the better ones, and adds a random offset. The offset is small when the stored values are close together and larger when they are spread out. If the offset would go past `min` or `max`, the value bounces back inside the range, so the edges are not over-sampled.
   - Each parameter also has a 5% chance (`explore`) of taking a completely random value instead. This means no value is ever ruled out for good.

   Log-scale parameters do all of this on the log scale. Integer parameters are rounded before your objective sees them.
2. **Run your objective** on that configuration.
3. **Turn the result into a percentile** among all results so far. The best so far is 1.0 and a middling result is 0.5. From here on, BUTChC uses only this percentile, never the raw value. So an accuracy between 0 and 1 and a cost in the millions behave the same, and one extreme result cannot distort the model.
4. **Update the categorical parameters that were used.** Each option's score is the average percentile of the trials that picked it, with recent trials counting more. An option that hasn't been tried counts as average (0.5), so it keeps a fair chance. Options with higher scores get higher probabilities.
5. **Update the numeric parameters that were used**, but only if the trial is in the top 15% of results so far (`gamma=0.85`). The new value joins that parameter's list, and the worst value on the list is dropped. The list is then re-weighted by rank, so that its best value is picked about 400× as often as its worst.

**BUTChC starts broad and gradually commits.** Early on, the small differences between option scores barely change the probabilities, so options are sampled almost evenly. As the budget runs down, those differences are amplified more and more, and BUTChC settles on what works. Options that have their own sub-parameters settle more slowly, so a branch isn't dropped before its sub-parameters have had a chance to be tuned.

Each branch learns only from the trials that used it. For example, in the scenario presented at the start, trials using the neural net teach BUTChC about `learning_rate` and `dropout`, and never change anything under `svm`.

The [design notes](https://github.com/AdventuresInDataScience/BUTChC/blob/main/docs/design.md) explain why each step is designed this way.

---

## Defining a search space

A search space is a plain Python dict. Each key is a parameter name, and the value says what kind of parameter it is:

| Kind | Example | Your objective receives |
|---|---|---|
| Categorical | `{'values': ['relu', 'tanh', 'elu']}` | one of the listed values |
| Continuous | `{'min': 0.0, 'max': 0.5}` | a `float` in the range |
| Continuous, log scale | `{'min': 1e-5, 'max': 1e-1, 'log': True}` | a `float`, searched evenly across orders of magnitude |
| Integer | `{'min': 1, 'max': 6, 'int': True}` | an `int` in the range, inclusive |

Categorical values can be anything hashable: strings, numbers, `True`/`False`, `None`, or a mix. Use a categorical for a short list of specific numbers, like `{'values': [16, 32, 64, 128]}`. Use an integer range when every whole number in between is a valid choice.

**Use `log: True` for ranges that span several orders of magnitude**, like learning rates or regularization strengths. Without it, almost every sample from `[1e-5, 1e-1]` lands above `1e-3`, and small values are never tried.

**Use `next_level` to give a choice its own sub-parameters.** It can be nested as deep as you like:

```python
'optimizer': {
    'values': ['adam', 'sgd', 'lbfgs'],
    'next_level': {
        'adam': {'lr': {'min': 1e-4, 'max': 1e-2, 'log': True}},
        'sgd':  {'lr':       {'min': 1e-3, 'max': 1e-1, 'log': True},
                 'momentum': {'min': 0.0,  'max': 0.99}},
        # 'lbfgs' has no sub-parameters, so it is simply left out
    },
}
```

The same name (`lr` here) can appear in different branches, and each is learned separately.

This is also how to rule out invalid combinations. If `penalty='elasticnet'` only works with `solver='saga'`, make `penalty` a sub-parameter of each `solver` option, and list only the penalties that solver supports.

The [API reference](https://github.com/AdventuresInDataScience/BUTChC/blob/main/docs/api.md#search-space-format) covers the full format, including priors (starting BUTChC with a preference for certain values).

---

## Using ConfigSpace

If you already have a [ConfigSpace](https://automl.github.io/ConfigSpace/) `ConfigurationSpace`, from SMAC, auto-sklearn or an HPO benchmark such as YAHPO Gym, BUTChC has built-in converters. Install the optional extra:

```bash
pip install "butchc[configspace]"
```

**Simplest: pass it in directly.** BUTChC detects a `ConfigurationSpace` and converts it for you:

```python
results = BUTChC_optimize(configuration_space, objective, budget=200, seed=0)
```

Your objective still receives complete configs. Constant parameters are added back in, and log-scaled integers are returned as `int`.

**For more control, convert it yourself** with `from_configspace`. This lets you drop parameters you don't want searched, such as fidelity settings or task IDs in a benchmark:

```python
from butchc.interop import from_configspace, wrap_objective

space, fixed, casts = from_configspace(configuration_space, drop=['task_id'])
results = BUTChC_optimize(space, wrap_objective(objective, fixed, casts),
                          budget=200, seed=0)
```

`wrap_objective` does the same restoring of constants and integer types as the direct route.

**To go the other way**, `to_configspace(searchspace)` turns a BUTChC search space into a `ConfigurationSpace`. Use it to run Optuna or SMAC on exactly the same space for comparison.

**Some ConfigSpace features cannot be expressed as a tree:** a parameter that depends on more than one parent (including `AndConjunction` and `OrConjunction`), and forbidden clauses. Conversion stops with an `UnsupportedSpace` error naming the parameter involved, rather than quietly optimizing a different problem. Ordinal parameters are converted to plain categoricals, so their order is not used.

More in [examples](https://github.com/AdventuresInDataScience/BUTChC/blob/main/docs/examples.md#running-against-a-configspace-benchmark).

---

## Running trials in parallel

If your objective is slow, pass an executor and a `batch` size. BUTChC proposes `batch` configurations at once and evaluates them on your executor:

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=8) as pool:
    results = BUTChC_optimize(searchspace, objective, budget=200,
                              batch=8, executor=pool, seed=0)
```

Anything with a `map` or `submit` method works: `concurrent.futures`, `multiprocessing.Pool`, joblib or dask. Batches of up to 8 cost almost nothing in result quality. Larger batches start to hurt, and there is no benefit in setting `batch` above your number of workers.

---

## Saving and resuming a run

The results are plain Python data, so you can save them with `pickle`. To continue a run, pass the saved `prob_tree` back in as `start_prob_tree`. BUTChC then picks up with everything the first run learned:

```python
import pickle

results = BUTChC_optimize(searchspace, objective, budget=100, seed=0)

with open('run.pkl', 'wb') as f:
    pickle.dump(results, f)

# Later, possibly in a new session:
with open('run.pkl', 'rb') as f:
    previous = pickle.load(f)

more = BUTChC_optimize(searchspace, objective, budget=100, seed=1,
                       start_prob_tree=previous['prob_tree'])

best = max(previous, more, key=lambda r: r['best_value'])
```

A few things to know:

- **Use the same search space and `direction`** as the saved run. A tree that doesn't match the search space is rejected with an error.
- **`best_value` and `history` cover only the new run's trials.** Compare with the saved results to get the overall best, as in the last line above. When minimizing, use `min` there instead.
- **Use `pickle` rather than `json` for the tree.** JSON turns number keys into strings, so a tree with a categorical like `[16, 32, 64]` would not load back correctly.

[Examples](https://github.com/AdventuresInDataScience/BUTChC/blob/main/docs/examples.md#warm-starting-and-chaining-runs) shows how to change settings between runs and how to save the search space itself as JSON.

---

## Other features

- **Reproducible.** Setting `seed` gives the same result every time, even with parallel workers finishing in different orders.
- **Fails fast.** A mistake in the search space or arguments is reported before your objective is ever called.
- **Narrow a space.** `prune(searchspace, results)` drops branches a run has ruled out, giving a smaller space to search again or hand to another optimizer.
- **Failed trials are fine.** Return `float('nan')` from the objective and that trial is recorded but ignored.

See [examples](https://github.com/AdventuresInDataScience/BUTChC/blob/main/docs/examples.md) for each of these.

---

## Settings

The defaults are a good starting point. The settings most worth changing:

| Setting | Default | What it does |
|---|---|---|
| `budget` | required | Number of trials. A rough starting point is 10× the number of parameters. |
| `direction` | `'maximize'` | Set to `'minimize'` for losses and errors. |
| `seed` | `None` | Set it for reproducible runs. |
| `batch`, `executor` | `1`, `None` | Parallel evaluation. See above. |
| `explore` | `0.05` | Chance of trying a random value instead of a learned one. Raise it (e.g. `0.1`) if the search settles too early or the objective is noisy. |
| `verbose` | `True` | Print a line per trial. |

The finer model settings (`lambda_`, `alpha`, `temp`, `gamma`, `n_warmup`) and advice on tuning them for different kinds of problems are in the [API reference](https://github.com/AdventuresInDataScience/BUTChC/blob/main/docs/api.md#butchc_optimize).

---

## When to use it

**Good fit:** conditional search spaces, many parameters, a mix of categorical and numeric settings, objectives on awkward scales, or projects that cannot take on extra dependencies.

**Consider something else:**

- **Very small budgets (under ~50 trials) on a smooth objective with few parameters.** A Gaussian-process optimizer will usually do better.
- **Parameters that strongly interact within a branch.** BUTChC learns each parameter separately, so it cannot capture effects like "a high learning rate only works with a low momentum".

The full list is in **[limitations](https://github.com/AdventuresInDataScience/BUTChC/blob/main/docs/limitations.md)**.

---

## Documentation

- **[API reference](https://github.com/AdventuresInDataScience/BUTChC/blob/main/docs/api.md)**: every function and argument
- **[Examples](https://github.com/AdventuresInDataScience/BUTChC/blob/main/docs/examples.md)**: worked examples for common tasks
- **[Benchmarks](https://github.com/AdventuresInDataScience/BUTChC/blob/main/docs/benchmarks.md)**: full results and how to reproduce them
- **[Design notes](https://github.com/AdventuresInDataScience/BUTChC/blob/main/docs/design.md)**: why the algorithm works the way it does
- **[Limitations](https://github.com/AdventuresInDataScience/BUTChC/blob/main/docs/limitations.md)**: what it can't do and what hasn't been tested
- **[Changelog](https://github.com/AdventuresInDataScience/BUTChC/blob/main/CHANGELOG.md)**

Defaults can change between minor versions, so **a seeded run may give different results after an upgrade**.

## Development

```bash
git clone https://github.com/AdventuresInDataScience/BUTChC
cd BUTChC
pip install -e ".[test]"
pytest
```

The benchmark scripts are in `benchmarks/`, and a map of the source code is in [docs/codemap.md](https://github.com/AdventuresInDataScience/BUTChC/blob/main/docs/codemap.md).

## License

MIT. See [LICENSE](https://github.com/AdventuresInDataScience/BUTChC/blob/main/LICENSE).
