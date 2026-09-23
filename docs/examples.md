# Examples

Task-shaped recipes. For the full parameter list see [api.md](api.md).

- [Tuning a scikit-learn pipeline](#tuning-a-scikit-learn-pipeline)
- [Nesting more than one level](#nesting-more-than-one-level)
- [Minimizing](#minimizing)
- [Passing data to the objective](#passing-data-to-the-objective)
- [Parallel evaluation](#parallel-evaluation)
- [Stating what you already believe](#stating-what-you-already-believe)
- [Warm starting and chaining runs](#warm-starting-and-chaining-runs)
- [Monitoring convergence](#monitoring-convergence)
- [Reading the learned tree](#reading-the-learned-tree)
- [Pruning a space and handing it off](#pruning-a-space-and-handing-it-off)
- [Saving a search space as JSON](#saving-a-search-space-as-json)
- [Running against a ConfigSpace benchmark](#running-against-a-configspace-benchmark)
- [Comparing against Optuna or SMAC](#comparing-against-optuna-or-smac)
- [Handling a noisy objective](#handling-a-noisy-objective)
- [Objectives that can fail](#objectives-that-can-fail)

---

## Tuning a scikit-learn pipeline

The shape BUTChC is built for: a choice of model, then that model's own
hyperparameters. No trial is spent on an `svm` config carrying a `dropout`.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from butchc import BUTChC_optimize

searchspace = {
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
    'preprocessing': {'values': ['standard_scaler', 'min_max', 'none']},
}

SCALERS = {'standard_scaler': StandardScaler(), 'min_max': MinMaxScaler(),
           'none': None}


def build(config):
    if config['model_type'] == 'svm':
        model = SVC(kernel=config['kernel'], C=config['C'],
                    gamma=config['gamma'])
    elif config['model_type'] == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            max_features=config['max_features'],
        )
    else:
        model = MLPClassifier(
            learning_rate_init=config['learning_rate'],
            hidden_layer_sizes=(config['hidden_units'],),
        )

    scaler = SCALERS[config['preprocessing']]
    return make_pipeline(scaler, model) if scaler else model


def objective(config, X, y):
    return cross_val_score(build(config), X, y, cv=5).mean()


results = BUTChC_optimize(searchspace, objective, budget=150, seed=0,
                          X=X_train, y=y_train)

print(results['best_params'])
print(f"CV accuracy: {results['best_value']:.4f}")
```

---

## Nesting more than one level

`next_level` composes. A sub-searchspace is just a searchspace, so any
categorical inside one can carry its own `next_level`, to any depth. This is
the only nesting mechanism — a sub-space hung directly off a choice name is
rejected at validation (see
[api.md](api.md#there-is-no-implicit-nesting)).

Four levels, and each level earns its place:

```python
searchspace = {
    'family': {
        'values': ['linear', 'tree', 'net'],
        'next_level': {
            'linear': {
                'C': {'min': 1e-3, 'max': 1e3, 'log': True},
                'solver': {
                    'values': ['liblinear', 'saga'],
                    'next_level': {
                        # liblinear cannot do elasticnet; saga can
                        'liblinear': {'penalty': {'values': ['l1', 'l2']}},
                        'saga': {
                            'penalty': {
                                'values': ['l1', 'l2', 'elasticnet'],
                                'next_level': {
                                    # only meaningful for elasticnet
                                    'elasticnet': {
                                        'l1_ratio': {'min': 0.0, 'max': 1.0},
                                    },
                                },
                            },
                        },
                    },
                },
            },
            'tree': {
                'boost': {
                    'values': ['gbm', 'rf'],
                    'next_level': {
                        'gbm': {'shrink': {'min': 1e-3, 'max': 0.3, 'log': True},
                                'leaves': {'min': 2, 'max': 64, 'int': True}},
                        'rf':  {'trees': {'min': 50, 'max': 500, 'int': True}},
                    },
                },
            },
            'net': {'width': {'values': [64, 128, 256]},
                    'lr': {'min': 1e-5, 'max': 1e-2, 'log': True}},
        },
    },
    'scaler': {'values': ['standard', 'robust', 'none']},   # always present
}
```

Every configuration carries only the parameters on its own path:

```python
{'family': 'net',    'width': 64, 'lr': 7.3e-05, 'scaler': 'standard'}
{'family': 'tree',   'boost': 'gbm', 'leaves': 64, 'shrink': 0.018, 'scaler': 'none'}
{'family': 'tree',   'boost': 'rf', 'trees': 283, 'scaler': 'none'}
{'family': 'linear', 'solver': 'saga', 'penalty': 'l2', 'C': 0.0157, 'scaler': 'robust'}
{'family': 'linear', 'solver': 'saga', 'penalty': 'elasticnet',
                     'l1_ratio': 0.142, 'C': 18.8, 'scaler': 'standard'}
```

### Nesting is how you express an invalid combination

The `solver`/`penalty` levels above are not decoration. `liblinear` does not
support an elasticnet penalty, so the pairing is illegal — and because
`penalty` lives *under* `solver`, the illegal pairing cannot be sampled. Over
200 trials on the space above, the count of `liblinear` + `elasticnet`
configurations is zero, and `l1_ratio` never appears without `elasticnet`.

That is the difference between a conditional optimiser and a flat one with a
penalty term. A flat optimiser must sample the invalid region, score it badly,
and learn to avoid it, spending budget on all three steps. Here the region does
not exist.

The technique generalises: **to forbid a combination, make one parameter the
parent of the other and give each branch only the values that are legal under
it.** Constraints that cannot be arranged that way — a joint budget over two
independent continuous parameters, say — are the genuine exception, and are
covered in [limitations](limitations.md#search-spaces-butchc-cannot-express).

### When to nest and when not to

Nest when a parameter is **meaningless** under some choices — `l1_ratio`
without elasticnet, `momentum` without sgd. Nesting is what tells the optimiser
the parameter does not apply.

Do not nest merely to express that a parameter *interacts* with another. Nesting
requires the parent to be categorical and gives the child a separate model per
branch, which costs you the shared evidence. If two continuous parameters simply
depend on each other, leave them as siblings and accept that the marginals will
not capture it — see
[limitations](limitations.md#where-the-model-runs-out).

One consequence worth knowing: a name may repeat across branches that cannot
co-occur, and each occurrence is an independent node. `lr` under `net` and `lr`
under some other family are two models, learning from their own trials only.
The [uniqueness rule](api.md#names-must-be-unique-across-the-tree) says exactly
when a repeat is legal.

---

## Minimizing

BUTChC maximizes by default. Pass `direction='minimize'` for a loss, an error,
or anything where lower is better.

```python
def rosenbrock(config):
    x, y = config['x'], config['y']
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

results = BUTChC_optimize(
    {'x': {'min': -2.0, 'max': 2.0}, 'y': {'min': -1.0, 'max': 3.0}},
    rosenbrock, budget=500, seed=0, verbose=False,
    direction='minimize',
)
print(f"Minimum found: {results['best_value']:.4f}")
```

---

## Passing data to the objective

Any keyword argument `BUTChC_optimize` does not recognise is forwarded to every
objective call. Extra arguments **must** be keyword.

```python
def objective(config, X_train, y_train, X_val, y_val):
    model = build(config)
    model.fit(X_train, y_train)
    return model.score(X_val, y_val)

results = BUTChC_optimize(
    searchspace, objective, budget=100, seed=0,
    X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
)
```

This is also how you keep an objective picklable for a process pool — data
travels as arguments rather than being captured in a closure.

---

## Parallel evaluation

Worth it when the objective is slow enough that the wall clock dominates.
BUTChC does not create the pool; you pass one in.

### Threads — the usual right answer

An objective that spends its time inside NumPy, scikit-learn, PyTorch, or a
subprocess releases the GIL and parallelises fine on threads. Nothing is
pickled, so closures and interactively defined functions work.

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=8) as pool:
    results = BUTChC_optimize(
        searchspace, objective, budget=200,
        batch=8, executor=pool, seed=0,
        X=X_train, y=y_train,
    )
```

### Processes — for pure-Python objectives

Only when the objective is genuinely GIL-bound. A process pool transmits the
objective by pickle, so it must be defined at module level, and its data must
travel via `**kwargs` rather than a closure.

```python
from concurrent.futures import ProcessPoolExecutor

def objective(config, data):        # module level, not nested
    return score(config, data)

if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=8) as pool:
        results = BUTChC_optimize(
            searchspace, objective, budget=200,
            batch=-1,               # one config per CPU
            executor=pool, seed=0,
            data=data,
        )
```

If pickling fails, BUTChC raises a `RuntimeError` naming the cause rather than
letting a bare `AttributeError` surface from deep in `multiprocessing`.

### Choosing `batch`

Match it to the workers you actually have. Measured cost at fixed budget over
18 problems and 20 seeds: `k=8` costs about 2.6% in median regret, `k=16` about
21%, `k=32` about 43%. Below `k=8` the cost is inside seed noise.

```bash
python benchmarks/batch_cost.py 20 --sizes 1,2,4,8,16,32
```

Two things that make batching *worse* than it needs to be:

- **`batch` larger than your worker count.** You pay staleness for evaluations
  that then queue anyway.
- **Oversubscription.** `batch=-1` gives one config per CPU, but if the
  objective already uses every core — `n_jobs=-1`, a threaded BLAS, a GPU —
  you now have `n_cpu²` threads fighting. Set `n_jobs=1` inside the objective,
  or pick a `batch` well below the core count.

A seeded run gives the same answer inline, on threads, and on processes.
Updates are applied in sample order, never completion order.

---

## Stating what you already believe

```python
searchspace = {
    'optimizer': {
        'values': ['adam', 'sgd', 'lbfgs'],
        'prior': {'adam': 0.7, 'sgd': 0.25, 'lbfgs': 0.05},
        'prior_strength': 30,        # worth 30 trials of evidence
    },
    'learning_rate': {
        'min': 1e-5, 'max': 1e-1, 'log': True,
        'prior': {'mean': 1e-3, 'std': 5e-4},
        'prior_strength': 15,        # the most a continuous prior can hold
    },
}
```

A prior is a starting point, not a constraint — `prior_strength` sets how much
contrary evidence it takes to overturn. Continuous priors always leave at least
10 grid points spanning the full range, so no strength can make part of the
range unreachable; with the default reservoir of 25 that caps a continuous
`prior_strength` at 15 (see [priors](api.md#priors)).

---

## Warm starting and chaining runs

Continue where a previous run stopped, optionally with different settings.

```python
first = BUTChC_optimize(searchspace, objective, budget=100, seed=0,
                        verbose=False)

second = BUTChC_optimize(
    searchspace, objective, budget=100, seed=1, verbose=False,
    lambda_=0.5,                          # finer refinement
    start_prob_tree=first['prob_tree'],
)
```

The tree is validated against `searchspace` and deep-copied, never mutated.

Warm starting does not change how `n_warmup` is treated — there is no
detection of "this run already knows something". The default is 0, so a warm
start goes straight to modelling unless you ask for warm-up explicitly, and if
you pass `n_warmup=20` alongside a `start_prob_tree` you get twenty uniform
trials against the tree you just loaded.

To survive a crash or a scheduler timeout, checkpoint the tree with `pickle`:

```python
import pickle

with open('tree.pkl', 'wb') as fh:
    pickle.dump(results['prob_tree'], fh)

with open('tree.pkl', 'rb') as fh:
    tree = pickle.load(fh)
```

Don't use `json` for the tree. JSON object keys are always strings, so a
categorical over `[16, 32, 64]` comes back keyed by `"16"`, and the reloaded
tree is rejected as not matching the search space. (`to_json`/`from_json`
handle this for search spaces, but not for trees.)

---

## Monitoring convergence

```python
results = BUTChC_optimize(searchspace, objective, budget=500, seed=0,
                          verbose=False)

print("Final rolling loss:", results['rolling_loss'][-1])
print("Trials that moved the tree:", results['n_updates'], "of 500")
print("Trials that cleared the gate:", results['n_gated'], "of 500")

THRESHOLD = 1e-4
for i, rl in enumerate(results['rolling_loss']):
    if i > 50 and rl < THRESHOLD:
        print(f"Settled around trial {i + 1}")
        break
```

`loss` measures how far the tree moved, not how good the configuration was.
Read `n_updates` and `n_gated` together: a falling rolling loss means either
the distribution has settled, or few trials are clearing the gate. Those call
for opposite responses — stop early, versus lower `gamma`.

Plotting the actual search progress is a different series:

```python
best_so_far, best = [], -float('inf')
for h in results['history']:
    best = max(best, h['objective'])
    best_so_far.append(best)
```

---

## Reading the learned tree

```python
from butchc import reservoir_summary

tree = results['prob_tree']

print(tree['optimizer']['prob'])
# {'adam': 0.975, 'sgd': 0.016, 'lbfgs': 0.009}

print(reservoir_summary(tree['optimizer']['next_level']['adam']['lr']))
# {'mean': 0.00106, 'std': 0.00087, 'ess': 21.7, 'n': 25}
```

`mean` and `std` come back in the units you defined, so a `log` parameter
reports on its original scale. An `ess` far below `n` means the distribution
has collapsed onto a few points — informative if you expected it, a warning if
you did not.

---

## Pruning a space and handing it off

BUTChC's clearest margins on conditional problems come from deciding *which
branch* — `Branch trap` and `Categorical mix` — while on `Nested pipeline`,
where a branch must be tuned well once chosen, it only ties TPE. `prune` lets
you use BUTChC to discard branches and then hand the survivors to a second run,
or to another optimiser.

Spend part of the budget narrowing the space, then search what survives:

```python
from butchc import BUTChC_optimize, prune, prune_report

scout = BUTChC_optimize(searchspace, objective, budget=200, seed=0)

smaller, tree = prune(searchspace, scout, return_tree=True)
for line in prune_report(searchspace, smaller):
    print(line)
# model: dropped ['naive_bayes', 'knn'], kept ['xgboost', 'random_forest']

final = BUTChC_optimize(smaller, objective, budget=200, seed=1,
                        start_prob_tree=tree)
```

`return_tree=True` is what makes `start_prob_tree` usable here. `scout['prob_tree']`
itself still describes the *original*, unpruned space — every dropped choice and
the original, wider continuous bounds are still in it — so `BUTChC_optimize`
rejects it against the smaller space with "choices differ from searchspace" or
"bounds differ from the searchspace". The tree `prune` hands back instead keeps
the learned statistics for only the choices and archive points that survived,
so the second run resumes rather than restarting. It also drops any `prior` on
a node it touched, because a prior over choices that no longer exist does not
sum to anything meaningful — the carried-over tree is the better warm start.

### Handing off to Optuna

The output is an ordinary search space, so `butchc.interop` can convert it to
whatever the next optimiser speaks:

```python
import optuna
from butchc import BUTChC_optimize, prune

scout = BUTChC_optimize(searchspace, objective, budget=150, seed=0)
smaller = prune(searchspace, scout, narrow=2.0)

def optuna_objective(trial):
    config = {'model': trial.suggest_categorical(
        'model', smaller['model']['values'])}
    branch = smaller['model']['next_level'][config['model']]
    for name, spec in branch.items():
        config[name] = trial.suggest_float(name, spec['min'], spec['max'],
                                           log=spec.get('log', False))
    return objective(config)

study = optuna.create_study(direction='maximize')
study.optimize(optuna_objective, n_trials=150)
```

This sketch assumes every branch parameter is continuous.
`butchc.interop.to_configspace(smaller)` gives the same reduced space as a
`ConfigurationSpace` for SMAC or anything else that consumes one.

### Narrowing continuous bounds too

`narrow=k` also tightens continuous ranges to `mean ± k * std` of the learned
distribution, clipped to the original bounds:

```python
smaller = prune(searchspace, scout, narrow=2.0)
# z: narrowed [0, 10] -> [6.73624, 7.10541]
# n: narrowed [1, 200] -> [48, 165]
```

This is off by default and deserves more suspicion than branch pruning. A
dropped branch is visible in `prune_report`; a bound narrowed past the optimum
looks exactly like a bound that was always there, and the second run cannot
tell you it happened. Use it when the first run was long enough that the
archive means something, and prefer `k = 3` over `k = 2` if the objective is
noisy.

### What it will not do

`prune` protects the branch that produced `best_params` regardless of its
probability, keeps at least two choices per node, and requires a probability
at or below a third of the node's uniform share before dropping anything. Those
defaults exist because a branch can have a poor average and an excellent best —
that is precisely what the `Branch trap` benchmark is built from, and it is the
shape careless pruning destroys.

The threshold is a fraction of uniform, not a bare probability, so it means the
same thing on a three-way choice as on a thirty-way one: uniform is `1/n`, and
the default drops anything at or below `1/(3n)`.

```python
prune(searchspace, scout, threshold=0.99, keep_min=1)
# still keeps the branch holding the best result

prune(searchspace, scout, threshold=0.99, keep_min=1, protect_best=False)
# probability alone decides — the risky mode, and it says so
```

Pruning early in a run undoes all of it. Probabilities move a great deal in the
first trials, so give the scout run enough budget that you would have trusted
its answer on its own.

Budget has to scale with the number of branches, and this is the failure mode
worth knowing about, because it is silent. Measured on a synthetic space with
one good branch, one adequate one and the rest useless:

| Branches | Scout budget | Trials per branch | Dropped |
|---|---|---|---|
| 2 | 200 | 100 | none — `keep_min` floor |
| 4 | 200 | 50 | the two useless ones |
| 12 | 200 | 17 | **none** |

At twelve branches and 200 trials the useless options still sit near uniform,
so nothing clears the threshold and `prune` returns the space unchanged. That
is the correct outcome — there was no evidence to prune on — but it is also the
case where pruning would have been worth most. If `prune_report` comes back
empty on a wide node, the scout run was too short, not the space too good.

---

## Saving a search space as JSON

The dict format is plain data, so a space can live in version control next to
the code that uses it.

```python
from butchc.interop import to_json, from_json

with open('space.json', 'w') as fh:
    fh.write(to_json(searchspace, indent=2))

with open('space.json') as fh:
    searchspace = from_json(fh.read())
```

Use `from_json` rather than `json.loads`. JSON object keys are always strings,
so a space branching on integer values comes back with its `next_level` keyed
by `"1"` instead of `1` — the branch would be silently unreachable and every
config would arrive missing its sub-parameters. `from_json` repairs that.

---

## Running against a ConfigSpace benchmark

Every standard HPO benchmark — YAHPO Gym, HPOBench, HPO-B, SMAC's suites —
hands you a `ConfigurationSpace`. Pass it straight in.

```bash
pip install butchc[configspace]
```

```python
results = BUTChC_optimize(configuration_space, objective, budget=200, seed=0)
```

Constants are restored and log-scaled integers are cast back to `int` before
your objective sees them. For control over what is searched, convert
explicitly:

```python
from butchc.interop import from_configspace, wrap_objective

space, fixed, casts = from_configspace(
    configuration_space,
    drop=['trainsize', 'repl', 'task_id'],   # fidelity knobs and task selectors
)
results = BUTChC_optimize(space, wrap_objective(objective, fixed, casts),
                          budget=200, seed=0)
```

Conversion refuses what a tree cannot hold — conjunctions, multiple parents,
forbidden clauses — with an `UnsupportedSpace` naming the parameter, rather
than approximating and silently optimising a different problem. Ordinals become
unordered categoricals, which loses neighbourhood structure and is worth
remembering when reading results.

---

## Comparing against Optuna or SMAC

Convert in the other direction, so every method sees the same space.

```python
from butchc.interop import to_configspace

cs_space, names = to_configspace(searchspace)
```

`names` maps each ConfigSpace name back to its BUTChC name. It is the identity
except where a name repeated across sibling branches — BUTChC permits that,
ConfigSpace does not, so collisions are qualified as `optimizer:sgd:lr`.

`benchmarks/evaluate.py` runs the paired comparison end to end, with medians,
win-loss records and sign tests:

```bash
python benchmarks/evaluate.py 30 --methods random,tpe,optuna,butchc
```

---

## Handling a noisy objective

Raise `gamma` above its default of `0.85` so only strongly ranked trials reach
the continuous archive, consider `explore=0.1` (what the `noisy` regime sweep
selects — see [tuning by problem shape](api.md#tuning-by-problem-shape)), and
score the reported best on a clean evaluation rather than trusting the noisy
one that won.

```python
results = BUTChC_optimize(searchspace, noisy_objective, budget=400,
                          gamma=0.9, explore=0.1, seed=0, verbose=False)

true_value = clean_objective(results['best_params'])
```

Because updates use rank rather than raw value, an objective in the millions
behaves like one in `[0, 1]` and a single catastrophic outlier cannot dominate
the archive permanently.

---

## Objectives that can fail

Return a non-finite value. It is recorded in `history` but excluded from
ranking and from best-tracking, so a failed config neither wins nor poisons the
model.

```python
def objective(config):
    try:
        return train_and_score(config)
    except (ValueError, MemoryError):
        return float('nan')      # or float('-inf')
```

Prefer `nan` over a large negative sentinel. A sentinel is a finite value, so it
enters the rank distribution and shifts every other trial's rank.

If *every* trial is non-finite, `best_params` is `None` and `best_value` is
`-inf` rather than an exception — check before indexing.
