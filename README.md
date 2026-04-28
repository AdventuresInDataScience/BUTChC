# BUTChC

**B**ayesian **U**pdate **T**ree **Ch**ained **C**onditionally — a dependency-free, probabilistic black-box hyperparameter optimizer.

BUTChC maintains a probability distribution over a **hierarchical, conditional search space** and iteratively refines it based on observed objective values. The key idea: parameters can be conditional on the choices made higher up the tree — `momentum` only exists when `optimizer=sgd`; `kernel_size` only exists when `model=cnn`. This means the optimizer never wastes trials sampling irrelevant combinations, and learns each branch's parameters independently.

It requires no gradients, no differentiability, and makes no assumptions about the objective function's internals.

---

## Features

- **Zero dependencies** — pure Python standard library, nothing to install beyond the package itself
- **Hierarchical search spaces** — parameters can be conditional on parent choices (`optimizer=adam` unlocks `beta1`; `optimizer=sgd` unlocks `momentum`)
- **Non-parametric continuous modelling** — weighted KDE reservoir, no distributional assumptions, naturally handles multimodality
- **Warm-startable** — chain runs, resume interrupted optimizations, or seed from prior knowledge
- **Built-in convergence signal** — per-trial loss metric tracks model surprise; use it for early stopping
- **Interpretable** — the probability tree is human-readable after training

---

## Installation

**From source (recommended):**
```bash
git clone https://github.com/AdventuresInDataScience/BUTChC.git
cd BUTChC
pip install .
```

**Editable install for development:**
```bash
pip install -e .
```

**Requirements:** Python ≥ 3.8, no external dependencies.

**Run the test suite:**
```bash
pip install pytest
pytest
```

---

## Quick Start

Three model types, each with their own set of hyperparameters. BUTChC only ever samples and updates parameters that are relevant to the model chosen in a given trial.

```python
from butchc import BUTChC_optimize

searchspace = {
    # Top-level choice: which model family to use.
    # Each choice unlocks its own set of sub-parameters via 'next_level'.
    'model_type': {
        'values': ['svm', 'random_forest', 'neural_net'],
        'next_level': {
            'svm': {
                # These params are ONLY sampled when model_type == 'svm'
                'kernel': {'values': ['rbf', 'linear', 'poly']},
                'C':      {'min': 0.01, 'max': 100.0},
                'gamma':  {'min': 1e-4, 'max': 10.0},
            },
            'random_forest': {
                # These params are ONLY sampled when model_type == 'random_forest'
                'n_estimators': {'values': [50, 100, 200, 500]},
                'max_depth':    {'min': 2.0, 'max': 30.0},
                'max_features': {'values': ['sqrt', 'log2']},
            },
            'neural_net': {
                # These params are ONLY sampled when model_type == 'neural_net'
                'learning_rate': {'min': 1e-4, 'max': 1e-1},
                'hidden_units':  {'values': [64, 128, 256, 512]},
                'dropout':       {'min': 0.0, 'max': 0.5},
            },
        },
    },
    # This param is always present, regardless of model_type
    'preprocessing': {'values': ['standard_scaler', 'min_max', 'none']},
}

def objective(config):
    # config always contains 'model_type' and 'preprocessing',
    # PLUS only the params for whichever model was chosen. For example:
    #   {'model_type': 'svm', 'kernel': 'rbf', 'C': 4.2, 'gamma': 0.01, 'preprocessing': 'standard_scaler'}
    #   {'model_type': 'neural_net', 'learning_rate': 0.003, 'hidden_units': 128, 'dropout': 0.2, 'preprocessing': 'none'}
    # 'momentum' will never appear here; 'C' will never appear in a neural_net config.
    model = build_model(config)
    return model.cross_val_score(X, y)   # higher = better

results = BUTChC_optimize(
    searchspace = searchspace,
    objective   = objective,
    budget      = 150,
    lambda_     = 1.0,
    alpha       = 1.0,
    verbose     = True,
)

print(results['best_params'])
# e.g. {'model_type': 'neural_net', 'learning_rate': 0.0031,
#        'hidden_units': 256, 'dropout': 0.15, 'preprocessing': 'standard_scaler'}
print(f"Best score: {results['best_value']:.4f}")
```

---

## Examples

### 1. Finding the peak of a mathematical function

```python
from butchc import BUTChC_optimize

# Maximize -(x-2)^2 - (y+1)^2; global peak at (x=2, y=-1)
def objective(config):
    x, y = config['x'], config['y']
    return -((x - 2.0) ** 2) - ((y + 1.0) ** 2)

searchspace = {
    'x': {'min': -5.0, 'max': 5.0},
    'y': {'min': -5.0, 'max': 5.0},
}

results = BUTChC_optimize(
    searchspace, objective,
    budget=200, lambda_=1.5, alpha=1.0,
    verbose=False,
)

print(f"x = {results['best_params']['x']:.3f}  (target: 2.0)")
print(f"y = {results['best_params']['y']:.3f}  (target: -1.0)")
print(f"Best value: {results['best_value']:.4f}  (target: 0.0)")
```

---

### 2. Neural network optimizer tuning with conditional sub-parameters

`adam` and `sgd` have completely different tuning knobs. With `next_level`, BUTChC learns each optimizer's best parameters from only the trials that used that optimizer — no cross-contamination.

```python
from butchc import BUTChC_optimize

searchspace = {
    'optimizer': {
        'values': ['adam', 'sgd'],
        'next_level': {
            'adam': {
                # Adam-specific: lr range tuned for Adam, plus its beta1 moment
                'lr':    {'min': 1e-4, 'max': 1e-1},
                'beta1': {'values': [0.85, 0.90, 0.95, 0.99]},
            },
            'sgd': {
                # SGD-specific: different lr range + momentum (meaningless for Adam)
                'lr':       {'min': 1e-3, 'max': 1e-1},
                'momentum': {'min': 0.0, 'max': 0.99},
            },
        },
    },
    # These appear in every config regardless of optimizer
    'batch_size': {'values': [16, 32, 64, 128]},
    'dropout':    {'min': 0.0, 'max': 0.5},
}

def train_and_evaluate(config):
    # Possible config shapes:
    #   {'optimizer': 'adam',  'lr': 0.001, 'beta1': 0.95, 'batch_size': 64, 'dropout': 0.2}
    #   {'optimizer': 'sgd',   'lr': 0.01,  'momentum': 0.9, 'batch_size': 32, 'dropout': 0.1}
    # 'momentum' will NEVER appear in an adam config; 'beta1' will NEVER appear in an sgd config.
    model = build_and_train(config)
    return evaluate(model)   # return validation accuracy (higher = better)

results = BUTChC_optimize(
    searchspace, train_and_evaluate,
    budget=200, lambda_=1.0, alpha=1.0, temp=1.2,
    verbose=True,
)
print(results['best_params'])


---

### 3. Minimization (negate the objective)

BUTChC always maximizes. To minimize, negate:

```python
from butchc import BUTChC_optimize

# Minimize the Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
# True minimum is 0 at (x=1, y=1).
def rosenbrock(config):
    x, y = config['x'], config['y']
    return -((1 - x) ** 2 + 100 * (y - x ** 2) ** 2)  # negate to maximize

searchspace = {
    'x': {'min': -2.0, 'max': 2.0},
    'y': {'min': -1.0, 'max': 3.0},
}

results = BUTChC_optimize(
    searchspace, rosenbrock,
    budget=500, lambda_=2.0, alpha=1.0,
    verbose=False,
)

print(f"x = {results['best_params']['x']:.3f}  (target: 1.0)")
print(f"y = {results['best_params']['y']:.3f}  (target: 1.0)")
print(f"Minimum found: {-results['best_value']:.4f}  (target: 0.0)")
```

---

### 4. Warm starting — chaining runs

Pass the `prob_tree` from one run as the `start_prob_tree` of the next. The second run picks up exactly where the first left off.

```python
from butchc import BUTChC_optimize

searchspace = {'x': {'min': -5.0, 'max': 5.0}, 'method': {'values': ['a', 'b']}}

def objective(config):
    return -(config['x'] ** 2)

# First run: explore broadly
results1 = BUTChC_optimize(
    searchspace, objective,
    budget=100, lambda_=1.0, alpha=1.0,
    verbose=False,
)

# Second run: continue from the learned distribution
results2 = BUTChC_optimize(
    searchspace, objective,
    budget=100, lambda_=0.5, alpha=1.0,   # lower lambda_ for finer refinement
    start_prob_tree=results1['prob_tree'],
    verbose=False,
)

print(f"After 200 total trials: best x = {results2['best_params']['x']:.3f}")
```

---

### 5. Passing extra arguments to the objective

Keyword arguments not consumed by `BUTChC_optimize` are forwarded to your objective.

```python
from butchc import BUTChC_optimize

def objective(config, X_train, y_train, X_val, y_val):
    model = build_model(config)
    model.fit(X_train, y_train)
    return model.score(X_val, y_val)

results = BUTChC_optimize(
    searchspace, objective,
    budget=100, lambda_=1.0, alpha=1.0,
    verbose=False,
    # These are forwarded directly to objective():
    X_train=X_train, y_train=y_train,
    X_val=X_val,     y_val=y_val,
)
```

---

### 6. Monitoring convergence

The `loss_history` and `rolling_loss` fields track how much the model updates each trial. Declining rolling loss = the distribution is settling.

```python
from butchc import BUTChC_optimize

results = BUTChC_optimize(
    searchspace, objective,
    budget=500, lambda_=1.0, alpha=1.0,
    verbose=False,
)

# Inspect raw losses
print("Final 10 losses:", results['loss_history'][-10:])
print("Final rolling loss:", results['rolling_loss'][-1])

# Early-stopping heuristic: stop when rolling loss < threshold
THRESHOLD = 1e-4
for i, rl in enumerate(results['rolling_loss']):
    if rl < THRESHOLD:
        print(f"Converged at trial {i + 1}")
        break

# Plot with matplotlib (optional)
try:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(results['loss_history'],  alpha=0.4, label='Trial loss')
    ax1.plot(results['rolling_loss'],  label='Rolling mean')
    ax1.set_xlabel('Trial'); ax1.set_ylabel('Loss'); ax1.legend()
    ax1.set_title('Convergence')

    running_best, best_curve = -float('inf'), []
    for h in results['history']:
        running_best = max(running_best, h['objective'])
        best_curve.append(running_best)
    ax2.plot(best_curve)
    ax2.set_xlabel('Trial'); ax2.set_ylabel('Best objective')
    ax2.set_title('Optimization progress')

    plt.tight_layout(); plt.show()
except ImportError:
    pass  # matplotlib is optional
```

---

### 7. Reading the learned probability tree

After optimization, you can inspect what the model learned:

```python
results = BUTChC_optimize(...)
tree = results['prob_tree']

# Categorical probabilities
print("Optimizer probabilities:", tree['optimizer']['prob'])
# e.g. {'adam': 0.73, 'sgd': 0.27}

# The model has learned adam is better 73% of the time

# Continuous: look at weighted mean of the reservoir
opt_subtree = tree['optimizer']['next_level']['adam']
lr_node     = opt_subtree['lr']
weighted_mean = sum(w * x for w, x in zip(lr_node['weights'], lr_node['reservoir']))
print(f"Learned centre of lr distribution: {weighted_mean:.5f}")
```

---

## API Reference

### `BUTChC_optimize`

```python
from butchc import BUTChC_optimize

results = BUTChC_optimize(
    searchspace,
    objective,
    budget,
    lambda_,
    alpha,
    temp            = 1.0,
    start_prob_tree = None,
    verbose         = True,
    **kwargs,        # forwarded to objective
)
```

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `searchspace` | `dict` | — | Search space definition (see below) |
| `objective` | `callable` | — | `(config, **kwargs) → float`. Higher return value = better. |
| `budget` | `int` | — | Number of objective evaluations |
| `lambda_` | `float` | — | Initial update weight. Higher = faster adaptation, more risk of premature convergence. Start with `1.0`. |
| `alpha` | `float` | — | Laplace smoothing for categorical nodes. Higher = distribution stays flat longer. Start with `1.0`. |
| `temp` | `float` | `1.0` | Softmax temperature for categorical sampling. `>1` explores more, `<1` exploits more. |
| `start_prob_tree` | `dict \| None` | `None` | Warm-start from a previous run's `prob_tree`. |
| `verbose` | `bool` | `True` | Print per-trial progress to stdout. |
| `**kwargs` | | | Forwarded to every call of `objective`. |

#### Return value

```python
{
    'best_params':  dict,    # Config that achieved the highest objective
    'best_value':   float,   # Corresponding objective value
    'prob_tree':    dict,    # Final probability tree — pass to start_prob_tree to warm-start
    'history':      list,    # Per-trial: {'params': ..., 'objective': ..., 'loss': ..., 'rolling_loss': ...}
    'loss_history': list,    # Per-trial loss values (convenience copy)
    'rolling_loss': list,    # Per-trial rolling mean loss (convenience copy)
}
```

---

## Defining the Search Space

### The two node types

Every parameter in the search space is one of two forms:

**Categorical** — pick one value from a fixed list:
```python
'activation': {'values': ['relu', 'tanh', 'elu']}
```

**Continuous** — sample a float from `[min, max]` using a non-parametric KDE:
```python
'learning_rate': {'min': 1e-5, 'max': 1e-1}
```

For integer-valued parameters, use continuous and round in your objective:
```python
'n_layers': {'min': 1.0, 'max': 6.0}
# In your objective: n = round(config['n_layers'])
```

---

### Conditional parameters with `next_level`

This is the core feature of BUTChC. A categorical node can carry a `next_level` dict that maps each choice to a sub-searchspace. Parameters defined inside `next_level` are **only ever sampled and updated** when their parent value was chosen.

```python
'optimizer': {
    'values': ['adam', 'sgd'],
    'next_level': {
        'adam': {
            # Only active when optimizer == 'adam'
            'lr':    {'min': 1e-4, 'max': 1e-2},
            'beta1': {'values': [0.9, 0.95, 0.99]},
        },
        'sgd': {
            # Only active when optimizer == 'sgd'
            'lr':       {'min': 1e-3, 'max': 1e-1},
            'momentum': {'min': 0.0, 'max': 0.99},
        },
    },
}
```

When `optimizer=adam` is sampled, the returned config contains `lr` and `beta1` but **never** `momentum`. When `optimizer=sgd` is sampled, it contains `lr` and `momentum` but **never** `beta1`. The probability model for each branch is updated entirely independently — learning the best `lr` for adam does not interfere with learning the best `lr` for sgd, even though they share the same parameter name.

**Why this matters:** Without conditional parameters, you'd have to put `momentum` in the flat space and hope the optimizer figures out to ignore it for adam. With `next_level`, the structure is explicit, no budget is wasted on invalid combinations, and the model learns each branch's parameters from only the trials that actually used that branch.

---

### Parameters without `next_level` are always active

Any parameter defined at the root level (outside any `next_level`) appears in **every** config, regardless of what was chosen elsewhere:

```python
searchspace = {
    'optimizer': {
        'values': ['adam', 'sgd'],
        'next_level': { ... },
    },
    'batch_size': {'values': [16, 32, 64]},   # always present
    'dropout':    {'min': 0.0, 'max': 0.5},   # always present
}
# Every config will have 'optimizer', 'batch_size', 'dropout',
# plus whichever sub-params belong to the chosen optimizer.
```

---

### `next_level` doesn't have to cover every choice

If a choice has no sub-parameters, just omit it from `next_level`. You can also omit `next_level` entirely for a plain categorical node:

```python
'optimizer': {
    'values': ['adam', 'sgd', 'lbfgs'],
    'next_level': {
        'adam': {'lr': {'min': 1e-4, 'max': 1e-1}},
        'sgd':  {'lr': {'min': 1e-3, 'max': 1e-1}, 'momentum': {'min': 0.0, 'max': 0.99}},
        # 'lbfgs' is not listed — it has no sub-params, and that's fine
    },
}
```

---

### Nesting can go multiple levels deep

Sub-parameters can themselves be categorical nodes with their own `next_level`, creating arbitrary depth:

```python
searchspace = {
    'model': {
        'values': ['linear', 'neural_net'],
        'next_level': {
            'linear': {
                'regularizer': {
                    'values': ['l1', 'l2', 'elasticnet'],
                    'next_level': {
                        'l1':         {'C': {'min': 0.001, 'max': 10.0}},
                        'l2':         {'C': {'min': 0.001, 'max': 10.0}},
                        'elasticnet': {'C': {'min': 0.001, 'max': 10.0},
                                       'l1_ratio': {'min': 0.0, 'max': 1.0}},
                    },
                },
            },
            'neural_net': {
                'architecture': {
                    'values': ['mlp', 'cnn'],
                    'next_level': {
                        'mlp': {'hidden_units': {'values': [64, 128, 256]},
                                'dropout': {'min': 0.0, 'max': 0.5}},
                        'cnn': {'n_filters': {'values': [32, 64, 128]},
                                'kernel_size': {'values': [3, 5, 7]}},
                    },
                },
                'learning_rate': {'min': 1e-4, 'max': 1e-1},
            },
        },
    },
    'epochs': {'min': 10.0, 'max': 100.0},   # always present
}
```

A trial that picks `model=neural_net, architecture=cnn` will produce a config containing `model`, `architecture`, `n_filters`, `kernel_size`, `learning_rate`, and `epochs` — nothing else. The `hidden_units`, `dropout`, `C`, and `l1_ratio` nodes are untouched.

---

## Tuning Guide

| Parameter | Start | Increase if… | Decrease if… |
|---|---|---|---|
| `lambda_` | `1.0` | tree updates slowly / flat | converging too fast to suboptimal |
| `alpha` | `1.0` | too many categorical options, small budget | want to commit faster to early observations |
| `temp` | `1.0` | categorical exploration too greedy | spending budget on clearly bad choices |
| `budget` | 10× param count | rolling loss hasn't plateaued | rolling loss is zero after 20% of run |

**Rule of thumb:** `λ` and `α` interact. High `λ` + low `α` commits quickly. Low `λ` + high `α` keeps the distribution flat. Tune them together.

---

## How It Works

1. **Initialize** a probability tree from the search space. Categorical nodes start uniform. Continuous nodes start with an evenly-spaced KDE reservoir covering `[min, max]`.
2. **Sample** a configuration by traversing the tree top-to-bottom. Categorical nodes use temperature-scaled softmax. Continuous nodes use a weighted KDE draw with Silverman-bandwidth Gaussian jitter, clipped to `[min, max]`.
3. **Evaluate** the objective on the sampled configuration.
4. **Update** the tree. Categorical nodes: Laplace-smoothed count increment. Continuous nodes: append the new observation to the reservoir, prune the lowest-weight point if over capacity, renormalize weights.
5. **Decay** both the update weight (`λ`) and the update probability (`β`) exponentially with `τ = budget`, so the tree stabilizes naturally by the final trial.
6. **Repeat** for `budget` trials.

---

## Limitations

- **Independence assumption** — sibling nodes are updated independently. Encode known interactions via `next_level`.
- **Sequential only** — no native parallel evaluation.
- **Maximization only** — negate the objective to minimize.
- **No formal uncertainty bounds** — use rolling loss as a convergence heuristic.

---

## License

MIT
