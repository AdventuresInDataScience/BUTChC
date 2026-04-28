# BUTChC — Bayesian Update Tree Chained Conditionally

## What Is It?

BUTChC is a probabilistic, tree-structured black-box optimiser. It maintains a probability distribution over a structured search space and iteratively refines that distribution based on observed objective values. It is designed to find good configurations efficiently without requiring gradients, differentiability, or any assumptions about the objective function's internal structure.

The name reflects its core mechanism: a Bayesian-style update rule applied to a tree of conditional probability nodes, where each branch of the tree is only relevant given a specific choice made higher up.

---

## Core Algorithm

1. **Initialise** a probability tree from the search space definition. Categorical nodes start with uniform probability over all choices. Continuous nodes start with a flat KDE reservoir uniformly covering `[min, max]`.
2. **Sample** a configuration by traversing the tree top-to-bottom, drawing one value per node. Categorical nodes use temperature-scaled softmax sampling. Continuous nodes use a weighted KDE draw with Silverman bandwidth jitter.
3. **Evaluate** the objective function on the sampled configuration. The objective is a black box — BUTChC only requires a scalar return value (higher = better).
4. **Update** the tree in-place. Categorical nodes use Laplace-smoothed count updates. Continuous nodes append the new observation to a weighted reservoir and prune the lowest-weight point if over budget.
5. **Repeat** for `budget` trials, with decaying update weight (`λ`) and decaying update probability (`β`) — both annealed over time with `τ = budget`.

---

## Key Design Decisions

### Hierarchical / Conditional Search Space
Parameters can be conditional on parent choices via `next_level`. Sub-parameters only exist in the tree when their parent value is selected. This avoids wasting budget sampling irrelevant parameter combinations.

### Temperature-Controlled Exploration
The softmax temperature `temp` controls exploration vs exploitation for categorical nodes. High temperature → near-uniform sampling (explore). Low temperature → greedy towards high-probability choices (exploit). Defaults to `1.0`.

### Dual Decay Schedule
Both `λ` (update weight) and `β` (update probability) decay exponentially with `τ = budget`:
- **λ decay** prevents late-stage observations from overwriting well-founded early estimates.
- **β decay** acts like a shrinking effective batch size — dense early updates stabilise the tree quickly; sparse late updates let good estimates settle without being disrupted.

Both decay to ~37% of their initial values by the final trial. This is deliberate and principled — one budget cycle of decay is a natural, scale-invariant annealing schedule.

### KDE Continuous Nodes
Continuous parameters are modelled using a weighted reservoir of `KDE_RESERVOIR_SIZE` sample points rather than a parametric distribution. This makes no assumptions about shape, naturally captures multimodality, and requires no moments. The reservoir starts uniform (flat prior) and shifts toward good regions as trials accumulate. Silverman's rule adapts the bandwidth automatically.

### Convergence / Loss Tracking
At each trial, BUTChC records a `loss` — the mean normalised magnitude of probability updates across all touched nodes. This is not the objective value; it measures model surprise. A rolling mean is also tracked. Useful for:
- Early stopping when rolling loss plateaus
- Detecting overfitting (loss rises after plateau)
- Estimating uncertainty in the final model

### Warm Starting
The `start_prob_tree` argument accepts a probability tree from a previous run. This allows chaining runs, resuming interrupted optimisations, or seeding from domain knowledge.

---

## Search Space Definition

```python
searchspace = {
    # Categorical parameter
    'optimizer': {
        'values': ['adam', 'sgd'],
        'next_level': {
            'adam': {
                'lr':       {'min': 1e-4, 'max': 1e-1},
                'beta1':    {'values': [0.85, 0.90, 0.95, 0.99]},
            },
            'sgd': {
                'lr':       {'min': 1e-3, 'max': 1e-1},
                'momentum': {'min': 0.0,  'max': 0.99},
            },
        },
    },
    # Categorical with no sub-parameters
    'batch_size': {'values': [16, 32, 64, 128]},
    # Continuous parameter (KDE)
    'dropout':    {'min': 0.0, 'max': 0.5},
}
```

Each node is either:
- **Categorical**: `{'values': [...], 'next_level': {...}}` — `next_level` is optional
- **Continuous**: `{'min': float, 'max': float}` — modelled with KDE reservoir

---

## Usage Example

```python
from butchc_optimizer import BUTChC_optimize

def objective(config):
    # Your evaluation logic here — must return a scalar, higher = better
    lr = config['lr']
    return -train_model(lr=lr)  # negate loss to maximise

results = BUTChC_optimize(
    searchspace = searchspace,
    objective   = objective,
    budget      = 200,
    lambda_     = 1.0,   # initial update weight
    alpha       = 1.0,   # Laplace smoothing strength
    temp        = 1.0,   # softmax temperature
    verbose     = True,
)

print(results['best_params'])   # best configuration found
print(results['best_value'])    # corresponding objective value
print(results['rolling_loss'])  # convergence curve
```

### Warm Start Example

```python
# Continue from a previous run
results2 = BUTChC_optimize(
    searchspace     = searchspace,
    objective       = objective,
    budget          = 100,
    lambda_         = 1.0,
    alpha           = 1.0,
    start_prob_tree = results['prob_tree'],  # warm start
)
```

---

## Return Value

```python
{
    'best_params':  {...},   # config dict with highest objective
    'best_value':   float,   # objective value at best_params
    'prob_tree':    {...},   # final probability tree (usable as warm start)
    'history':      [...],   # per-trial list of {params, objective, loss, rolling_loss}
    'loss_history': [...],   # per-trial loss values (convenience)
    'rolling_loss': [...],   # per-trial rolling mean loss (convenience)
}
```

---

## Parameters

| Parameter | Type | Description |
|---|---|---|
| `searchspace` | dict | Search space definition |
| `objective` | callable | Black-box function returning scalar (higher = better) |
| `budget` | int | Total number of objective evaluations |
| `lambda_` | float | Initial update weight. Higher = faster adaptation, more risk of premature convergence |
| `alpha` | float | Laplace smoothing for categorical nodes. Higher = smoother, slower to commit |
| `temp` | float | Softmax temperature. Higher = more exploration |
| `start_prob_tree` | dict or None | Optional warm-start tree from a previous run |
| `verbose` | bool | Print per-trial progress |

---

## Assumptions

- **Independence between sibling nodes.** Each node in the tree updates independently. Interaction effects between sibling parameters (e.g. parameter A is only good when parameter B has a specific value) are not captured unless encoded via `next_level` conditioning.
- **Maximisation.** The objective is maximised. To minimise, negate the return value.
- **Scalar objective.** The objective must return a single scalar per evaluation. Multi-objective use requires scalarisation.
- **Stationarity.** The objective landscape is assumed not to change between trials. BUTChC is not designed for non-stationary or adversarial objectives.

---

## Pros

- **No gradients required.** Works with any objective, including non-differentiable, stochastic, or simulation-based functions.
- **Hierarchical / conditional spaces.** Native support for parameters that only exist given specific parent choices — far more expressive than flat search spaces.
- **Interpretable.** The probability tree is directly readable after training. High-probability paths tell you what the model has learned without any post-hoc attribution.
- **Lightweight.** Pure Python standard library (no dependencies). Fast per-iteration cost — the bottleneck is always the objective, not the tree operations.
- **Warm startable.** Runs can be chained, resumed, or seeded from prior knowledge.
- **Convergence signal.** Built-in loss tracking provides a principled stopping criterion without needing a validation set.
- **Non-parametric continuous modelling.** KDE reservoir makes no distributional assumptions — naturally handles multimodal and asymmetric continuous parameter landscapes.

## Cons / Limitations

- **Independence assumption.** Sibling node interactions are not modelled. For problems where parameter interactions dominate, BUTChC may converge slowly or to suboptimal regions. Mitigation: encode known interactions via `next_level` conditioning.
- **Single-select categorical only.** Each categorical node selects exactly one value per trial. Multi-select (choose k from n) requires structural extension not yet implemented.
- **Sequential evaluation.** Each trial is evaluated one at a time. No native parallelism — parallel objective evaluation would require batching the sample/update loop, which adds coordination overhead.
- **Budget sensitivity.** The decay schedule is tied to `budget`. If the budget is too small for the search space size, the tree may not have converged. If too large, compute is wasted on a converged tree.
- **No explicit uncertainty quantification.** Loss tracking gives a convergence signal but not formal confidence intervals over the objective landscape.

---

## Caveats

- **λ and α interact.** High λ with low α commits quickly to early observations. Low λ with high α keeps the distribution flat for longer. Tune these together.
- **Temperature affects only categorical nodes.** Continuous node exploration is controlled entirely by the KDE bandwidth, which adapts automatically via Silverman's rule.
- **Loss of 0.0 in later trials** is expected and correct — it reflects skipped updates from β decay, not a malfunction.
- **The best_params in history may differ from best_value's config** if the best trial occurred before the tree had converged. Always use `results['best_params']` rather than inspecting the final tree's mode.
