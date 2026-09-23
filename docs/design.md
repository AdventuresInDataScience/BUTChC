# Design notes

Why BUTChC is built the way it is. For the parameter list see
[api.md](api.md); for recipes see [examples.md](examples.md); for what the
method does not do see [limitations.md](limitations.md).

---

## What it is

A probabilistic, tree-structured black-box optimizer. It maintains a
distribution over a structured search space and refines it from observed
objective values. No gradients, no differentiability, no assumptions about the
objective's internals.

The name reflects the mechanism: a Bayesian-style update applied to a tree of
conditional probability nodes, where each branch is relevant only given a
specific choice higher up.

---

## The loop

1. **Initialize** a probability tree from the validated search space.
   Categorical nodes start uniform. Continuous nodes start with a flat
   reservoir spanning the range in *internal* coordinates.

2. **Warm up** for `n_warmup` trials, sampling every node uniformly. Defaults
   to 0 — a fresh tree is already an even grid at even weights, so it samples
   near-uniformly anyway. Warm-up holds `quality` at 0, so no continuous
   archive moves; categorical nodes still record their visit and rank, since a
   mean over visits wants exactly this unbiased sample (see
   [categorical nodes score mean rank](#categorical-nodes-score-mean-rank-ungated)).

3. **Sample** a configuration top-to-bottom. Categorical nodes use
   temperature-scaled softmax over stored probabilities. Continuous nodes draw
   a reservoir point by weight, add Gaussian jitter at Silverman bandwidth, and
   reflect into range. With probability `explore` a node ignores the model and
   draws uniformly.

4. **Evaluate** the objective. Non-finite returns are recorded but excluded
   from ranking and best-tracking.

5. **Rank** the result against all finite objectives seen so far, as a midrank
   in `[0, 1]` with ties counted half. The two node types consume it
   differently: continuous nodes apply the `gamma` gate and convert what
   survives into `quality = (rank - gamma) / (1 - gamma)`; categorical nodes
   use the raw midrank, ungated.

6. **Update**. Categorical nodes record a discounted visit and rank, then
   recompute probabilities from the recency-weighted mean. Continuous nodes
   append to an elite archive and reweight by within-archive rank.

7. **Repeat**, with categorical commitment ramping up over the budget.

With `batch > 1`, steps 3 and 4 run for `k` configurations before step 6 is
applied `k` times. See [batching](#batching) below.

---

## Coordinate spaces

Continuous parameters are modelled in an internal space and reported in the
external space the user defined.

| Kind | Internal | External |
|---|---|---|
| plain | `x` | `x` |
| `log` | `log10(x)` | `10 ** internal` |
| `int` | continuous | `round(internal)` |

Modelling log parameters in log space is not a convenience. A search over
`lr ∈ [1e-5, 1e-1]` sampled linearly places 99.99% of its mass above `1e-3`;
the bottom three decades are unreachable in any realistic budget. Uniform-in-log
spacing makes each decade equally likely.

For integer parameters the value written into the reservoir is the *rounded*
value the objective actually received, not the raw draw. Otherwise the model
and the objective disagree about what was evaluated.

---

## Node types

### Categorical

```
{'counts':   {choice: float},   # discounted sum of ranks earned
 'visits':   {choice: float},   # discounted visit count
 'prob':     {choice: float},   # derived, recomputed every update
 'sub_size':  int,              # most params one config below this node can carry
 'prior':    {choice: float},
 'prior_strength': float,
 ['next_level': {choice: subtree}]}
```

Each choice is scored by mean rank per visit, smoothed toward `NEUTRAL_QUALITY`
by `alpha` pseudo-visits:

```
value[k] = (counts[k] + alpha * NEUTRAL_QUALITY) / (visits[k] + alpha)
prob[k]  ∝ value[k] ** exponent  ×  prior[k] ** (strength / (strength + visits))
```

Because the score is a midrank, `NEUTRAL_QUALITY` (0.5) is knowable in advance rather than
estimated: a configuration drawn at random has expected rank 0.5. A choice above
that gains mass, one below loses it, and one nobody has tried sits in the middle
rather than at zero.

### Continuous

```
{'min', 'max',            # internal-space bounds
 'ext_min', 'ext_max',    # as the user wrote them
 'log', 'int',
 'reservoir': [float],    # internal-space sample values
 'weights':   [float],    # normalised, sum to 1
 'scores':    [float]}    # objective value each entry earned
```

The reservoir is an **elite archive**. On update the observation is appended
with its score; over capacity, the worst-scoring entry is evicted. Weights decay
geometrically with within-archive score rank:

```
sharpness = min(RANK_SHARPNESS * lambda_, MAX_SHARPNESS)
w_i       ∝ exp(sharpness * (u_i - 1))       # u_i = normalised rank, 1 = best
```

The initial grid carries a sentinel score of `-inf` so real observations
displace it. Eviction ties are broken uniformly at random — otherwise the grid
erodes from one end and skews early coverage toward one half of the range.

---

## Key design decisions

### The objective drives the update

This is the difference between an optimizer and a random walk. Weighting by
rank rather than raw objective makes the rule invariant to scale, offset and
outliers: an accuracy in `[0, 1]` and a negated MSE in the millions behave
identically, and one catastrophic outlier cannot permanently dominate the
archive.

The quantile gate (`gamma`) is the same idea as TPE's split between good and bad
observations, restricted to modelling the good side. Ranking against the full
history is self-correcting: as the model improves, the observed set fills with
good values, so the bar for clearing `gamma` rises automatically.

### Categorical nodes score mean rank, ungated

Two separate choices, each load-bearing.

**Mean, not total.** A mean separates "this branch is good" from "this branch
was tried a lot". A total does not: a branch sampled three times as often
accumulates roughly three times the score at equal quality, amplifying its own
early luck.

**Ungated rank, not gated quality.** The gate is self-referential. A trial is
ranked against the optimizer's own increasingly biased history, so a choice that
comes to dominate sampling stops clearing the gate it is winning, and its
discounted counts decay away. Mean rank keeps separating under the same
pressure — a choice taking 90% of the budget and winning every time still sits
near 0.55 against a loser's 0.05. The `gamma` gate still governs the continuous
archive, where elitism is the point.

### Discounting

These arms are not stationary: a branch's reward distribution improves as its
own sub-model learns, so a branch with a wide or sensitive sub-space scores
badly exactly while it is being tuned. An undiscounted mean fixes that early
verdict permanently, and a starved branch stops being visited, so the stale
estimate is never revised.

Statistics decay by `DISCOUNT` per visit, giving a recency-weighted mean over
roughly `1 / (1 - DISCOUNT)` recent visits. An unvisited branch's evidence fades
back toward neutral — the difference between "we tried it and it was bad" and
"we have not tried it lately".

### Sub-space-aware commitment

The commitment exponent is `max(sharpen, 1) / (1 + sub_size)` and `sharpen`
ramps from 0 to `COMMITMENT` over the run.

An exponent of 1 is already decisive: a branch with four times a sibling's mean
score takes four times the budget from the first trials on. That is enough to
starve a branch whose sub-space has not been tuned yet — and a wide sub-space
always scores badly until it is. Dividing by `1 + sub_size` makes a node commit
more slowly the more there is to tune underneath, so one exponent can serve both
a flat six-way choice of activation function, where fast commitment is free, and
a choice of model family, where committing early means judging an untuned
sub-model.

The `Branch trap` problem in `benchmarks/problems.py` isolates this case.

### Priors multiply, they do not contribute counts

A prior over probabilities is inert under a mean rule: a choice holding 47 of 50
pseudo-visits at the average rank still has the average rank, so it cancels
exactly. Nor can a prior be pinned to a fixed score, because rank-based scores
regress toward neutral as a choice comes to dominate, putting a fixed target
permanently out of reach of any evidence.

So a categorical prior is a **multiplicative bias** on the probabilities, with
an exponent decaying as `strength / (strength + visits)`. With no trials yet the
probabilities are exactly what you stated; every visit dilutes them. It is kept
apart from the discounted counts, so a belief is overturned by contrary evidence
rather than forgotten because time passed.

Continuous priors seed the reservoir directly, with points drawn from an explicit
list or from the quantiles of a normal centred on `{'mean', 'std'}`
(deterministically, so seeding never depends on RNG state). Those points carry a
score tier between the uniform grid and any real objective value:

```
GRID_SCORE  = -inf              # the flat start
PRIOR_SCORE = -MAX_FLOAT        # your belief
<any finite objective value>    # evidence
```

This orders the three exactly as intended. Prior points outrank the flat grid,
so they dominate early sampling. Every real observation outranks them, so
evidence displaces belief rather than competing with it forever. And because
eviction removes the worst-scoring entry, the prior is consumed gradually as the
archive fills — no separate decay schedule to tune.

Priors never fill the whole reservoir: at least `MIN_GRID_POINTS` grid points
remain, spanning the full range, so no `prior_strength` can make part of the
range unreachable.

### Elite archive, not exponential forgetting

Two eviction schemes fail:

- **Weight-based eviction with a large learning rate.** A single new
  observation takes a large share of total mass and halves every prior point.
  Effective sample size collapses to a handful of `KDE_RESERVOIR_SIZE` — a
  random walk with a memory of six samples.
- **Weight-based eviction with a small learning rate.** The new point enters
  below the weight of every existing grid point, so it is the minimum and is
  evicted on the same call. The reservoir never changes at all.

Evicting by *score* avoids both. The archive is the best K observations, which
is exactly the set a KDE should sit over.

### Reflection, not clipping

Adding jitter and clipping to `[min, max]` deposits an atom of probability on
each bound. Over a run that biases every continuous search toward interval
edges — a systematic error invisible in aggregate metrics. Reflection folds the
excess back inside and preserves the density shape.

### Bandwidth floor

Silverman's rule collapses to zero when the archive concentrates on identical
points, which would freeze local exploration. The bandwidth is floored at
`MIN_BANDWIDTH_FRACTION` of the node's range and capped at the range itself.

### Exploration floor

With probability `explore` a node draws uniformly, independent of the model.
This bounds how far any single parameter's marginal can collapse. Note it is a
floor **per node**: a config in which all `d` nodes explore has probability
`explore ** d`, so it does not make BUTChC a superset of random search in high
dimensions.

### Structural traces

Sampling returns both the flat config and a trace mirroring the tree's shape.
Updates walk the trace rather than matching parameter names against the flat
config, so a name appearing in two branches can never update the wrong node.
The validator rejects genuinely ambiguous spaces outright; the trace makes the
mechanism correct regardless.

### Validation before evaluation

Every detectable problem — malformed nodes, duplicate names, inverted bounds,
`log` with non-positive `min`, out-of-range hyperparameters, mismatched
warm-start trees — raises before the first objective call. When a trial costs an
hour of GPU time, failing on trial 40 because of a typo is expensive.

### Private RNG

The optimizer seeds a `random.Random` instance rather than calling the global
module. Runs are reproducible via `seed=` regardless of surrounding global
state, and running an optimization does not perturb the caller's own stream.

---

## Batching

`batch=k` draws `k` configurations from one state of the tree before any update
is applied, so the model is stale for `k-1` evaluations. That is a genuine cost
in sample efficiency and the reason `batch=1` is the default: sequential
model-based search earns its advantage precisely by updating after every
observation.

Three rules make the batched path correct rather than merely fast.

**Rank against the pre-batch history.** Every member of a batch is ranked
against the observations that existed when it was sampled, never against its
batch-mates. Ranking within the batch would give the last member `k-1`
comparisons the first one did not have, making a trial's score depend on its
position in the batch. At `k=1` this is identical to ranking after insertion,
which is why the sequential path is byte-for-byte unchanged.

**Update in sample order.** Not completion order. A seeded run must give the
same answer whichever worker finishes first, so an executor is required to
preserve input order and the updates are replayed in the order the configs were
drawn.

**Truncate the final batch.** A budget is exactly a budget however `k` divides
it.

The update rule itself needs no change: applying `k` updates in a loop discounts
`k` times and evicts `k` times, exactly as `k` sequential trials would.

### Why batch diversity is not a crisis here

Methods that *argmax* an acquisition function — GP-EI, TPE's `l(x)/g(x)` — draw
`k` identical points from an unchanged model, which is why Optuna needs
`constant_liar`. BUTChC has no argmax. Continuous nodes draw a reservoir point
and add Silverman-bandwidth jitter; categorical nodes sample a softmax. `k` draws
from a frozen tree are diverse by construction.

The remaining concentration case is a categorical node late in a run, when
`prob` has collapsed onto one choice and the whole batch takes it. That is
exploitation, and a sequential run spends the same consecutive trials the same
way. It is not a batch pathology.

Measured cost at fixed budget over 18 problems and 20 seeds: `k=8` costs about
2.6% in median regret, `k=16` about 21%, `k=32` about 43%. Below `k=8` the cost
is inside seed noise. Reproduce with `benchmarks/batch_cost.py`.

### BUTChC does not own the pool

You pass an `executor`. This keeps the package dependency-free, keeps process
lifecycle out of a library that has no business owning it, and leaves the
picklability decision with the only person who can make it — whoever wrote the
objective. Process pools transmit by pickle and cannot carry closures or
interactively defined functions; threads pickle nothing and parallelise fine for
any objective that releases the GIL, which covers NumPy, scikit-learn, PyTorch
and anything shelling out to a subprocess.

---

## The loss signal

Each trial records a `loss`: the mean normalised magnitude of the change the
update made across the nodes it touched. It measures how far the model moved,
not how good the configuration was.

Categorical nodes move on every trial and contribute their delta whether or not
the gate passed; continuous archives move only when the gate passes. Read
`rolling_loss` alongside `n_updates` and `n_gated` — a low rolling loss means
either the distribution has settled or few trials are clearing the gate, and
those are different situations calling for opposite responses.

`rolling_loss` is a heuristic, not a posterior. It says the model stopped
moving; it does not say the optimum was found.

---

## Complexity

Per trial, with `d` nodes on the sampled path and reservoir size `K`:

- Sampling: `O(d · K)` — the reservoir-weighted draw and bandwidth computation
  dominate
- Ranking: `O(log t)` lookup, `O(t)` insertion into a sorted list
- Update: `O(d · K log K)` — rank sorting of the archive

Memory is `O(total_nodes · K)` for the tree plus `O(budget)` for history. With
defaults (`K = 25`) all of this is negligible next to any real objective
evaluation.

---

## What this method does not do

The consequences of the trades above — no within-branch interactions, branch
selection on average rather than best case, no acquisition function, no formal
uncertainty bounds, and the small-budget gap to a Gaussian process — are
catalogued with their measured costs in
**[limitations.md](limitations.md#where-the-model-runs-out)**.

Known tensions in the current design, including the two places where two knobs
share one degree of freedom, are in
**[limitations.md](limitations.md#open-design-questions)**.
