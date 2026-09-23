# Limitations

Everything BUTChC does not do, does not model, or has not yet measured, kept in
one place. Each entry is either a deliberate design trade or a piece of
evidence not yet gathered.

If you are deciding whether to use the library, the two sections worth reading
first are [Where the model runs out](#where-the-model-runs-out) and
[What the benchmarks establish, and what they do not](#what-the-benchmarks-establish-and-what-they-do-not).

- [Where the model runs out](#where-the-model-runs-out)
- [Search spaces BUTChC cannot express](#search-spaces-butchc-cannot-express)
- [What the benchmarks establish, and what they do not](#what-the-benchmarks-establish-and-what-they-do-not)
- [The non-results, in full](#the-non-results-in-full)
- [Reading the tables](#reading-the-tables)
- [Open design questions](#open-design-questions)

---

## Where the model runs out

**Parameters in the same branch are modelled independently.** Sibling nodes get
their own marginal, so an interaction between two parameters at the same level
is invisible to the model. This is the single largest modelling trade in the
library and it is deliberate: independence is what makes the update `O(d·K)`
and dependency-free. Encode interactions you know about with `next_level`,
which turns an interaction into structure the tree can see. The `Rosenbrock`
benchmark measures the cost directly — see [the non-results](#the-non-results-in-full).

**Branch selection uses average, not best-case, performance.** A branch whose
best configuration is excellent but whose typical configuration is poor is at a
disadvantage while its sub-parameters are still untuned. Three mechanisms push
back — a node commits more slowly the more sub-space its choices open, its
statistics are recency-weighted so an early verdict decays, and the commitment
exponent ramps from zero rather than from one — but they are mitigation, not a
guarantee. Raise `alpha` or lower `COMMITMENT` if a branch you believe in is
being abandoned. The `Branch trap` benchmark isolates this case, and BUTChC
wins it 25-5 against TPE, so the mitigation works on the case built to break it.

**No acquisition function.** There is no explicit exploration/exploitation
optimisation. `temp`, `gamma` and `explore` are hand-set knobs, not the output
of an uncertainty model.

**No formal uncertainty bounds.** `rolling_loss` is a heuristic, not a
posterior. It tells you the model stopped moving, which is not the same as the
optimum having been found. Read it alongside `n_updates` and `n_gated`: a low
rolling loss means either the distribution has settled or few trials are
clearing the gate, and those call for opposite responses.

**No ordinal node type.** Ordered discrete parameters are modelled as unordered
choices, losing neighbourhood structure. Converted ConfigSpace ordinals are
flattened the same way. This is a real handicap on long ordered sequences.

**Categorical ties commit arbitrarily.** When two branches reach the same
optimum, budget concentrates on whichever got there first. Raise `alpha` or
`temp` if you need both explored.

**`best_params` and `prob_tree` can disagree.** The best configuration found may
sit in a branch the tree assigns low probability. They answer different
questions — one is the best single point seen, the other is where the model
would spend its next trial. A large disagreement means the budget was too small
for the tree to settle.

**Maximization only.** Negate to minimize.

**Not a Gaussian-process method.** On very small budgets — under roughly 50
trials — against a cheap-to-model smooth objective, a GP with a proper
acquisition function will typically do better. BUTChC's measured advantages are
dimension, categorical mixtures, observation noise, objective scale, and zero
dependencies.

---

## Search spaces BUTChC cannot express

BUTChC gives every parameter exactly one parent. ConfigSpace expresses
conditionality as a DAG, which is strictly more general, so some spaces have no
faithful tree form. `butchc.interop.from_configspace` refuses these rather than
approximating them, because a silently approximated space means optimising a
different problem than the one you described.

| Refused | Why |
|---|---|
| `AndConjunction`, `OrConjunction` | a child gated on two parents has no single place in a tree |
| multiple parents for one child | same |
| forbidden clauses | a cross-cutting "this combination is invalid" has no tree position |
| non-categorical parent | only a categorical node can carry a `next_level` |

Two clarifications, because this list is easy to over-read.

**Most invalidity is hierarchical, and hierarchical invalidity is free.** A
constraint like "`penalty=elasticnet` requires `solver=saga`" is expressible:
root on `solver` and give each branch its own `penalty` values. The invalid
region is then unreachable by construction, and BUTChC spends nothing
discovering it. The refusals above concern only constraints that genuinely
cross the tree — two parameters in different subtrees constrained against each
other, or a joint budget over two independent continuous parameters.

**Some of these refusals are converter limits, not model limits.** A child
gated on `learner == svm` *and* `svm.kernel == radial` has two parents, but one
is an ancestor of the other, so the pair is a single root-to-leaf path that
`next_level` expresses natively. The converter refuses it anyway. Measured
across eleven published configuration spaces
([`dev/eval/pcs_stats.py`](../dev/eval/pcs_stats.py)), conjunctions of this
chain-shaped kind are the common case and genuinely independent parents are the
exception — AutoWEKA's 174 multi-parent children are *all* chain-shaped. So the
practical reach of the tree model is wider than the converter currently
admits, and extending the converter to accept chain-shaped conjunctions is
planned.

---

## What the benchmarks establish, and what they do not

**What they establish.** Against TPE across 18 problems at 30 paired seeds: 13
significant wins, zero significant losses, 5 ties, with the held-out half
tested against defaults that never saw it. That is a real result about sample
efficiency on dimension, categorical mixtures, objective scale and noise, and
it is reproducible from a clean checkout with no dependencies.

**They are synthetic.** Every one of the 18 problems was written in this
repository alongside the optimiser, held out or not. Holding half the suite
back from the tuning sweep controls for overfitting the *defaults*; it does not
control for the suite and the optimiser having the same author. This is the
single biggest caveat on every number in the README.

**The founding claim has no head-to-head evidence yet.** BUTChC is built on the
premise that a pre-specified conditional space skips budget a flat optimiser
wastes on inactive parameters. Two halves to that, at different stages:

- *The premise is now measured.* [`dev/eval/pcs_stats.py`](../dev/eval/pcs_stats.py)
  reports the inactive share of eleven configuration spaces published by other
  people for other purposes. AutoWEKA is 98.2% inactive per configuration, the
  2017 auto-sklearn space 88.4%. The waste the library targets is real and
  large in spaces nobody here designed.
- *The head-to-head is not run.* No BUTChC-versus-TPE comparison has been
  executed on any of those spaces. The suite's own `flat_tpe_search` exists to
  isolate the effect on the synthetic problems and finds almost nothing there,
  because those spaces are too narrow for the waste to matter — which is
  consistent with the premise but is not evidence for it.

**No real-world benchmark has been run yet.** The obvious candidate, YAHPO
Gym's `rbv2_super` — 41 parameters, 75.6% inactive, 103 real datasets — is
blocked on the chain-conjunction refusal described above. Until that lands,
every published result is on the synthetic suite.

---

## The non-results, in full

None of these is a statistically significant loss to TPE. They are the problems
where BUTChC does not demonstrate an advantage.

**`Rosenbrock` — 13-17, p=0.585.** Expected and structural. A narrow curved
valley *is* parameter interaction, which the independence assumption cannot
model. Read it as "no advantage", and as the honest price of modelling siblings
independently.

**`Griewank 6D` — 18-12, p=0.362.** Better on the median (-0.4383 against
-0.4915), not separable at 30 seeds.

**`Noisy 5D sphere` — 19-11, p=0.200.** Better on the median (-0.4490 against
-0.8099), not separable at 30 seeds. More seeds would likely settle both this
and Griewank; neither has been run at higher n.

**`Nested pipeline` — 15-15, p=1.000.** The one worth dwelling on, because
conditional structure is what BUTChC exists for and here it draws rather than
leads. The medians are effectively equal (-0.0089 against -0.0096) and BUTChC
arrives *later* — 324 trials to match TPE's final answer against TPE's own 188.
Conditional structure alone is not a sufficient reason to choose this library;
benchmark your own space.

**`Plateau (ties)` — 0-0, saturated.** All three methods reach the discretised
optimum. It is a regression guard for tie handling, not a discriminator.

**`20D sphere` regressed between releases** — from -0.0021 to -0.0746 when the
0.6.0 defaults were compared side by side with 0.5.1's (the benchmark table's
-0.1255 is a separate 30-seed run). It costs no win (still 30-0 against TPE,
still 150× nearer the optimum) and was accepted as the price of gains
elsewhere. A smooth high-dimensional space can recover it by setting
`KDE_RESERVOIR_SIZE = 50` and `MIN_BANDWIDTH_FRACTION = 0.001`.

---

## Reading the tables

- **Ratios flatter.** Most of these objectives are squared errors, where an
  800× ratio is about 28× in distance. Read the raw columns.
- **Medians hide the spread.** A median gap of 2× on one problem and a 30-0
  record on another are different kinds of evidence. Prefer the win-loss column
  to the median column throughout.
- **Rastrigin is the hard case.** Dense local optima limit how much any method
  modelling parameters independently can gain.
- **Overhead is not the headline.** The per-trial cost table measures the
  optimiser with the objective stubbed out. Against a 100 ms objective
  BUTChC's own overhead is at most 0.2% of wall clock and TPE's between 1% and
  28%; against a minute-long objective neither matters. It matters for cheap
  objectives and for very wide spaces, not in general.

---

## Open design questions

Known tensions in the current design. None affects correctness.

**`temp` and `COMMITMENT` fight for one degree of freedom.** `_recompute_prob`
raises choice values to `max(sharpen, 1) / (1 + sub_size)`; `sample_node` then
softmaxes `log(prob)` at temperature `temp`, which is `prob ** (1/temp)`
renormalised. Composed, a choice's sampling probability goes as
`value ** (COMMITMENT * progress / ((1 + sub_size) * temp))`. Three quantities
multiplying into one exponent is very likely why `temp` reads as inert in
sweeps. One of the two should own categorical sharpness.

**`lambda_` and `RANK_SHARPNESS` are also one degree of freedom.** The same
collapse on the continuous side, measured rather than suspected: weights depend
only on `min(RANK_SHARPNESS * lambda_, MAX_SHARPNESS)`, so `lambda_=4.0` and
`lambda_=8.0` give identical medians, as do `lambda_=1.0` and
`rank_sharpness=1.5`. At the shipped `RANK_SHARPNESS=3.0` every
`lambda_ >= 3.33` is clipped and does nothing. Unlike the `temp` case the
ceiling is justified — above 10 the search degrades monotonically — so the
question is which knob exposes it, not whether to raise it.

**`rank_weights` recomputes every weight on each continuous update**, `O(n log n)`
per trial. Negligible at `K = 25`; it becomes the hot path if the reservoir
ever grows.

**Ordinal parameters have no node type.** See above; it needs a new node kind
rather than a parameter change.
