"""Baseline optimisers to measure BUTChC against.

Two are provided, both dependency-free and both consuming the same search-space
dicts BUTChC accepts:

``random_search``
    Uniform sampling. The floor any method must clear to be worth running.

``tpe_search``
    A tree-structured Parzen estimator, the algorithm behind Hyperopt and
    Optuna's default sampler. This is the baseline that matters: it occupies
    the same niche as BUTChC — sequential, model-based, handles conditional
    spaces, makes no gradient assumptions — so a win over random search says
    little that a win over TPE would not say better. Dependency-free, so it
    runs wherever the library does.

``optuna_search``
    Optuna's own ``TPESampler``, for when the claim being made is about the
    tool people actually reach for rather than about the algorithm. Needs
    ``pip install optuna``.

``flat_tpe_search``
    The same TPE, denied any knowledge of the conditional structure. It
    searches the union of every branch's parameters on every trial, which is
    what using an optimiser without a conditional schema costs you.

TPE models ``p(x | y)`` rather than ``p(y | x)``. It splits the trials seen so
far into a good fraction ``gamma`` and the rest, fits a density to each, and
proposes the candidate maximising ``l(x) / g(x)``. Conditional parameters are
handled by fitting each node only on the trials in which that node was
actually sampled, so a parameter living under one branch is never fitted on
trials that took another.

That last property is a deliberate advantage handed to the baseline, and it is
why ``flat_tpe_search`` exists as well. Optimisers that cannot express "this
parameter only exists under that branch" — a flattened grid, a GP over one-hot
encodings, ``RandomizedSearchCV`` over a parameter dict — have to search every
branch's parameters at once. `xgboost.max_depth` gets sampled, and fitted,
on trials that ran a random forest. The difference between ``tpe_search`` and
``flat_tpe_search`` on the same problem is the price of that, isolated.
"""

import math
import random

#: Trials drawn uniformly before a node starts modelling. Below roughly this
#: many observations the good/bad split is dominated by sampling noise.
N_STARTUP = 20

#: Fraction of observations treated as "good" when splitting.
TPE_GAMMA = 0.25

#: Candidates drawn from ``l(x)`` per decision. More candidates approximate
#: the ``argmax l/g`` better at linear cost in arithmetic, not in objective
#: evaluations.
N_CANDIDATES = 24

#: Cap on the good set, so late in a long run the density stays local rather
#: than smearing over every decent trial ever seen.
MAX_GOOD = 25

#: Cap on the bad set. Fitting every rejected trial makes each decision cost
#: O(trials) and the whole run O(budget**2), which dominates the longer
#: problems here. The set is thinned by an even stride over the value ordering
#: rather than truncated, so it keeps the shape of the rejected distribution;
#: truncating to the best of the rejected instead pulls g(x) on top of l(x)
#: and destroys the very contrast the ratio is measuring.
MAX_BAD = 120


# --------------------------------------------------------------- random

def random_config(space, rng):
    """Draw one configuration uniformly, respecting log scales and nesting."""
    config = {}
    for key, spec in space.items():
        if "values" in spec:
            choice = rng.choice(list(spec["values"]))
            config[key] = choice
            sub = spec.get("next_level", {}).get(choice)
            if sub:
                config.update(random_config(sub, rng))
        elif spec.get("log"):
            config[key] = 10 ** rng.uniform(math.log10(spec["min"]),
                                            math.log10(spec["max"]))
        elif spec.get("int"):
            config[key] = rng.randint(int(math.ceil(spec["min"])),
                                      int(math.floor(spec["max"])))
        else:
            config[key] = rng.uniform(spec["min"], spec["max"])
    return config


def random_search(space, objective, budget, seed, report=None):
    """Uniform random search on a fixed budget.

    Args:
        space:     Search-space dict, in BUTChC's format.
        objective: ``callable(config) -> float``, maximised.
        budget:    Number of objective evaluations.
        seed:      Seed for this run's private RNG.
        report:    Optional noise-free scoring function. When given, the
                   return value is ``report(best_config)`` rather than the
                   observed objective, which is what a noisy problem needs:
                   the question is how good the reported configuration truly
                   is, not how lucky its draw was.

    Returns:
        Best objective value seen, or its reported score.
    """
    rng = random.Random(seed)
    best_seen, best_reported = -float("inf"), -float("inf")
    for _ in range(budget):
        config = random_config(space, rng)
        value = objective(config)
        if value > best_seen:
            best_seen = value
            best_reported = report(config) if report else value
    return best_reported


# --------------------------------------------------------------- TPE

def tpe_search(space, objective, budget, seed, report=None,
               n_startup=N_STARTUP, gamma=TPE_GAMMA,
               n_candidates=N_CANDIDATES):
    """Tree-structured Parzen estimator on a fixed budget.

    Args and return value match ``random_search``.

    Args:
        n_startup:    Uniform trials per node before modelling begins.
        gamma:        Fraction of a node's observations treated as good.
        n_candidates: Candidates scored per decision.
    """
    rng = random.Random(seed)
    history = []                     # [(flat_config, objective_value)]
    best_seen, best_reported = -float("inf"), -float("inf")

    for _ in range(budget):
        config = _suggest(space, history, rng, n_startup, gamma, n_candidates)
        value = objective(config)
        if math.isfinite(value):
            history.append((config, value))
        if value > best_seen:
            best_seen = value
            best_reported = report(config) if report else value
    return best_reported


def _suggest(space, history, rng, n_startup, gamma, n_candidates):
    """Propose one configuration, descending into whichever branch is chosen."""
    config = {}
    for key, spec in space.items():
        observed = [(cfg[key], val) for cfg, val in history if key in cfg]

        if "values" in spec:
            choice = _suggest_categorical(spec, observed, rng, n_startup,
                                          gamma, n_candidates)
            config[key] = choice
            sub = spec.get("next_level", {}).get(choice)
            if sub:
                # Sub-trees are fitted on the whole history; _suggest filters
                # per key, so trials that took another branch simply have no
                # observation for these parameters and drop out.
                config.update(_suggest(sub, history, rng, n_startup, gamma,
                                       n_candidates))
        else:
            config[key] = _suggest_continuous(spec, observed, rng, n_startup,
                                              gamma, n_candidates)
    return config


def _split(observed, gamma):
    """Partition observations into the good fraction and the rest."""
    ordered = sorted(observed, key=lambda pair: pair[1], reverse=True)
    n_good = max(1, min(int(math.ceil(gamma * len(ordered))), MAX_GOOD))
    good = [v for v, _ in ordered[:n_good]]
    rest = [v for v, _ in ordered[n_good:]]
    if len(rest) > MAX_BAD:
        stride = len(rest) / MAX_BAD
        rest = [rest[int(i * stride)] for i in range(MAX_BAD)]
    return good, rest


def _suggest_categorical(spec, observed, rng, n_startup, gamma, n_candidates):
    """Pick the choice maximising ``l/g``, or draw uniformly during startup."""
    choices = list(spec["values"])
    if len(observed) < n_startup:
        return rng.choice(choices)

    good, bad = _split(observed, gamma)
    # One pseudo-count per choice keeps an untried choice reachable and stops
    # a zero in the denominator.
    l = _counts(choices, good)
    g = _counts(choices, bad)
    weights = [l[c] for c in choices]
    candidates = rng.choices(choices, weights=weights, k=n_candidates)
    return max(candidates, key=lambda c: l[c] / g[c])


def _counts(choices, values):
    """Laplace-smoothed frequency of each choice, normalised to sum to 1."""
    tally = {c: 1.0 for c in choices}
    for v in values:
        if v in tally:
            tally[v] += 1.0
    total = sum(tally.values())
    return {c: n / total for c, n in tally.items()}


def _suggest_continuous(spec, observed, rng, n_startup, gamma, n_candidates):
    """Draw from the good density and keep the candidate maximising ``l/g``."""
    log_scale = bool(spec.get("log"))
    integral = bool(spec.get("int"))
    lo = math.log10(spec["min"]) if log_scale else float(spec["min"])
    hi = math.log10(spec["max"]) if log_scale else float(spec["max"])
    if integral:
        # Match BUTChC: rounding maps [k-0.5, k+0.5) to k, so the endpoints
        # need half-unit shoulders to be drawn as often as interior integers.
        lo, hi = lo - 0.5, hi + 0.5

    def externalise(x):
        x = min(max(x, lo), hi)
        if log_scale:
            return 10.0 ** x
        if integral:
            return min(max(int(round(x)), int(math.ceil(spec["min"]))),
                       int(math.floor(spec["max"])))
        return x

    if len(observed) < n_startup:
        return externalise(rng.uniform(lo, hi))

    internal = [(math.log10(v) if log_scale else float(v), y)
                for v, y in observed]
    good, bad = _split(internal, gamma)

    l_mus, l_sigmas = _parzen(good, lo, hi)
    g_mus, g_sigmas = _parzen(bad, lo, hi)

    best, best_score = None, -float("inf")
    for _ in range(n_candidates):
        i = rng.randrange(len(l_mus))
        x = min(max(rng.gauss(l_mus[i], l_sigmas[i]), lo), hi)
        score = (_log_density(x, l_mus, l_sigmas, lo, hi)
                 - _log_density(x, g_mus, g_sigmas, lo, hi))
        if score > best_score:
            best, best_score = x, score
    return externalise(best)


def _parzen(values, lo, hi):
    """Place a Gaussian on each observation, plus one flat prior component.

    Each bandwidth is the larger gap to its two neighbours, which makes the
    density tight where observations cluster and loose where they are sparse.
    Bandwidths are then clamped: no narrower than the range divided by the
    component count, so a duplicated observation cannot become a spike, and no
    wider than the range itself.
    """
    span = hi - lo
    mus = sorted(values) + [(lo + hi) / 2.0]
    mus.sort()
    n = len(mus)

    sigma_min = span / min(100.0, n + 1.0)
    sigmas = []
    for i, mu in enumerate(mus):
        left = mu - mus[i - 1] if i > 0 else mu - lo
        right = mus[i + 1] - mu if i + 1 < n else hi - mu
        sigmas.append(min(max(max(left, right), sigma_min), span))
    # The prior component spans the whole range, so a candidate far from every
    # observation still has somewhere to come from.
    sigmas[mus.index((lo + hi) / 2.0)] = span
    return mus, sigmas


# ------------------------------------------------------------ flat TPE

def flatten_space(space, _prefix=""):
    """Collapse a conditional space into one flat space of qualified names.

    Every branch's parameters are hoisted to the top level. The categorical
    nodes survive as ordinary choices — a flat optimiser can still *pick* the
    model, it just cannot express that the choice gates anything. Names are
    qualified by their path so that two branches with a `lr` each stay
    separate parameters, which is the generous reading: a user flattening by
    hand would more likely collide them into one.

    Returns:
        ``(flat_space, routes)`` where ``routes`` maps a qualified name to the
        ``(path, key)`` it came from, so a drawn configuration can be routed
        back to the shape the objective expects.
    """
    flat, routes = {}, {}
    for key, spec in space.items():
        name = _prefix + key
        if "values" in spec:
            flat[name] = {"values": list(spec["values"])}
            routes[name] = (_prefix, key)
            for choice, sub in spec.get("next_level", {}).items():
                sub_flat, sub_routes = flatten_space(
                    sub, f"{name}={choice}:")
                flat.update(sub_flat)
                routes.update(sub_routes)
        else:
            flat[name] = dict(spec)
            routes[name] = (_prefix, key)
    return flat, routes


def unflatten_config(flat_config, space, _prefix=""):
    """Keep only the parameters the chosen branches actually use.

    The flat optimiser proposes a value for every parameter in the union. This
    walks the real tree and picks out the ones that apply, which is what the
    objective is given. The rest were sampled, modelled, and thrown away —
    that waste is the thing being measured.
    """
    config = {}
    for key, spec in space.items():
        name = _prefix + key
        if "values" in spec:
            choice = flat_config[name]
            config[key] = choice
            sub = spec.get("next_level", {}).get(choice)
            if sub:
                config.update(unflatten_config(
                    flat_config, sub, f"{name}={choice}:"))
        else:
            config[key] = flat_config[name]
    return config


def flat_tpe_search(space, objective, budget, seed, report=None,
                    n_startup=N_STARTUP, gamma=TPE_GAMMA,
                    n_candidates=N_CANDIDATES):
    """TPE with the conditional structure taken away.

    Identical to ``tpe_search`` in every respect except the space it is given:
    the union of all branches, searched jointly, every trial. This is the
    comparison that isolates what a conditional schema is worth, because the
    algorithm, the seeds, the budget and the code path are all held fixed and
    only the structure changes.

    Args are as ``tpe_search``.
    """
    flat, _routes = flatten_space(space)
    rng = random.Random(seed)
    history = []
    best_seen, best_reported = -float("inf"), -float("inf")

    for _ in range(budget):
        flat_config = _suggest(flat, history, rng, n_startup, gamma,
                               n_candidates)
        config = unflatten_config(flat_config, space)
        value = objective(config)
        # The model is fitted on the flat draw, not the routed one: the flat
        # optimiser has no way to know which of its parameters were ignored,
        # so every one of them takes credit or blame for the result.
        history.append((flat_config, value))
        if value > best_seen:
            best_seen = value
            best_reported = report(config) if report else value
    return best_reported


# --------------------------------------------------------------- Optuna

def optuna_search(space, objective, budget, seed, report=None,
                  n_startup=N_STARTUP):
    """Optuna's ``TPESampler`` on the same budget, as a reference baseline.

    ``tpe_search`` above is the same algorithm but not the same code, and the
    defaults differ (Optuna recency-weights its observations and splits at
    ``min(ceil(0.1 n), 25)`` where this module splits at a flat quarter). When
    the point is "how does BUTChC compare to what people actually run", run
    this one; when the point is a quick regression check with no dependencies,
    run ``tpe_search``.

    Conditional parameters are expressed the way Optuna expects: a suggestion
    is only requested once its parent choice is known, so a trial that took
    another branch simply has no observation for the parameters under this one.

    Requires ``pip install optuna``. Raises ``ImportError`` otherwise rather
    than silently substituting a different method.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def suggest(trial, sub_space, config):
        for key, spec in sub_space.items():
            if "values" in spec:
                choice = trial.suggest_categorical(key, list(spec["values"]))
                config[key] = choice
                nested = spec.get("next_level", {}).get(choice)
                if nested:
                    suggest(trial, nested, config)
            elif spec.get("int"):
                config[key] = trial.suggest_int(key, int(math.ceil(spec["min"])),
                                                int(math.floor(spec["max"])))
            else:
                config[key] = trial.suggest_float(key, spec["min"], spec["max"],
                                                  log=bool(spec.get("log")))
        return config

    seen = {}

    def wrapped(trial):
        config = suggest(trial, space, {})
        value = objective(config)
        seen[trial.number] = config
        return value

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup),
    )
    study.optimize(wrapped, n_trials=budget)

    if report is None:
        return study.best_value
    return report(seen[study.best_trial.number])


_SQRT2 = math.sqrt(2.0)
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def _log_density(x, mus, sigmas, lo, hi):
    """Log density of an equally-weighted mixture, truncated to ``[lo, hi]``."""
    total = 0.0
    for mu, sigma in zip(mus, sigmas):
        z = (x - mu) / sigma
        # Renormalise for the truncation, or mass falling outside the bounds
        # would make components near an edge look less likely than they are.
        mass = 0.5 * (math.erf((hi - mu) / (sigma * _SQRT2))
                      - math.erf((lo - mu) / (sigma * _SQRT2)))
        if mass <= 1e-12:
            continue
        total += math.exp(-0.5 * z * z - _LOG_SQRT_2PI) / (sigma * mass)
    return math.log(max(total / len(mus), 1e-300))
