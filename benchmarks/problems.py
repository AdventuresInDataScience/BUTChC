"""Benchmark problems.

Three sets. ``TUNING`` is what the defaults in :mod:`butchc.optimizer` are
selected on. ``HELDOUT`` is not consulted during tuning and shows whether that
selection generalises or merely fits the first set; anything reported from
``TUNING`` alone should be read as optimistic. ``NOISY`` carries an extra
noise-free scoring function, because on a noisy objective the question is how
good the reported configuration truly is, not how lucky its draw was.

Every problem is a maximisation with a known optimum of 0.0, so results are
comparable across rows and a value nearer zero is better.
"""

import math

PI = math.pi


def _sphere(centre):
    keys = [f"x{i}" for i in range(len(centre))]
    space = {k: {"min": -5.0, "max": 5.0} for k in keys}
    def f(c):
        return -sum((c[k] - m) ** 2 for k, m in zip(keys, centre))
    return space, f


# --------------------------------------------------------------- tuning set

def _quad2d():
    space = {"x": {"min": -3.0, "max": 3.0}, "y": {"min": -3.0, "max": 3.0}}
    return space, lambda c: -((c["x"] - 1) ** 2) - ((c["y"] + 1) ** 2)


def _rosenbrock():
    space = {"x": {"min": -2.0, "max": 2.0}, "y": {"min": -1.0, "max": 3.0}}
    return space, lambda c: -((1 - c["x"]) ** 2 + 100 * (c["y"] - c["x"] ** 2) ** 2)


def _rastrigin(d):
    keys = [f"x{i}" for i in range(d)]
    space = {k: {"min": -5.12, "max": 5.12} for k in keys}
    def f(c):
        return -(10 * d + sum(c[k] ** 2 - 10 * math.cos(2 * PI * c[k]) for k in keys))
    return space, f


def _ackley(d):
    keys = [f"x{i}" for i in range(d)]
    space = {k: {"min": -32.0, "max": 32.0} for k in keys}
    def f(c):
        xs = [c[k] for k in keys]
        s1 = sum(x * x for x in xs) / d
        s2 = sum(math.cos(2 * PI * x) for x in xs) / d
        return -(-20 * math.exp(-0.2 * math.sqrt(s1)) - math.exp(s2) + 20 + math.e)
    return space, f


def _log_target():
    space = {"lr": {"min": 1e-6, "max": 1.0, "log": True}}
    return space, lambda c: -abs(math.log10(c["lr"]) + 4.5)


def _branch_trap():
    """The branch whose optimum is best has the worst average.

    A wide, sensitive sub-space looks bad until it is tuned. Judging branches
    on mean quality alone abandons it first. This is the shape of choosing
    between a neural net and a logistic regression.
    """
    space = {
        "model": {
            "values": ["narrow", "wide", "safe"],
            "next_level": {
                "narrow": {"pa": {"min": -5.0, "max": 5.0}},
                "wide": {"pb": {"min": -5.0, "max": 5.0},
                         "qb": {"min": -5.0, "max": 5.0}},
                "safe": {"pc": {"min": 1e-5, "max": 1.0, "log": True}},
            },
        },
        "prep": {"values": ["s", "m", "n"]},
    }

    def f(c):
        base = {"s": 0.0, "m": -0.5, "n": -2.0}[c["prep"]]
        if c["model"] == "narrow":
            return base - (c["pa"] - 1.0) ** 2 - 3.0
        if c["model"] == "wide":
            return base - (c["pb"] - 2.0) ** 2 - (c["qb"] + 1.0) ** 2
        return base - abs(math.log10(c["pc"]) + 3.0) - 1.0

    return space, f


def _categorical_mix():
    space = {
        "act": {"values": ["relu", "tanh", "elu", "gelu", "selu", "silu"]},
        "norm": {"values": ["batch", "layer", "none"]},
        "lr": {"min": 1e-5, "max": 1.0, "log": True},
        "wd": {"min": 0.0, "max": 1.0},
    }
    best_act = {"relu": -0.3, "tanh": -1.5, "elu": -0.6, "gelu": 0.0,
                "selu": -1.1, "silu": -0.2}

    def f(c):
        return (best_act[c["act"]]
                - {"batch": 0.0, "layer": 0.4, "none": 1.2}[c["norm"]]
                - abs(math.log10(c["lr"]) + 3.0)
                - 3.0 * (c["wd"] - 0.1) ** 2)

    return space, f


def _integer_mix():
    space = {
        "depth": {"min": 1, "max": 20, "int": True},
        "width": {"min": 8, "max": 512, "int": True},
        "drop": {"min": 0.0, "max": 0.9},
    }

    def f(c):
        return (-((c["depth"] - 6) ** 2) / 4.0
                - ((c["width"] - 128) / 64.0) ** 2
                - 10.0 * (c["drop"] - 0.2) ** 2)

    return space, f


def _plateau():
    """A discrete objective with heavy ties, like accuracy on a small set."""
    space = {"x": {"min": -5.0, "max": 5.0}, "y": {"min": -5.0, "max": 5.0}}

    def f(c):
        exact = -((c["x"] - 2.0) ** 2 + (c["y"] - 2.0) ** 2)
        return math.floor(exact * 2.0) / 2.0

    return space, f


def _noisy_sphere(d, sigma):
    space, clean = _sphere([2.0] * d)
    noise = {"rng": None}

    def f(c):
        return clean(c) + noise["rng"].gauss(0.0, sigma)

    return space, f, clean, noise


_noisy_space, _noisy_f, _noisy_clean, _noisy_state = _noisy_sphere(5, 1.0)

TUNING = {
    "2D quadratic":      (*_quad2d(), 200),
    "5D sphere":         (*_sphere([2.0] * 5), 500),
    "Rosenbrock":        (*_rosenbrock(), 500),
    "Rastrigin 4D":      (*_rastrigin(4), 600),
    "10D sphere":        (*_sphere([1.0] * 10), 1000),
    "Ackley 5D":         (*_ackley(5), 600),
    "Log-scale target":  (*_log_target(), 200),
    "Branch trap":       (*_branch_trap(), 300),
    "Categorical mix":   (*_categorical_mix(), 300),
    "Integer mix":       (*_integer_mix(), 300),
    "Plateau (ties)":    (*_plateau(), 300),
}

#: Objectives whose reported value should be scored on the noise-free function.
NOISY = {
    "Noisy 5D sphere": (_noisy_space, _noisy_f, 400, _noisy_clean, _noisy_state),
}


# --------------------------------------------------------------- held-out set

def _griewank(d):
    keys = [f"x{i}" for i in range(d)]
    space = {k: {"min": -50.0, "max": 50.0} for k in keys}

    def f(c):
        xs = [c[k] for k in keys]
        s = sum(x * x for x in xs) / 4000.0
        p = 1.0
        for i, x in enumerate(xs):
            p *= math.cos(x / math.sqrt(i + 1))
        return -(s - p + 1.0)

    return space, f


def _styblinski(d):
    keys = [f"x{i}" for i in range(d)]
    space = {k: {"min": -5.0, "max": 5.0} for k in keys}
    best = -39.16599 * d

    def f(c):
        v = sum(c[k] ** 4 - 16 * c[k] ** 2 + 5 * c[k] for k in keys) / 2.0
        return -(v - best)

    return space, f


def _nested_pipeline():
    """Three levels of conditioning, the structure the library exists for."""
    space = {
        "family": {
            "values": ["tree", "linear", "net"],
            "next_level": {
                "tree": {
                    "boost": {
                        "values": ["gbm", "rf"],
                        "next_level": {
                            "gbm": {"shrink": {"min": 1e-3, "max": 1.0, "log": True},
                                    "leaves": {"min": 2, "max": 64, "int": True}},
                            "rf": {"trees": {"min": 10, "max": 500, "int": True}},
                        },
                    },
                },
                "linear": {"reg": {"min": 1e-4, "max": 1e2, "log": True}},
                "net": {"width": {"min": 8, "max": 256, "int": True},
                        "eta": {"min": 1e-5, "max": 1e-1, "log": True}},
            },
        },
        "scaler": {"values": ["standard", "robust", "none"]},
    }

    def f(c):
        s = -{"standard": 0.0, "robust": 0.3, "none": 1.0}[c["scaler"]]
        if c["family"] == "linear":
            return s - abs(math.log10(c["reg"]) - 0.5) - 1.5
        if c["family"] == "net":
            return (s - abs(math.log10(c["eta"]) + 3.0)
                    - ((c["width"] - 96) / 48.0) ** 2 - 0.8)
        if c["boost"] == "rf":
            return s - ((c["trees"] - 300) / 150.0) ** 2 - 1.0
        return (s - abs(math.log10(c["shrink"]) + 1.3)
                - ((c["leaves"] - 24) / 12.0) ** 2)

    return space, f


def _mixed_conditional_wide():
    space = {
        "opt": {
            "values": ["adam", "sgd", "lbfgs"],
            "next_level": {
                "adam": {"alr": {"min": 1e-5, "max": 1e-1, "log": True},
                         "b1": {"values": [0.9, 0.95, 0.99]}},
                "sgd": {"slr": {"min": 1e-4, "max": 1.0, "log": True},
                        "mom": {"min": 0.0, "max": 0.99}},
            },
        },
    }

    def f(c):
        if c["opt"] == "lbfgs":
            return -2.0
        if c["opt"] == "adam":
            return -abs(math.log10(c["alr"]) + 3.0) - abs(c["b1"] - 0.95) * 10
        return -abs(math.log10(c["slr"]) + 1.5) - 4.0 * (c["mom"] - 0.9) ** 2

    return space, f


HELDOUT = {
    "Griewank 6D":       (*_griewank(6), 800),
    "Styblinski 4D":     (*_styblinski(4), 600),
    "Nested pipeline":   (*_nested_pipeline(), 400),
    "Optimiser choice":  (*_mixed_conditional_wide(), 300),
    "Rastrigin 8D":      (*_rastrigin(8), 1000),
    "20D sphere":        (*_sphere([1.0] * 20), 1500),
}


# The uniform-random and TPE baselines live in benchmarks/baselines.py.
