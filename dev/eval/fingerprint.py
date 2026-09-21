"""Hash a spread of seeded runs, so a refactor can be proved behaviour-neutral.

    python dev/eval/fingerprint.py            # sequential baseline
    python dev/eval/fingerprint.py batch=1    # must match the above

Run before and after any change that claims not to alter search behaviour. A
matching hash is the evidence; a passing test suite is not, since the suite
checks properties rather than exact trajectories.

Covers categorical, continuous, log, int, nested conditional, priors, warm
starts and NaN objectives — anything whose sampling or update path could shift.
"""

import hashlib
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir)))

from butchc import BUTChC_optimize  # noqa: E402

SPACE_FLAT = {
    "x": {"min": -5.0, "max": 5.0},
    "y": {"min": -5.0, "max": 5.0},
    "k": {"values": ["a", "b", "c"]},
}

SPACE_NESTED = {
    "model": {
        "values": ["svm", "nn"],
        "next_level": {
            "svm": {"C": {"min": 1e-3, "max": 1e3, "log": True}},
            "nn": {
                "depth": {"min": 1, "max": 8, "int": True},
                "act": {"values": ["relu", "tanh"]},
            },
        },
    },
    "scale": {"values": [0.1, 1.0, 10.0]},
}

SPACE_PRIOR = {
    "opt": {
        "values": ["adam", "sgd"],
        "prior": {"adam": 0.8, "sgd": 0.2},
        "prior_strength": 25,
    },
    "lr": {
        "min": 1e-5,
        "max": 1e-1,
        "log": True,
        "prior": {"mean": 1e-3, "std": 5e-4},
        "prior_strength": 15,
    },
}


def f_flat(c):
    bonus = {"a": 0.0, "b": 0.5, "c": -0.5}[c["k"]]
    return -(c["x"] ** 2 + c["y"] ** 2) + bonus


def f_nested(c):
    if c["model"] == "svm":
        base = -abs(math.log10(c["C"]) - 1.0)
    else:
        base = -abs(c["depth"] - 4) - (0.0 if c["act"] == "relu" else 1.0)
    return base * c["scale"] / 10.0


def f_prior(c):
    target = -3.0 if c["opt"] == "adam" else -2.0
    return -abs(math.log10(c["lr"]) - target)


def f_nan(c):
    return float("nan") if c["x"] > 0 else -(c["x"] ** 2)


def summarise(res):
    return {
        "best_value": round(res["best_value"], 12),
        "best_params": {
            k: (round(v, 12) if isinstance(v, float) else v)
            for k, v in sorted((res["best_params"] or {}).items())
        },
        "objectives": [round(h["objective"], 12) if math.isfinite(h["objective"])
                       else "nan" for h in res["history"]],
        "losses": [round(h["loss"], 12) for h in res["history"]],
        "n_updates": res["n_updates"],
        "n_gated": res["n_gated"],
    }


def main(extra=None):
    extra = extra or {}
    records = []

    for seed in range(5):
        records.append(summarise(BUTChC_optimize(
            SPACE_FLAT, f_flat, 120, seed=seed, verbose=False, **extra)))
        records.append(summarise(BUTChC_optimize(
            SPACE_NESTED, f_nested, 100, seed=seed, verbose=False, **extra)))
        records.append(summarise(BUTChC_optimize(
            SPACE_PRIOR, f_prior, 80, seed=seed, verbose=False, **extra)))

    # non-default knobs
    records.append(summarise(BUTChC_optimize(
        SPACE_FLAT, f_flat, 100, seed=7, verbose=False,
        lambda_=0.5, alpha=2.0, temp=1.5, gamma=0.3, explore=0.2,
        n_warmup=10, **extra)))

    # NaN handling
    records.append(summarise(BUTChC_optimize(
        {"x": {"min": -3.0, "max": 3.0}}, f_nan, 60, seed=3, verbose=False,
        **extra)))

    # warm-start chain
    first = BUTChC_optimize(SPACE_NESTED, f_nested, 60, seed=11, verbose=False,
                            **extra)
    second = BUTChC_optimize(SPACE_NESTED, f_nested, 60, seed=12, verbose=False,
                             start_prob_tree=first["prob_tree"], **extra)
    records.append(summarise(second))

    blob = json.dumps(records, sort_keys=True).encode()
    print(hashlib.sha256(blob).hexdigest())


if __name__ == "__main__":
    kw = {}
    for arg in sys.argv[1:]:
        key, _, val = arg.partition("=")
        kw[key] = int(val)
    main(kw)
