"""Drawing configurations from the probability tree.

Two properties of the sampler are load-bearing:

1. **Reflection, not clipping.** Gaussian jitter that is clipped to
   ``[min, max]`` deposits an atom of probability on each bound and biases
   the search toward the edges of every interval. Reflecting the excess back
   inside preserves the density shape.

2. **A structural trace.** Sampling returns both the flat config the user
   sees and a trace mirroring the tree's shape. Updates walk the trace rather
   than matching parameter names against the flat config, so a name appearing
   in two branches can never update the wrong node.
"""

import math

from ._tree import snap_internal, to_external
from ._utils import silverman_bandwidth, softmax, reflect


def sample_node(node, temp, rng, explore=0.0):
    """
    Draw a single value from a probability-tree node.

    With probability ``explore`` the draw ignores the learned distribution and
    samples uniformly from the node's support.

    The floor is *per node*, not per configuration: a config in which every one
    of ``d`` nodes explored has probability ``explore ** d``, which at the
    default and ``d = 10`` is 1e-10. So this bounds how far any single
    parameter's marginal can collapse; it does not make the optimiser a
    superset of random search in high dimensions.

    Args:
        node:    A node dict from the probability tree.
        temp:    Softmax temperature for categorical nodes (> 0).
        rng:     A ``random.Random`` instance.
        explore: Probability of a uniform draw instead of a modelled one.

    Returns:
        ``(external_value, internal_value)``. For categorical nodes the two
        are identical; for continuous nodes the internal value is what the
        update rule records.

    Raises:
        ValueError: If the node has neither 'prob' nor 'reservoir'.
    """
    exploring = explore > 0.0 and rng.random() < explore

    if "prob" in node:
        keys = list(node["prob"])
        if exploring:
            choice = rng.choice(keys)
        else:
            log_probs = [math.log(max(node["prob"][k], 1e-12)) for k in keys]
            choice = rng.choices(keys, weights=softmax(log_probs, temp), k=1)[0]
        return choice, choice

    if "reservoir" in node:
        lo, hi = node["min"], node["max"]
        if exploring:
            x = rng.uniform(lo, hi)
        else:
            centre = rng.choices(node["reservoir"], weights=node["weights"], k=1)[0]
            h = silverman_bandwidth(node["reservoir"], node["weights"], hi - lo)
            x = reflect(centre + rng.gauss(0.0, h), lo, hi)
        x = snap_internal(node, x)
        x = min(max(x, lo), hi)
        return to_external(node, x), x

    raise ValueError(
        f"Malformed node — expected 'prob' (categorical) or 'reservoir' "
        f"(continuous). Got keys: {sorted(map(str, node))}"
    )


def traverse_sample(tree, temp, rng, explore=0.0):
    """
    Recursively sample a full configuration from the probability tree.

    A categorical node carrying a ``next_level`` opens exactly one sub-tree —
    the one matching its sampled choice — so only parameters relevant to that
    branch appear in the config.

    Returns:
        ``(config, trace)``. ``config`` is the flat ``{name: value}`` dict
        passed to the objective. ``trace`` mirrors the tree structure as
        ``{name: {'internal': v, 'sub': trace_or_None}}`` and drives updates.
    """
    config = {}
    trace = {}

    for key, node in tree.items():
        external, internal = sample_node(node, temp, rng, explore)
        config[key] = external
        entry = {"internal": internal, "sub": None}

        if "next_level" in node and external in node["next_level"]:
            sub_config, sub_trace = traverse_sample(
                node["next_level"][external], temp, rng, explore
            )
            config.update(sub_config)
            entry["sub"] = sub_trace

        trace[key] = entry

    return config, trace
