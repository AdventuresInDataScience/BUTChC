import math
import random

from ._utils import softmax, _silverman_bandwidth


def _sample_node(node, temp):
    """
    Draw a single value from a probability-tree node.

    Categorical: temperature-scaled softmax sampling over stored probabilities.
    Continuous: KDE draw — pick a reservoir point weighted by history, add
    Silverman-bandwidth Gaussian jitter, clip to [min, max].

    Args:
        node: A single node dict from the probability tree.
        temp: Softmax temperature for categorical nodes (must be > 0).

    Returns:
        The sampled value (str for categorical, float for continuous).

    Raises:
        ValueError: If the node dict has neither 'prob' nor 'reservoir'.
    """
    if 'prob' in node:
        keys     = list(node['prob'].keys())
        probs    = [node['prob'][k] for k in keys]
        log_probs = [math.log(max(p, 1e-12)) for p in probs]
        weights  = softmax(log_probs, temp)
        return random.choices(keys, weights=weights, k=1)[0]

    elif 'reservoir' in node:
        reservoir = node['reservoir']
        weights   = node['weights']
        centre    = random.choices(reservoir, weights=weights, k=1)[0]
        h         = _silverman_bandwidth(reservoir, weights)
        jitter    = random.gauss(0, h)
        return max(node['min'], min(node['max'], centre + jitter))

    else:
        raise ValueError(
            f"Malformed node — expected 'prob' (categorical) or "
            f"'reservoir' (continuous). Got keys: {list(node.keys())}"
        )


def _traverse_sample(tree, temp):
    """
    Recursively sample a full configuration from the probability tree.

    When a categorical parameter has a 'next_level', the sampled choice
    determines which conditional sub-tree is explored, so only parameters
    relevant to that branch are included in the returned config.

    Args:
        tree: A (sub-)tree dict as produced by _initialize_prob_tree.
        temp: Softmax temperature forwarded to _sample_node.

    Returns:
        Flat dict of {parameter_name: sampled_value} for the whole (sub-)tree.
    """
    config = {}
    for key, node in tree.items():
        val = _sample_node(node, temp)
        config[key] = val

        if 'next_level' in node and val in node['next_level']:
            sub_config = _traverse_sample(node['next_level'][val], temp)
            config.update(sub_config)

    return config
