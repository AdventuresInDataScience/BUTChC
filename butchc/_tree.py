from ._utils import KDE_RESERVOIR_SIZE


def _initialize_prob_tree(searchspace):
    """
    Recursively build a probability tree from a search-space definition.

    Each node takes one of two forms:
      • Categorical: {'counts': {...}, 'prob': {...}, ['next_level': {...}]}
      • Continuous:  {'min': float, 'max': float,
                      'reservoir': [float, ...], 'weights': [float, ...]}

    Continuous nodes are initialised with a uniform reservoir of
    KDE_RESERVOIR_SIZE evenly-spaced points across [min, max], each with
    equal weight — a flat, uninformative prior.

    Args:
        searchspace: Dict mapping parameter names to their definitions.

    Returns:
        Nested dict representing the initialised probability tree.
    """
    tree = {}
    for key, val in searchspace.items():
        node = {}

        if isinstance(val, dict) and "values" in val:
            node['counts'] = {k: 1 for k in val['values']}
            node['prob']   = {k: 1 / len(val['values']) for k in val['values']}

            if 'next_level' in val:
                node['next_level'] = {
                    choice: _initialize_prob_tree(sub_space)
                    for choice, sub_space in val['next_level'].items()
                }

        elif isinstance(val, dict) and 'min' in val and 'max' in val:
            lo, hi = val['min'], val['max']
            node['min'] = lo
            node['max'] = hi

            n = KDE_RESERVOIR_SIZE
            if n > 1:
                step = (hi - lo) / (n - 1)
                reservoir = [lo + i * step for i in range(n)]
            else:
                reservoir = [(lo + hi) / 2]
            uniform_w = 1.0 / len(reservoir)

            node['reservoir']    = reservoir
            node['weights']      = [uniform_w] * len(reservoir)
            node['total_weight'] = 1.0

        tree[key] = node

    return tree
