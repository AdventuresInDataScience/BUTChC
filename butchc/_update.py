from ._utils import KDE_RESERVOIR_SIZE


def _update_tree(tree, config, lambda_, alpha):
    """
    Update the probability tree in-place after observing a configuration.
    Returns a list of normalised per-node deltas for loss computation.

    Categorical: Laplace-smoothed count update.
        count[chosen] += lambda_
        prob[k] = (count[k] + alpha) / (sum(counts) + alpha * n_choices)

    Continuous (KDE reservoir):
        1. Append observed value with weight lambda_
        2. Prune lowest-weight point if over KDE_RESERVOIR_SIZE
        3. Renormalise all weights to sum to 1

    Args:
        tree:    Probability tree to update in-place.
        config:  The sampled configuration that was just evaluated.
        lambda_: Weight applied to the new observation (> 0).
        alpha:   Laplace smoothing strength for categorical nodes (> 0).

    Returns:
        List of normalised scalar deltas in [0, 1], one per updated node.
    """
    deltas = []

    for key, node in tree.items():
        if key not in config:
            continue

        val = config[key]

        if 'prob' in node:
            probs_before = dict(node['prob'])

            node['counts'][val] += lambda_
            total = sum(node['counts'].values()) + alpha * len(node['counts'])
            for k in node['counts']:
                node['prob'][k] = (node['counts'][k] + alpha) / total

            n_choices = len(node['prob'])
            delta = sum(
                abs(node['prob'][k] - probs_before[k]) for k in node['prob']
            ) / n_choices
            deltas.append(delta)

            if 'next_level' in node and val in node['next_level']:
                sub_deltas = _update_tree(node['next_level'][val], config, lambda_, alpha)
                deltas.extend(sub_deltas)

        elif 'reservoir' in node:
            w_mean_before = sum(
                w * x for w, x in zip(node['weights'], node['reservoir'])
            )

            node['reservoir'].append(val)
            node['weights'].append(lambda_)

            if len(node['reservoir']) > KDE_RESERVOIR_SIZE:
                min_idx = node['weights'].index(min(node['weights']))
                node['reservoir'].pop(min_idx)
                node['weights'].pop(min_idx)

            total_w = sum(node['weights'])
            node['weights'] = [w / total_w for w in node['weights']]

            w_mean_after = sum(
                w * x for w, x in zip(node['weights'], node['reservoir'])
            )
            node_range = node['max'] - node['min']
            delta = (
                abs(w_mean_after - w_mean_before) / node_range
                if node_range > 0 else 0.0
            )
            deltas.append(delta)

    return deltas
