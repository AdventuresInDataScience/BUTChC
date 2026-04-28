import math

KDE_RESERVOIR_SIZE = 50


def softmax(logits, temp=1.0):
    """
    Compute a temperature-scaled softmax over a list of log-space scores.

    Args:
        logits: Unnormalized log-scores (NOT raw probabilities).
        temp:   Temperature > 1 flattens the distribution (more exploration);
                temp < 1 sharpens it (more exploitation).

    Returns:
        List of probabilities that sum to 1.
    """
    scaled = [l / temp for l in logits]
    max_val = max(scaled)
    exp_vals = [math.exp(v - max_val) for v in scaled]
    s = sum(exp_vals)
    return [v / s for v in exp_vals]


def _silverman_bandwidth(reservoir, weights):
    """
    Compute Silverman's rule-of-thumb KDE bandwidth from a weighted reservoir.

    Silverman's rule: h = 1.06 * sigma * n^(-1/5)
    where sigma is the weighted standard deviation and n is the effective
    sample size (1 / sum(w^2)).

    Args:
        reservoir: List of float sample values.
        weights:   Corresponding normalised weights (must sum to 1).

    Returns:
        Scalar bandwidth h > 0.
    """
    n = len(reservoir)
    if n < 2:
        return 1.0

    w_mean = sum(w * x for w, x in zip(weights, reservoir))
    w_var = sum(w * (x - w_mean) ** 2 for w, x in zip(weights, reservoir))
    w_std = math.sqrt(max(w_var, 1e-12))

    n_eff = 1.0 / max(sum(w ** 2 for w in weights), 1e-12)
    return 1.06 * w_std * (n_eff ** (-0.2))


def _compute_trial_loss(deltas):
    """
    Aggregate per-node deltas from a single trial into a scalar loss value.

    Loss measures model surprise — the mean magnitude of probability updates
    across all nodes touched. High loss = model was far from expecting this
    config's outcome. Near-zero loss = tree has effectively stopped updating.

    Args:
        deltas: List of normalised per-node deltas returned by _update_tree.

    Returns:
        Scalar mean loss for this trial, or 0.0 if no updates were made.
    """
    if not deltas:
        return 0.0
    return sum(deltas) / len(deltas)
