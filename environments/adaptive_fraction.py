import numpy as np


def log_optimal_fraction(
    p_win,
):
    """
    Compute the log-growth-optimal bet fraction

        F*(p) = clip(4.5*p - 2, 0, 1)

    for the current coin-toss dynamics.
    """

    p_win = float(
        p_win
    )

    if not 0.0 <= p_win <= 1.0:
        raise ValueError(
            "p_win must be between 0 and 1"
        )

    fraction = (
        4.5 * p_win
        - 2.0
    )

    return float(
        np.clip(
            fraction,
            0.0,
            1.0,
        )
    )
