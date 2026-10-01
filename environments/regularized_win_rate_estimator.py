from collections import deque


class RegularizedWinRateEstimator:
    """
    Rolling Bernoulli estimator with Beta(alpha, beta) smoothing.

    Posterior mean:

        p_hat = (alpha + wins)
                / (alpha + beta + n)

    Only the most recent `window_size` outcomes are retained.
    """

    def __init__(
        self,
        window_size=20,
        alpha=20.0,
        beta=20.0,
    ):

        if window_size <= 0:
            raise ValueError(
                "window_size must be positive"
            )

        if alpha <= 0 or beta <= 0:
            raise ValueError(
                "alpha and beta must be positive"
            )

        self.window_size = int(
            window_size
        )

        self.alpha = float(
            alpha
        )

        self.beta = float(
            beta
        )

        self.outcomes = deque(
            maxlen=self.window_size
        )


    def reset(self):

        self.outcomes.clear()


    def update(
        self,
        win,
    ):

        self.outcomes.append(
            1.0 if bool(win) else 0.0
        )

        return self.estimate()


    def estimate(self):

        wins = sum(
            self.outcomes
        )

        n = len(
            self.outcomes
        )

        return (
            self.alpha + wins
        ) / (
            self.alpha
            + self.beta
            + n
        )


    def sample_count(self):

        return len(
            self.outcomes
        )
