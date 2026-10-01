from collections import deque


class RollingWinRateEstimator:
    """
    Online estimator of p(win) using a rolling binary window.

    Each observation must be:
        True / 1  -> win
        False / 0 -> loss
    """

    def __init__(
        self,
        window_size=20,
        initial_estimate=0.5,
    ):

        if window_size <= 0:
            raise ValueError(
                "window_size must be positive"
            )

        if not 0.0 <= initial_estimate <= 1.0:
            raise ValueError(
                "initial_estimate must be between 0 and 1"
            )

        self.window_size = int(
            window_size
        )

        self.initial_estimate = float(
            initial_estimate
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

        value = 1.0 if bool(win) else 0.0

        self.outcomes.append(
            value
        )

        return self.estimate()


    def estimate(self):

        if len(
            self.outcomes
        ) == 0:

            return (
                self.initial_estimate
            )

        return (
            sum(
                self.outcomes
            )
            / len(
                self.outcomes
            )
        )


    def sample_count(self):

        return len(
            self.outcomes
        )
