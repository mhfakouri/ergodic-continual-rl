import numpy as np
import gymnasium as gym
from gymnasium import spaces


class LearnedTransformCoinTossEnv(gym.Env):
    """
    Coin-toss environment using a learned ergodicity transformation h.

    Observation:
        h(W_t) - h(W_0)

    Reward:
        h(W_{t+1}) - h(W_t)

    PPO action:
        a in [-1, 1]

    Bet fraction:
        F = (a + 1) / 2
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        transform,
        initial_wealth=100.0,
        episode_length=100,
    ):
        super().__init__()

        self.transform = transform
        self.initial_wealth = float(initial_wealth)
        self.episode_length = int(episode_length)

        self.transform_baseline = float(
            self.transform(
                self.initial_wealth
            )
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1,),
            dtype=np.float32,
        )

        self.wealth = self.initial_wealth
        self.steps = 0

    @staticmethod
    def action_to_fraction(action):

        a = float(
            np.asarray(action).reshape(-1)[0]
        )

        a = np.clip(
            a,
            -1.0,
            1.0,
        )

        return 0.5 * (
            a + 1.0
        )

    def _transform(self, wealth):

        return float(
            self.transform(
                wealth
            )
        )

    def _observation(self):

        transformed = (
            self._transform(
                self.wealth
            )
            - self.transform_baseline
        )

        return np.array(
            [transformed],
            dtype=np.float32,
        )

    def reset(
        self,
        seed=None,
        options=None,
    ):

        super().reset(
            seed=seed
        )

        self.wealth = (
            self.initial_wealth
        )

        self.steps = 0

        return (
            self._observation(),
            {
                "wealth":
                    float(
                        self.wealth
                    )
            },
        )

    def step(self, action):

        fraction = (
            self.action_to_fraction(
                action
            )
        )

        old_wealth = (
            self.wealth
        )

        old_transformed = (
            self._transform(
                old_wealth
            )
        )

        win = (
            self.np_random.integers(
                0,
                2,
            )
        )

        if win == 1:

            multiplier = (
                1.0
                + 0.5 * fraction
            )

        else:

            multiplier = (
                1.0
                - 0.4 * fraction
            )

        self.wealth = max(
            old_wealth
            * multiplier,
            1e-30,
        )

        new_transformed = (
            self._transform(
                self.wealth
            )
        )

        reward = (
            new_transformed
            - old_transformed
        )

        self.steps += 1

        terminated = False

        truncated = (
            self.steps
            >= self.episode_length
        )

        observation = np.array(
            [
                new_transformed
                - self.transform_baseline
            ],
            dtype=np.float32,
        )

        info = {
            "wealth":
                float(
                    self.wealth
                ),
            "bet_fraction":
                fraction,
        }

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info,
        )
