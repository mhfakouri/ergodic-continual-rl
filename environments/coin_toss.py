import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CoinTossEnv(gym.Env):
    """
    Non-ergodic multiplicative coin-toss environment.

    PPO action:
        a in [-1, 1]

    Bet fraction:
        F = (a + 1) / 2 in [0, 1]

    Objectives:
        standard -> change in wealth
        log      -> change in log wealth
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        objective="standard",
        initial_wealth=100.0,
        episode_length=100,
    ):
        super().__init__()

        if objective not in ("standard", "log"):
            raise ValueError(
                "objective must be 'standard' or 'log'"
            )

        self.objective = objective
        self.initial_wealth = float(initial_wealth)
        self.episode_length = int(episode_length)

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

        a = np.clip(a, -1.0, 1.0)

        return 0.5 * (a + 1.0)

    def _observation(self):
        value = np.log(
            max(self.wealth, 1e-30)
            / self.initial_wealth
        )

        return np.array(
            [value],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.wealth = self.initial_wealth
        self.steps = 0

        return (
            self._observation(),
            {"wealth": float(self.wealth)},
        )

    def step(self, action):
        fraction = self.action_to_fraction(action)

        old_wealth = self.wealth

        win = self.np_random.integers(0, 2)

        if win == 1:
            multiplier = 1.0 + 0.5 * fraction
        else:
            multiplier = 1.0 - 0.4 * fraction

        self.wealth = max(
            old_wealth * multiplier,
            1e-30,
        )

        if self.objective == "standard":
            reward = self.wealth - old_wealth
        else:
            reward = (
                np.log(self.wealth)
                - np.log(old_wealth)
            )

        self.steps += 1

        terminated = False
        truncated = (
            self.steps >= self.episode_length
        )

        return (
            self._observation(),
            float(reward),
            terminated,
            truncated,
            {"wealth": float(self.wealth)},
        )
