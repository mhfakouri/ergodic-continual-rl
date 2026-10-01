import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CoinTossEnv(gym.Env):
    """
    Multiplicative coin-toss environment.

    PPO action:
        a in [-1, 1]

    Bet fraction:
        F = (a + 1) / 2 in [0, 1]

    Objectives:
        standard -> change in wealth
        log      -> change in log wealth

    Optional deployment shift:
        p_win_before -> p_win_after
        starting at shift_step.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        objective="standard",
        initial_wealth=100.0,
        episode_length=100,
        p_win=0.5,
        shift_step=None,
        p_win_after=None,
    ):
        super().__init__()

        if objective not in (
            "standard",
            "log",
        ):
            raise ValueError(
                "objective must be 'standard' or 'log'"
            )

        if not 0.0 <= p_win <= 1.0:
            raise ValueError(
                "p_win must be between 0 and 1"
            )

        if p_win_after is not None:
            if not 0.0 <= p_win_after <= 1.0:
                raise ValueError(
                    "p_win_after must be between 0 and 1"
                )

        if shift_step is not None:
            if shift_step < 0:
                raise ValueError(
                    "shift_step must be non-negative"
                )

        self.objective = objective

        self.initial_wealth = float(
            initial_wealth
        )

        self.episode_length = int(
            episode_length
        )

        # ----------------------------------------------------
        # Deployment dynamics
        # ----------------------------------------------------

        self.p_win = float(
            p_win
        )

        self.shift_step = (
            shift_step
        )

        if p_win_after is None:
            self.p_win_after = (
                self.p_win
            )
        else:
            self.p_win_after = float(
                p_win_after
            )

        # ----------------------------------------------------
        # Spaces
        # ----------------------------------------------------

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

        self.wealth = (
            self.initial_wealth
        )

        self.steps = 0


    # ========================================================
    # Action conversion
    # ========================================================

    @staticmethod
    def action_to_fraction(
        action
    ):

        a = float(
            np.asarray(
                action
            ).reshape(-1)[0]
        )

        a = np.clip(
            a,
            -1.0,
            1.0,
        )

        return 0.5 * (
            a + 1.0
        )


    # ========================================================
    # Current deployment regime
    # ========================================================

    def current_p_win(
        self
    ):

        if (
            self.shift_step
            is not None
            and self.steps
            >= self.shift_step
        ):
            return (
                self.p_win_after
            )

        return self.p_win


    # ========================================================
    # Observation
    # ========================================================

    def _observation(
        self
    ):

        value = np.log(
            max(
                self.wealth,
                1e-30,
            )
            / self.initial_wealth
        )

        return np.array(
            [value],
            dtype=np.float32,
        )


    # ========================================================
    # Reset
    # ========================================================

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
                    ),

                "p_win":
                    self.current_p_win(),

                "step":
                    self.steps,
            },
        )


    # ========================================================
    # Step
    # ========================================================

    def step(
        self,
        action,
    ):

        fraction = (
            self.action_to_fraction(
                action
            )
        )

        old_wealth = (
            self.wealth
        )

        # Probability used for THIS transition.
        active_p_win = (
            self.current_p_win()
        )

        # Bernoulli outcome.
        win = (
            self.np_random.random()
            < active_p_win
        )

        if win:

            multiplier = (
                1.0
                + 0.5
                * fraction
            )

        else:

            multiplier = (
                1.0
                - 0.4
                * fraction
            )

        self.wealth = max(
            old_wealth
            * multiplier,
            1e-30,
        )

        # ----------------------------------------------------
        # Reward
        # ----------------------------------------------------

        if (
            self.objective
            == "standard"
        ):

            reward = (
                self.wealth
                - old_wealth
            )

        else:

            reward = (
                np.log(
                    self.wealth
                )
                - np.log(
                    old_wealth
                )
            )

        self.steps += 1

        terminated = False

        truncated = (
            self.steps
            >= self.episode_length
        )

        info = {
            "wealth":
                float(
                    self.wealth
                ),

            "bet_fraction":
                fraction,

            "p_win":
                active_p_win,

            "win":
                bool(
                    win
                ),

            "step":
                self.steps,

            "regime":
                (
                    "post-shift"
                    if (
                        self.shift_step
                        is not None
                        and self.steps
                        > self.shift_step
                    )
                    else "pre-shift"
                ),
        }

        return (
            self._observation(),
            float(
                reward
            ),
            terminated,
            truncated,
            info,
        )
