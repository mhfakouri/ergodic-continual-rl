import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from environments.coin_toss import CoinTossEnv


SEED = 0
TOTAL_TIMESTEPS = 1_000_000

INITIAL_WEALTH = 100.0
EPISODE_LENGTH = 100

N_EVAL_TRAJECTORIES = 5_000
EVAL_SEED = 12345

MODEL_DIR = "models"
RESULT_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


def expected_log_growth(f):
    return (
        0.5 * np.log(1.0 + 0.5 * f)
        + 0.5 * np.log(1.0 - 0.4 * f)
    )


grid = np.linspace(0.0, 1.0, 10001)

grid_values = np.array(
    [expected_log_growth(f) for f in grid]
)

grid_optimum = grid[
    np.argmax(grid_values)
]


print("=" * 70)
print("ERGODIC RL COIN-TOSS EXPERIMENT")
print("=" * 70)

print()
print("Analytical log-growth optimum:", grid_optimum)
print(
    "Expected log growth at F=0.25:",
    expected_log_growth(0.25),
)
print(
    "Expected log growth at F=1.00:",
    expected_log_growth(1.0),
)


def load_or_train(objective):

    model_path = os.path.join(
        MODEL_DIR,
        f"coin_toss_{objective}_ppo",
    )

    zip_path = model_path + ".zip"

    env = CoinTossEnv(
        objective=objective,
        initial_wealth=INITIAL_WEALTH,
        episode_length=EPISODE_LENGTH,
    )

    if os.path.exists(zip_path):

        print()
        print(
            f"Loading existing {objective} model..."
        )

        model = PPO.load(
            model_path,
            env=env,
            device="cpu",
        )

    else:

        print()
        print(
            f"Training {objective} model..."
        )

        model = PPO(
            "MlpPolicy",
            env,
            seed=SEED,
            verbose=0,
            device="cpu",
        )

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS
        )

        model.save(model_path)

        print(
            "Saved:",
            zip_path,
        )

    return model


standard_model = load_or_train(
    "standard"
)

log_model = load_or_train(
    "log"
)


def model_fraction(model, wealth):

    obs = np.array(
        [[
            np.log(
                max(wealth, 1e-30)
                / INITIAL_WEALTH
            )
        ]],
        dtype=np.float32,
    )

    action, _ = model.predict(
        obs,
        deterministic=True,
    )

    a = float(
        np.asarray(action).reshape(-1)[0]
    )

    return 0.5 * (
        np.clip(a, -1.0, 1.0)
        + 1.0
    )


print()
print("=" * 70)
print("LEARNED POLICY BY WEALTH")
print("=" * 70)

for wealth in [
    10.0,
    50.0,
    100.0,
    200.0,
    1000.0,
]:

    f_standard = model_fraction(
        standard_model,
        wealth,
    )

    f_log = model_fraction(
        log_model,
        wealth,
    )

    print()
    print(f"Wealth = {wealth:.1f}")
    print(
        f"  Standard PPO F = {f_standard:.6f}"
    )
    print(
        f"  Log PPO F      = {f_log:.6f}"
    )


def simulate_model(
    model,
    n_trajectories,
    seed,
):

    rng = np.random.default_rng(seed)

    wealth = np.full(
        n_trajectories,
        INITIAL_WEALTH,
        dtype=np.float64,
    )

    for _ in range(EPISODE_LENGTH):

        obs = np.log(
            np.maximum(
                wealth,
                1e-30,
            )
            / INITIAL_WEALTH
        ).reshape(-1, 1)

        obs = obs.astype(
            np.float32
        )

        actions, _ = model.predict(
            obs,
            deterministic=True,
        )

        actions = np.asarray(
            actions
        ).reshape(-1)

        fractions = 0.5 * (
            np.clip(
                actions,
                -1.0,
                1.0,
            )
            + 1.0
        )

        wins = rng.integers(
            0,
            2,
            size=n_trajectories,
        )

        multipliers = np.where(
            wins == 1,
            1.0 + 0.5 * fractions,
            1.0 - 0.4 * fractions,
        )

        wealth *= multipliers

    return wealth


def simulate_fixed_fraction(
    fraction,
    n_trajectories,
    seed,
):

    rng = np.random.default_rng(seed)

    wealth = np.full(
        n_trajectories,
        INITIAL_WEALTH,
        dtype=np.float64,
    )

    for _ in range(EPISODE_LENGTH):

        wins = rng.integers(
            0,
            2,
            size=n_trajectories,
        )

        multiplier = np.where(
            wins == 1,
            1.0 + 0.5 * fraction,
            1.0 - 0.4 * fraction,
        )

        wealth *= multiplier

    return wealth


def summarize(name, final_wealth):

    log_growth = (
        np.log(
            final_wealth
            / INITIAL_WEALTH
        )
        / EPISODE_LENGTH
    )

    return {
        "Policy": name,
        "Mean final wealth":
            np.mean(final_wealth),
        "Median final wealth":
            np.median(final_wealth),
        "Mean log growth":
            np.mean(log_growth),
        "Probability below start":
            np.mean(
                final_wealth
                < INITIAL_WEALTH
            ),
    }


print()
print("=" * 70)
print("MONTE CARLO EVALUATION")
print("=" * 70)


wealth_standard = simulate_model(
    standard_model,
    N_EVAL_TRAJECTORIES,
    EVAL_SEED,
)

wealth_log = simulate_model(
    log_model,
    N_EVAL_TRAJECTORIES,
    EVAL_SEED,
)

wealth_optimal = simulate_fixed_fraction(
    0.25,
    N_EVAL_TRAJECTORIES,
    EVAL_SEED,
)

wealth_all_in = simulate_fixed_fraction(
    1.0,
    N_EVAL_TRAJECTORIES,
    EVAL_SEED,
)


results = pd.DataFrame(
    [
        summarize(
            "Standard PPO",
            wealth_standard,
        ),
        summarize(
            "Log PPO",
            wealth_log,
        ),
        summarize(
            "Analytical log optimum F=0.25",
            wealth_optimal,
        ),
        summarize(
            "Expected-wealth extreme F=1",
            wealth_all_in,
        ),
    ]
)


print()
print(
    results.to_string(
        index=False
    )
)


output_path = os.path.join(
    RESULT_DIR,
    "coin_toss_summary.csv",
)

results.to_csv(
    output_path,
    index=False,
)


print()
print(
    "Saved evaluation:",
    output_path,
)

print()
print("=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)
