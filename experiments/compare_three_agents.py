import os
import numpy as np
import pandas as pd

from stable_baselines3 import PPO

from environments.coin_toss import CoinTossEnv
from environments.ergodicity_transform import (
    learn_ergodicity_transform,
)
from environments.fast_transform import (
    make_fast_transform,
)


# ============================================================
# Configuration
# ============================================================

INITIAL_WEALTH = 100.0
EPISODE_LENGTH = 100

N_TRAJECTORIES = 5000
EVAL_SEED = 12345

MODEL_DIR = "models"
RESULT_DIR = "results"

os.makedirs(
    RESULT_DIR,
    exist_ok=True,
)


# ============================================================
# Reconstruct learned transformation
# ============================================================

source_env = CoinTossEnv(
    objective="standard",
    initial_wealth=INITIAL_WEALTH,
    episode_length=EPISODE_LENGTH,
)

_, info = source_env.reset(
    seed=7
)

wealth_history = [
    info["wealth"]
]

action = np.array(
    [1.0],
    dtype=np.float32,
)

for _ in range(EPISODE_LENGTH):

    _, _, _, _, info = source_env.step(
        action
    )

    wealth_history.append(
        info["wealth"]
    )


wealth_history = np.asarray(
    wealth_history,
    dtype=np.float64,
)


exact_transform = learn_ergodicity_transform(
    wealth_history,
    loess_span=0.2,
    min_integrate_range=0.01,
)


learned_transform = make_fast_transform(
    exact_transform,
    min_value=1e-3,
    max_value=20000.0,
    n_points=800,
)


learned_baseline = learned_transform(
    INITIAL_WEALTH
)


# ============================================================
# Load models
# ============================================================

print("=" * 78)
print("LOADING MODELS")
print("=" * 78)


standard_model = PPO.load(
    os.path.join(
        MODEL_DIR,
        "coin_toss_standard_ppo",
    ),
    device="cpu",
)


log_model = PPO.load(
    os.path.join(
        MODEL_DIR,
        "coin_toss_log_ppo",
    ),
    device="cpu",
)


learned_model = PPO.load(
    os.path.join(
        MODEL_DIR,
        "coin_toss_learned_transform_ppo",
    ),
    device="cpu",
)


print("Standard PPO loaded")
print("Log PPO loaded")
print("Learned-transform PPO loaded")


# ============================================================
# Observation functions
# ============================================================

def log_observation(wealth):

    return np.log(
        np.maximum(
            wealth,
            1e-30,
        )
        / INITIAL_WEALTH
    )


def learned_observation(wealth):

    values = np.empty_like(
        wealth,
        dtype=np.float64,
    )

    for i, w in enumerate(wealth):

        values[i] = (
            learned_transform(w)
            - learned_baseline
        )

    return values


# ============================================================
# Model action -> fraction
# ============================================================

def predict_fraction(
    model,
    observation,
):

    observation = np.asarray(
        observation,
        dtype=np.float32,
    ).reshape(-1, 1)

    action, _ = model.predict(
        observation,
        deterministic=True,
    )

    action = np.asarray(
        action
    ).reshape(-1)

    action = np.clip(
        action,
        -1.0,
        1.0,
    )

    return 0.5 * (
        action + 1.0
    )


# ============================================================
# Generate common random outcomes
# ============================================================

rng = np.random.default_rng(
    EVAL_SEED
)


coin_outcomes = rng.integers(
    0,
    2,
    size=(
        EPISODE_LENGTH,
        N_TRAJECTORIES,
    ),
)


# ============================================================
# Simulate learned policies
# ============================================================

def simulate_policy(
    model,
    observation_function,
):

    wealth = np.full(
        N_TRAJECTORIES,
        INITIAL_WEALTH,
        dtype=np.float64,
    )

    fractions_history = []

    for step in range(
        EPISODE_LENGTH
    ):

        obs = observation_function(
            wealth
        )

        fractions = predict_fraction(
            model,
            obs,
        )

        fractions_history.append(
            fractions.copy()
        )

        wins = coin_outcomes[
            step
        ]

        multipliers = np.where(
            wins == 1,
            1.0
            + 0.5 * fractions,
            1.0
            - 0.4 * fractions,
        )

        wealth *= multipliers

    return (
        wealth,
        np.asarray(
            fractions_history
        ),
    )


# ============================================================
# Fixed-fraction policy
# ============================================================

def simulate_fixed(
    fraction,
):

    wealth = np.full(
        N_TRAJECTORIES,
        INITIAL_WEALTH,
        dtype=np.float64,
    )

    for step in range(
        EPISODE_LENGTH
    ):

        wins = coin_outcomes[
            step
        ]

        multiplier = np.where(
            wins == 1,
            1.0
            + 0.5 * fraction,
            1.0
            - 0.4 * fraction,
        )

        wealth *= multiplier

    return wealth


# ============================================================
# Run evaluation
# ============================================================

print()
print("=" * 78)
print("RUNNING COMMON-RANDOM-NUMBER EVALUATION")
print("=" * 78)


standard_wealth, standard_fractions = (
    simulate_policy(
        standard_model,
        log_observation,
    )
)


log_wealth, log_fractions = (
    simulate_policy(
        log_model,
        log_observation,
    )
)


learned_wealth, learned_fractions = (
    simulate_policy(
        learned_model,
        learned_observation,
    )
)


optimal_wealth = simulate_fixed(
    0.25
)


all_in_wealth = simulate_fixed(
    1.0
)


# ============================================================
# Metrics
# ============================================================

def summarize(
    name,
    final_wealth,
    fractions=None,
):

    log_growth = (
        np.log(
            final_wealth
            / INITIAL_WEALTH
        )
        / EPISODE_LENGTH
    )

    result = {
        "Policy":
            name,

        "Mean final wealth":
            np.mean(
                final_wealth
            ),

        "Median final wealth":
            np.median(
                final_wealth
            ),

        "Mean log growth":
            np.mean(
                log_growth
            ),

        "P(final wealth < 100)":
            np.mean(
                final_wealth
                < INITIAL_WEALTH
            ),
    }

    if fractions is not None:

        result[
            "Mean bet fraction"
        ] = np.mean(
            fractions
        )

    else:

        result[
            "Mean bet fraction"
        ] = np.nan

    return result


rows = [

    summarize(
        "Standard PPO",
        standard_wealth,
        standard_fractions,
    ),

    summarize(
        "Log PPO",
        log_wealth,
        log_fractions,
    ),

    summarize(
        "Learned-transform PPO",
        learned_wealth,
        learned_fractions,
    ),

    summarize(
        "Analytical optimum F=0.25",
        optimal_wealth,
    ),

    summarize(
        "Expected-wealth extreme F=1",
        all_in_wealth,
    ),
]


results = pd.DataFrame(
    rows
)


# ============================================================
# Print
# ============================================================

print()
print("=" * 78)
print("THREE-AGENT COMPARISON")
print("=" * 78)

print()

print(
    results.to_string(
        index=False
    )
)


# ============================================================
# Initial decisions
# ============================================================

initial_wealth_array = np.array(
    [INITIAL_WEALTH],
    dtype=np.float64,
)


standard_initial = (
    predict_fraction(
        standard_model,
        log_observation(
            initial_wealth_array
        ),
    )[0]
)


log_initial = (
    predict_fraction(
        log_model,
        log_observation(
            initial_wealth_array
        ),
    )[0]
)


learned_initial = (
    predict_fraction(
        learned_model,
        learned_observation(
            initial_wealth_array
        ),
    )[0]
)


print()
print("=" * 78)
print("INITIAL-STATE POLICY")
print("=" * 78)

print(
    "Standard PPO:",
    standard_initial,
)

print(
    "Log PPO:",
    log_initial,
)

print(
    "Learned-transform PPO:",
    learned_initial,
)

print(
    "Analytical optimum:",
    0.25,
)


# ============================================================
# Save
# ============================================================

output_path = os.path.join(
    RESULT_DIR,
    "three_agent_comparison.csv",
)


results.to_csv(
    output_path,
    index=False,
)


print()
print(
    "Saved:",
    output_path,
)

print()
print("=" * 78)
print("COMPARISON COMPLETE")
print("=" * 78)
