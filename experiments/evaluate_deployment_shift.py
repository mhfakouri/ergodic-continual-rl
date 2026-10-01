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

SHIFT_STEP = 50

P_WIN_BEFORE = 0.50
P_WIN_AFTER = 0.47

N_TRAJECTORIES = 5000
EVAL_SEED = 2026

MODEL_DIR = "models"
RESULT_DIR = "results"

os.makedirs(
    RESULT_DIR,
    exist_ok=True,
)


# ============================================================
# Analytical log-optimal fraction
# ============================================================

def log_optimal_fraction(p_win):

    f = (
        4.5 * p_win
        - 2.0
    )

    return float(
        np.clip(
            f,
            0.0,
            1.0,
        )
    )


F_BEFORE = log_optimal_fraction(
    P_WIN_BEFORE
)

F_AFTER = log_optimal_fraction(
    P_WIN_AFTER
)


print("=" * 78)
print("DEPLOYMENT-SHIFT EVALUATION")
print("=" * 78)

print(
    "Pre-shift p(win):",
    P_WIN_BEFORE,
)

print(
    "Post-shift p(win):",
    P_WIN_AFTER,
)

print(
    "Shift step:",
    SHIFT_STEP,
)

print(
    "Log-optimal fraction before shift:",
    F_BEFORE,
)

print(
    "Log-optimal fraction after shift:",
    F_AFTER,
)


# ============================================================
# Reconstruct learned transformation
# ============================================================

source_env = CoinTossEnv(
    objective="standard",
    initial_wealth=INITIAL_WEALTH,
    episode_length=EPISODE_LENGTH,
    p_win=0.5,
)


_, info = source_env.reset(
    seed=7
)


wealth_history = [
    info["wealth"]
]


identification_action = np.array(
    [1.0],
    dtype=np.float32,
)


for _ in range(
    EPISODE_LENGTH
):

    (
        _,
        _,
        _,
        _,
        info,
    ) = source_env.step(
        identification_action
    )

    wealth_history.append(
        info["wealth"]
    )


wealth_history = np.asarray(
    wealth_history,
    dtype=np.float64,
)


exact_transform = (
    learn_ergodicity_transform(
        wealth_history,
        loess_span=0.2,
        min_integrate_range=0.01,
    )
)


learned_transform = (
    make_fast_transform(
        exact_transform,
        min_value=1e-3,
        max_value=20000.0,
        n_points=800,
    )
)


learned_baseline = (
    learned_transform(
        INITIAL_WEALTH
    )
)


# ============================================================
# Load frozen models
# ============================================================

print()
print("Loading frozen models...")


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


print(
    "All three models loaded."
)


# ============================================================
# Observation functions
# ============================================================

def log_observation(
    wealth
):

    return np.log(
        np.maximum(
            wealth,
            1e-30,
        )
        / INITIAL_WEALTH
    )


def learned_observation(
    wealth
):

    values = np.empty_like(
        wealth,
        dtype=np.float64,
    )

    for i, w in enumerate(
        wealth
    ):

        values[i] = (
            learned_transform(
                w
            )
            - learned_baseline
        )

    return values


# ============================================================
# Model action -> physical bet fraction
# ============================================================

def predict_fraction(
    model,
    observation,
):

    obs = np.asarray(
        observation,
        dtype=np.float32,
    ).reshape(-1, 1)


    actions, _ = model.predict(
        obs,
        deterministic=True,
    )


    actions = np.asarray(
        actions
    ).reshape(-1)


    actions = np.clip(
        actions,
        -1.0,
        1.0,
    )


    return 0.5 * (
        actions
        + 1.0
    )


# ============================================================
# Common random numbers
# ============================================================

rng = np.random.default_rng(
    EVAL_SEED
)


uniforms = rng.random(
    (
        EPISODE_LENGTH,
        N_TRAJECTORIES,
    )
)


p_schedule = np.full(
    EPISODE_LENGTH,
    P_WIN_BEFORE,
    dtype=np.float64,
)


p_schedule[
    SHIFT_STEP:
] = P_WIN_AFTER


coin_outcomes = (
    uniforms
    < p_schedule[:, None]
)


# ============================================================
# Frozen learned policies
# ============================================================

def simulate_model(
    model,
    observation_function,
):

    wealth = np.full(
        N_TRAJECTORIES,
        INITIAL_WEALTH,
        dtype=np.float64,
    )


    fraction_history = []


    for step in range(
        EPISODE_LENGTH
    ):

        observation = (
            observation_function(
                wealth
            )
        )


        fractions = (
            predict_fraction(
                model,
                observation,
            )
        )


        fraction_history.append(
            fractions.copy()
        )


        wins = coin_outcomes[
            step
        ]


        multipliers = np.where(
            wins,
            1.0
            + 0.5 * fractions,
            1.0
            - 0.4 * fractions,
        )


        wealth *= multipliers


    return (
        wealth,
        np.asarray(
            fraction_history
        ),
    )


# ============================================================
# Fixed policy
# ============================================================

def simulate_fixed(
    fraction
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
            wins,
            1.0
            + 0.5 * fraction,
            1.0
            - 0.4 * fraction,
        )


        wealth *= multiplier


    return wealth


# ============================================================
# Oracle regime-aware policy
# ============================================================

def simulate_oracle():

    wealth = np.full(
        N_TRAJECTORIES,
        INITIAL_WEALTH,
        dtype=np.float64,
    )


    for step in range(
        EPISODE_LENGTH
    ):

        if step < SHIFT_STEP:
            fraction = F_BEFORE
        else:
            fraction = F_AFTER


        wins = coin_outcomes[
            step
        ]


        multiplier = np.where(
            wins,
            1.0
            + 0.5 * fraction,
            1.0
            - 0.4 * fraction,
        )


        wealth *= multiplier


    return wealth


# ============================================================
# Run simulations
# ============================================================

print()
print(
    "Running frozen-policy evaluation..."
)


standard_wealth, standard_fractions = (
    simulate_model(
        standard_model,
        log_observation,
    )
)


log_wealth, log_fractions = (
    simulate_model(
        log_model,
        log_observation,
    )
)


learned_wealth, learned_fractions = (
    simulate_model(
        learned_model,
        learned_observation,
    )
)


frozen_optimal_wealth = (
    simulate_fixed(
        F_BEFORE
    )
)


oracle_wealth = (
    simulate_oracle()
)


# ============================================================
# Summary metrics
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
            "Mean bet pre-shift"
        ] = np.mean(
            fractions[
                :SHIFT_STEP
            ]
        )


        result[
            "Mean bet post-shift"
        ] = np.mean(
            fractions[
                SHIFT_STEP:
            ]
        )

    else:

        result[
            "Mean bet pre-shift"
        ] = np.nan


        result[
            "Mean bet post-shift"
        ] = np.nan


    return result


results = pd.DataFrame(
    [

        summarize(
            "Standard PPO frozen",
            standard_wealth,
            standard_fractions,
        ),

        summarize(
            "Log PPO frozen",
            log_wealth,
            log_fractions,
        ),

        summarize(
            "Learned-transform PPO frozen",
            learned_wealth,
            learned_fractions,
        ),

        summarize(
            "Frozen F=0.25",
            frozen_optimal_wealth,
        ),

        summarize(
            "Oracle regime-aware",
            oracle_wealth,
        ),
    ]
)


print()
print("=" * 78)
print("DEPLOYMENT-SHIFT RESULTS")
print("=" * 78)

print()

print(
    results.to_string(
        index=False
    )
)


# ============================================================
# Save
# ============================================================

output_path = os.path.join(
    RESULT_DIR,
    "deployment_shift_comparison.csv",
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
print("SHIFT EVALUATION COMPLETE")
print("=" * 78)
