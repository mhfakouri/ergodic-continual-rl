import os
import numpy as np
import pandas as pd

from stable_baselines3 import PPO

from environments.coin_toss import CoinTossEnv
from environments.ergodicity_transform import learn_ergodicity_transform
from environments.fast_transform import make_fast_transform
from environments.regularized_win_rate_estimator import RegularizedWinRateEstimator
from environments.adaptive_fraction import log_optimal_fraction


# ============================================================
# Configuration
# ============================================================

INITIAL_WEALTH = 100.0
EPISODE_LENGTH = 100

SHIFT_STEP = 50
P_WIN_BEFORE = 0.50
P_WIN_AFTER = 0.47

WINDOW_SIZE = 20

N_TRAJECTORIES = 5000
EVAL_SEED = 2026

MODEL_DIR = "models"
RESULT_DIR = "results"

os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# Reconstruct learned transformation
# ============================================================

source_env = CoinTossEnv(
    objective="standard",
    initial_wealth=INITIAL_WEALTH,
    episode_length=EPISODE_LENGTH,
    p_win=0.5,
)

_, info = source_env.reset(seed=7)

wealth_history = [info["wealth"]]

identification_action = np.array(
    [1.0],
    dtype=np.float32,
)

for _ in range(EPISODE_LENGTH):

    _, _, _, _, info = source_env.step(
        identification_action
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
# Load frozen PPO agents
# ============================================================

print("=" * 78)
print("ADAPTIVE DEPLOYMENT-SHIFT EXPERIMENT")
print("=" * 78)

print()
print("Loading frozen PPO agents...")


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


print("All models loaded.")


# ============================================================
# Observation helpers
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


def predict_fraction(
    model,
    observation,
):

    obs = np.asarray(
        observation,
        dtype=np.float32,
    ).reshape(-1, 1)

    action, _ = model.predict(
        obs,
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
# Common random outcomes
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
# Frozen PPO simulation
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

        multiplier = np.where(
            wins,
            1.0 + 0.5 * fractions,
            1.0 - 0.4 * fractions,
        )

        wealth *= multiplier


    return (
        wealth,
        np.asarray(
            fractions_history
        ),
    )


# ============================================================
# Adaptive rolling-window policy
# ============================================================

def simulate_adaptive():

    wealth = np.full(
        N_TRAJECTORIES,
        INITIAL_WEALTH,
        dtype=np.float64,
    )

    estimators = [
    RegularizedWinRateEstimator(
        window_size=WINDOW_SIZE,
        alpha=20.0,
        beta=20.0,
        )
        for _ in range(
            N_TRAJECTORIES
        )
    ]

    fractions_history = np.zeros(
        (
            EPISODE_LENGTH,
            N_TRAJECTORIES,
        ),
        dtype=np.float64,
    )

    p_hat_history = np.zeros(
        (
            EPISODE_LENGTH,
            N_TRAJECTORIES,
        ),
        dtype=np.float64,
    )


    for step in range(
        EPISODE_LENGTH
    ):

        fractions = np.zeros(
            N_TRAJECTORIES,
            dtype=np.float64,
        )


        for i, estimator in enumerate(
            estimators
        ):

            # Hold the known pre-shift optimum until
            # the rolling estimator has a full window.
            if estimator.sample_count() < WINDOW_SIZE:

                p_hat = 0.5
                fraction = 0.25

            else:

                p_hat = estimator.estimate()

                fraction = log_optimal_fraction(
                    p_hat
                )


            p_hat_history[
                step,
                i,
            ] = p_hat

            fractions[
                i
            ] = fraction


        fractions_history[
            step
        ] = fractions


        wins = coin_outcomes[
            step
        ]


        multiplier = np.where(
            wins,
            1.0 + 0.5 * fractions,
            1.0 - 0.4 * fractions,
        )


        wealth *= multiplier


        # Update estimator AFTER observing current outcome.
        for i, estimator in enumerate(
            estimators
        ):

            estimator.update(
                wins[i]
            )


    return (
        wealth,
        fractions_history,
        p_hat_history,
    )


# ============================================================
# Oracle regime-aware policy
# ============================================================

def simulate_oracle():

    wealth = np.full(
        N_TRAJECTORIES,
        INITIAL_WEALTH,
        dtype=np.float64,
    )

    fractions_history = np.zeros(
        (
            EPISODE_LENGTH,
            N_TRAJECTORIES,
        ),
        dtype=np.float64,
    )


    f_before = log_optimal_fraction(
        P_WIN_BEFORE
    )

    f_after = log_optimal_fraction(
        P_WIN_AFTER
    )


    for step in range(
        EPISODE_LENGTH
    ):

        if step < SHIFT_STEP:

            fraction = f_before

        else:

            fraction = f_after


        fractions_history[
            step
        ] = fraction


        wins = coin_outcomes[
            step
        ]


        multiplier = np.where(
            wins,
            1.0 + 0.5 * fraction,
            1.0 - 0.4 * fraction,
        )


        wealth *= multiplier


    return (
        wealth,
        fractions_history,
    )


# ============================================================
# Run
# ============================================================

print()
print(
    "Running common-random-number evaluation..."
)


standard_wealth, standard_fractions = simulate_model(
    standard_model,
    log_observation,
)


log_wealth, log_fractions = simulate_model(
    log_model,
    log_observation,
)


learned_wealth, learned_fractions = simulate_model(
    learned_model,
    learned_observation,
)


adaptive_wealth, adaptive_fractions, adaptive_p_hat = (
    simulate_adaptive()
)


oracle_wealth, oracle_fractions = (
    simulate_oracle()
)


# ============================================================
# Metrics
# ============================================================

def summarize(
    name,
    final_wealth,
    fractions,
):

    mean_log_growth = np.mean(
        np.log(
            final_wealth
            / INITIAL_WEALTH
        )
        / EPISODE_LENGTH
    )


    return {
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
            mean_log_growth,

        "P(final wealth < 100)":
            np.mean(
                final_wealth
                < INITIAL_WEALTH
            ),

        "Mean bet pre-shift":
            np.mean(
                fractions[
                    :SHIFT_STEP
                ]
            ),

        "Mean bet post-shift":
            np.mean(
                fractions[
                    SHIFT_STEP:
                ]
            ),
    }


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
            "Regularized adaptive",
            adaptive_wealth,
            adaptive_fractions,
        ),

        summarize(
            "Oracle regime-aware",
            oracle_wealth,
            oracle_fractions,
        ),
    ]
)


# ============================================================
# Adaptive behavior summary
# ============================================================

mean_p_hat_by_step = np.mean(
    adaptive_p_hat,
    axis=1,
)


mean_fraction_by_step = np.mean(
    adaptive_fractions,
    axis=1,
)


print()
print("=" * 78)
print("FINAL ADAPTIVE COMPARISON")
print("=" * 78)

print()

print(
    results.to_string(
        index=False
    )
)


print()
print("=" * 78)
print("ADAPTATION TRACE")
print("=" * 78)


for step in [
    1,
    20,
    40,
    50,
    55,
    60,
    70,
    80,
    100,
]:

    idx = step - 1

    print(
        f"step={step:3d}  "
        f"mean p_hat="
        f"{mean_p_hat_by_step[idx]:.4f}  "
        f"mean F="
        f"{mean_fraction_by_step[idx]:.4f}"
    )


# ============================================================
# Save
# ============================================================

results_path = os.path.join(
    RESULT_DIR,
    "regularized_adaptive_shift_comparison.csv",
)


results.to_csv(
    results_path,
    index=False,
)


trace_path = os.path.join(
    RESULT_DIR,
    "regularized_adaptive_shift_trace.csv",
)


trace = pd.DataFrame(
    {
        "step":
            np.arange(
                1,
                EPISODE_LENGTH + 1,
            ),

        "true_p_win":
            p_schedule,

        "mean_p_hat":
            mean_p_hat_by_step,

        "mean_bet_fraction":
            mean_fraction_by_step,
    }
)


trace.to_csv(
    trace_path,
    index=False,
)


print()
print(
    "Saved:",
    results_path,
)

print(
    "Saved:",
    trace_path,
)

print()
print("=" * 78)
print("ADAPTIVE EXPERIMENT COMPLETE")
print("=" * 78)
