import os
import numpy as np

from stable_baselines3 import PPO

from environments.coin_toss import CoinTossEnv
from environments.ergodicity_transform import (
    learn_ergodicity_transform,
)
from environments.fast_transform import (
    make_fast_transform,
)
from environments.learned_transform_env import (
    LearnedTransformCoinTossEnv,
)


# ============================================================
# Configuration
# ============================================================

SEED = 0
IDENTIFICATION_SEED = 7

INITIAL_WEALTH = 100.0
EPISODE_LENGTH = 100

TOTAL_TIMESTEPS = 1_000_000

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "coin_toss_learned_transform_ppo",
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True,
)


# ============================================================
# 1. Generate trajectory used to learn h(W)
# ============================================================

print("=" * 70)
print("LEARNED-TRANSFORM PPO AGENT")
print("=" * 70)

print()
print("Generating identification trajectory...")


source_env = CoinTossEnv(
    objective="standard",
    initial_wealth=INITIAL_WEALTH,
    episode_length=EPISODE_LENGTH,
)

_, info = source_env.reset(
    seed=IDENTIFICATION_SEED
)

wealth_history = [
    info["wealth"]
]

# PPO-normalized action +1 means F = 1.
identification_action = np.array(
    [1.0],
    dtype=np.float32,
)

for _ in range(EPISODE_LENGTH):

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


print(
    "Trajectory points:",
    len(wealth_history),
)

print(
    "Minimum wealth:",
    wealth_history.min(),
)

print(
    "Maximum wealth:",
    wealth_history.max(),
)


# ============================================================
# 2. Learn exact ergodicity transformation
# ============================================================

print()
print(
    "Learning ergodicity transformation..."
)


exact_transform = learn_ergodicity_transform(
    wealth_history,
    loess_span=0.2,
    min_integrate_range=0.01,
)


print(
    "Exact h(100):",
    exact_transform(
        INITIAL_WEALTH
    ),
)


# ============================================================
# 3. Build fast interpolation approximation
# ============================================================

print()
print(
    "Building fast transformation..."
)


fast_transform = make_fast_transform(
    exact_transform,
    min_value=1e-3,
    max_value=20000.0,
    n_points=800,
)


print(
    "Fast h(100):",
    fast_transform(
        INITIAL_WEALTH
    ),
)

print(
    "Approximation error at W=100:",
    abs(
        exact_transform(
            INITIAL_WEALTH
        )
        - fast_transform(
            INITIAL_WEALTH
        )
    ),
)


# ============================================================
# 4. Construct PPO environment
# ============================================================

env = LearnedTransformCoinTossEnv(
    transform=fast_transform,
    initial_wealth=INITIAL_WEALTH,
    episode_length=EPISODE_LENGTH,
)


obs, info = env.reset(
    seed=123
)


print()
print("=" * 70)
print("ENVIRONMENT")
print("=" * 70)

print(
    "Initial wealth:",
    info["wealth"],
)

print(
    "Initial centered observation:",
    obs,
)


# ============================================================
# 5. Train or load learned-transform PPO
# ============================================================

zip_path = (
    MODEL_PATH
    + ".zip"
)


if os.path.exists(
    zip_path
):

    print()
    print(
        "Existing learned-transform model found."
    )

    print(
        "Loading:",
        zip_path,
    )

    model = PPO.load(
        MODEL_PATH,
        env=env,
        device="cpu",
    )

else:

    print()
    print("=" * 70)
    print("TRAINING LEARNED-TRANSFORM PPO")
    print("=" * 70)

    print(
        "Training timesteps:",
        TOTAL_TIMESTEPS,
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

    model.save(
        MODEL_PATH
    )

    print()
    print(
        "Saved model:",
        zip_path,
    )


# ============================================================
# 6. Inspect learned policy
# ============================================================

def learned_fraction(
    model,
    wealth,
):

    transformed_obs = (
        fast_transform(
            wealth
        )
        - fast_transform(
            INITIAL_WEALTH
        )
    )

    obs = np.array(
        [[transformed_obs]],
        dtype=np.float32,
    )

    action, _ = model.predict(
        obs,
        deterministic=True,
    )

    normalized_action = float(
        np.asarray(
            action
        ).reshape(-1)[0]
    )

    normalized_action = np.clip(
        normalized_action,
        -1.0,
        1.0,
    )

    fraction = 0.5 * (
        normalized_action
        + 1.0
    )

    return (
        normalized_action,
        fraction,
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

    action, fraction = (
        learned_fraction(
            model,
            wealth,
        )
    )

    print()
    print(
        f"Wealth = {wealth:.1f}"
    )

    print(
        f"  PPO action   = "
        f"{action:.6f}"
    )

    print(
        f"  Bet fraction = "
        f"{fraction:.6f}"
    )


# ============================================================
# 7. Initial-state comparison with theoretical optimum
# ============================================================

_, initial_fraction = learned_fraction(
    model,
    INITIAL_WEALTH,
)


print()
print("=" * 70)
print("INITIAL-STATE RESULT")
print("=" * 70)

print(
    "Learned-transform PPO fraction:",
    initial_fraction,
)

print(
    "Analytical log-growth optimum:",
    0.25,
)

print(
    "Absolute error:",
    abs(
        initial_fraction
        - 0.25
    ),
)


print()
print("=" * 70)
print("LEARNED-TRANSFORM AGENT COMPLETE")
print("=" * 70)
