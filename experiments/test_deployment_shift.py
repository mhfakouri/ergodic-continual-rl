import numpy as np

from environments.coin_toss import (
    CoinTossEnv,
)


env = CoinTossEnv(
    objective="log",
    initial_wealth=100.0,
    episode_length=10,

    # Original regime
    p_win=0.50,

    # Deployment shift
    shift_step=5,
    p_win_after=0.47,
)


obs, info = env.reset(
    seed=123
)

print("=" * 60)
print("DEPLOYMENT SHIFT TEST")
print("=" * 60)

print(
    "Reset:",
    info,
)

action = np.array(
    [0.0],
    dtype=np.float32,
)

for _ in range(10):

    (
        obs,
        reward,
        terminated,
        truncated,
        info,
    ) = env.step(
        action
    )

    print(
        f"step={info['step']:2d}  "
        f"p_win={info['p_win']:.2f}  "
        f"regime={info['regime']:10s}  "
        f"win={info['win']}  "
        f"wealth={info['wealth']:.4f}"
    )

print()
print("Deployment-shift environment test complete.")
