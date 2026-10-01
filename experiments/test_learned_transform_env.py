import numpy as np

from environments.learned_transform_env import (
    LearnedTransformCoinTossEnv,
)


def main():

    transform = np.log

    env = LearnedTransformCoinTossEnv(
        transform=transform,
        initial_wealth=100.0,
        episode_length=100,
    )

    obs0, info0 = env.reset(
        seed=42
    )

    # a = 0 -> F = 0.5
    action = np.array(
        [0.0],
        dtype=np.float32,
    )

    (
        obs1,
        reward,
        terminated,
        truncated,
        info1,
    ) = env.step(action)

    expected_obs0 = 0.0

    expected_obs1 = np.log(
        info1["wealth"]
        / 100.0
    )

    expected_reward = np.log(
        info1["wealth"]
    ) - np.log(
        100.0
    )

    print("=" * 60)
    print("LEARNED-TRANSFORM ENVIRONMENT TEST")
    print("=" * 60)

    print(
        "Initial wealth:",
        info0["wealth"],
    )

    print(
        "Initial observation:",
        obs0,
    )

    print()
    print(
        "New wealth:",
        info1["wealth"],
    )

    print(
        "Bet fraction:",
        info1["bet_fraction"],
    )

    print(
        "New observation:",
        obs1,
    )

    print(
        "Reward:",
        reward,
    )

    print()
    print(
        "Expected observation:",
        expected_obs1,
    )

    print(
        "Expected reward:",
        expected_reward,
    )

    assert np.isclose(
        obs0[0],
        expected_obs0,
    )

    assert np.isclose(
        obs1[0],
        expected_obs1,
    )

    assert np.isclose(
        reward,
        expected_reward,
    )

    assert terminated is False
    assert truncated is False

    print()
    print(
        "Centered learned-transform environment test PASSED"
    )


if __name__ == "__main__":
    main()
