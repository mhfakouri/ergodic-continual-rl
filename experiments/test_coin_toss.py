import numpy as np

from environments.coin_toss import CoinTossEnv


def main():

    standard_env = CoinTossEnv(
        objective="standard"
    )

    log_env = CoinTossEnv(
        objective="log"
    )

    # Same seed + same action = same coin outcome.
    obs_standard, _ = standard_env.reset(seed=42)
    obs_log, _ = log_env.reset(seed=42)

    action = np.array(
        [0.0],
        dtype=np.float32,
    )

    # Normalized action 0 corresponds to F = 0.5.
    result_standard = standard_env.step(action)
    result_log = log_env.step(action)

    next_obs_standard, reward_standard, _, _, info_standard = (
        result_standard
    )

    next_obs_log, reward_log, _, _, info_log = (
        result_log
    )

    print("Initial standard observation:", obs_standard)
    print("Initial log observation:", obs_log)

    print()
    print("STANDARD")
    print("Wealth:", info_standard["wealth"])
    print("Observation:", next_obs_standard)
    print("Reward:", reward_standard)

    print()
    print("LOG")
    print("Wealth:", info_log["wealth"])
    print("Observation:", next_obs_log)
    print("Reward:", reward_log)

    assert np.isclose(
        info_standard["wealth"],
        info_log["wealth"],
    )

    assert np.isclose(
        next_obs_standard[0],
        next_obs_log[0],
    )

    expected_log_reward = np.log(
        info_log["wealth"] / 100.0
    )

    assert np.isclose(
        reward_log,
        expected_log_reward,
    )

    print()
    print("Coin-toss environment test PASSED")


if __name__ == "__main__":
    main()
