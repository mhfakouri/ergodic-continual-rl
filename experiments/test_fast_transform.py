import numpy as np

from environments.coin_toss import CoinTossEnv
from environments.ergodicity_transform import (
    learn_ergodicity_transform,
)
from environments.fast_transform import (
    make_fast_transform,
)


def main():

    # ========================================================
    # Generate the same identification trajectory
    # ========================================================

    env = CoinTossEnv(
        objective="standard",
        initial_wealth=100.0,
        episode_length=100,
    )

    _, info = env.reset(seed=7)

    wealth_history = [
        info["wealth"]
    ]

    action = np.array(
        [1.0],
        dtype=np.float32,
    )

    for _ in range(100):

        _, _, _, _, info = env.step(
            action
        )

        wealth_history.append(
            info["wealth"]
        )

    wealth_history = np.asarray(
        wealth_history,
        dtype=np.float64,
    )


    # ========================================================
    # Exact learned transformation
    # ========================================================

    print(
        "Learning exact transformation..."
    )

    exact_transform = (
        learn_ergodicity_transform(
            wealth_history,
            loess_span=0.2,
            min_integrate_range=0.01,
        )
    )


    # ========================================================
    # Fast approximation
    # ========================================================

    print(
        "Building interpolation table..."
    )

    fast_transform = make_fast_transform(
        exact_transform,
        min_value=1e-3,
        max_value=20000.0,
        n_points=800,
    )


    # ========================================================
    # Accuracy test
    # ========================================================

    test_values = np.geomspace(
        0.01,
        10000.0,
        100,
    )

    exact_values = np.array(
        [
            exact_transform(x)
            for x in test_values
        ]
    )

    fast_values = np.array(
        [
            fast_transform(x)
            for x in test_values
        ]
    )

    errors = np.abs(
        exact_values
        - fast_values
    )

    max_error = np.max(
        errors
    )

    rmse = np.sqrt(
        np.mean(
            errors ** 2
        )
    )

    transform_range = (
        exact_values.max()
        - exact_values.min()
    )

    normalized_max_error = (
        max_error
        / transform_range
    )


    monotonic = np.all(
        np.diff(
            fast_values
        ) > 0
    )


    print()
    print("=" * 70)
    print("FAST TRANSFORM CHECK")
    print("=" * 70)

    print(
        "Test points:",
        len(test_values),
    )

    print(
        "Max absolute error:",
        max_error,
    )

    print(
        "RMSE:",
        rmse,
    )

    print(
        "Normalized max error:",
        normalized_max_error,
    )

    print(
        "Strictly increasing:",
        monotonic,
    )


    print()
    print("Sample comparison")
    print("-" * 70)

    for x in [
        1.0,
        10.0,
        100.0,
        1000.0,
        10000.0,
    ]:

        exact = exact_transform(x)
        fast = fast_transform(x)

        print(
            f"W={x:10.2f}  "
            f"exact={exact:12.6f}  "
            f"fast={fast:12.6f}  "
            f"error={abs(exact-fast):.8f}"
        )


    assert monotonic

    assert np.all(
        np.isfinite(
            fast_values
        )
    )


    print()
    print(
        "Fast-transform test PASSED"
    )


if __name__ == "__main__":
    main()
