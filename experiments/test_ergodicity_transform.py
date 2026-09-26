import numpy as np

from environments.coin_toss import CoinTossEnv
from environments.ergodicity_transform import (
    learn_ergodicity_transform,
)


def main():

    # --------------------------------------------------------
    # 1. Generate one trajectory, similar to Baumann's demo
    # --------------------------------------------------------

    env = CoinTossEnv(
        objective="standard",
        initial_wealth=100.0,
        episode_length=100,
    )

    _, info = env.reset(seed=7)

    wealth_history = [
        info["wealth"]
    ]

    # Normalized action +1 corresponds to betting F = 1.
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


    print("=" * 70)
    print("TRAJECTORY")
    print("=" * 70)

    print(
        "Number of points:",
        len(wealth_history),
    )

    print(
        "Initial wealth:",
        wealth_history[0],
    )

    print(
        "Final wealth:",
        wealth_history[-1],
    )

    print(
        "Minimum wealth:",
        wealth_history.min(),
    )

    print(
        "Maximum wealth:",
        wealth_history.max(),
    )


    # --------------------------------------------------------
    # 2. Learn transformation from the trajectory
    # --------------------------------------------------------

    print()
    print(
        "Learning ergodicity transformation..."
    )

    transform = learn_ergodicity_transform(
        wealth_history,
        loess_span=0.2,
        min_integrate_range=0.01,
    )


    # --------------------------------------------------------
    # 3. Compare learned transform with log
    # --------------------------------------------------------

    positive = wealth_history[
        wealth_history > 0.01
    ]

    test_values = np.quantile(
        positive,
        np.linspace(
            0.05,
            0.95,
            15,
        ),
    )

    test_values = np.unique(
        test_values
    )


    learned_values = np.array(
        [
            transform(x)
            for x in test_values
        ]
    )

    log_values = np.log(
        test_values
    )


    correlation = np.corrcoef(
        learned_values,
        log_values,
    )[0, 1]


    differences = np.diff(
        learned_values
    )

    monotonic = np.all(
        differences > 0
    )


    print()
    print("=" * 70)
    print("TRANSFORMATION CHECK")
    print("=" * 70)

    print(
        "Correlation with log(x):",
        correlation,
    )

    print(
        "Strictly increasing:",
        monotonic,
    )


    print()
    print(
        f"{'Wealth':>14}"
        f"{'Learned h(x)':>20}"
        f"{'log(x)':>18}"
    )

    print(
        "-" * 52
    )

    for x, h, log_x in zip(
        test_values,
        learned_values,
        log_values,
    ):

        print(
            f"{x:14.6f}"
            f"{h:20.6f}"
            f"{log_x:18.6f}"
        )


    print()
    print("=" * 70)

    if monotonic:
        print(
            "Transformation test PASSED"
        )
    else:
        print(
            "WARNING: learned transformation "
            "is not monotonic"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
