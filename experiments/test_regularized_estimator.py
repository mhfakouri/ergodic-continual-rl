from environments.win_rate_estimator import (
    RollingWinRateEstimator,
)

from environments.regularized_win_rate_estimator import (
    RegularizedWinRateEstimator,
)

from environments.adaptive_fraction import (
    log_optimal_fraction,
)


raw = RollingWinRateEstimator(
    window_size=5,
    initial_estimate=0.5,
)


regularized = RegularizedWinRateEstimator(
    window_size=5,
    alpha=20.0,
    beta=20.0,
)


outcomes = [
    True,
    True,
    False,
    True,
    False,
    False,
    False,
]


print("=" * 72)
print("REGULARIZED WIN-RATE ESTIMATOR TEST")
print("=" * 72)

print(
    f"{'step':>4}  "
    f"{'win':>5}  "
    f"{'raw p':>8}  "
    f"{'raw F*':>8}  "
    f"{'reg p':>8}  "
    f"{'reg F*':>8}"
)

print("-" * 72)


for step, outcome in enumerate(
    outcomes,
    start=1,
):

    p_raw = raw.update(
        outcome
    )

    p_reg = regularized.update(
        outcome
    )

    f_raw = log_optimal_fraction(
        p_raw
    )

    f_reg = log_optimal_fraction(
        p_reg
    )

    print(
        f"{step:4d}  "
        f"{str(outcome):>5}  "
        f"{p_raw:8.3f}  "
        f"{f_raw:8.3f}  "
        f"{p_reg:8.3f}  "
        f"{f_reg:8.3f}"
    )


print()
print(
    "Reference p=0.47 -> F*=",
    log_optimal_fraction(
        0.47
    ),
)

print()
print(
    "Regularized estimator test complete."
)
