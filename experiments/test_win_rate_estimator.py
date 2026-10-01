from environments.win_rate_estimator import (
    RollingWinRateEstimator,
)


estimator = RollingWinRateEstimator(
    window_size=5,
    initial_estimate=0.5,
)


print("=" * 60)
print("ROLLING WIN-RATE ESTIMATOR TEST")
print("=" * 60)


print(
    "Initial estimate:",
    estimator.estimate(),
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


for step, outcome in enumerate(
    outcomes,
    start=1,
):

    estimate = estimator.update(
        outcome
    )

    print(
        f"step={step:2d}  "
        f"win={str(outcome):5s}  "
        f"samples={estimator.sample_count():2d}  "
        f"p_hat={estimate:.3f}"
    )


print()
print(
    "Final rolling estimate:",
    estimator.estimate(),
)

print()
print(
    "Estimator test complete."
)
