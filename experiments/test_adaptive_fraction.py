from environments.win_rate_estimator import (
    RollingWinRateEstimator,
)

from environments.adaptive_fraction import (
    log_optimal_fraction,
)


estimator = RollingWinRateEstimator(
    window_size=5,
    initial_estimate=0.5,
)


print("=" * 65)
print("ESTIMATOR -> LOG-OPTIMAL FRACTION TEST")
print("=" * 65)


initial_p = estimator.estimate()

print(
    f"Initial p_hat = {initial_p:.3f}"
)

print(
    f"Initial F*    = "
    f"{log_optimal_fraction(initial_p):.3f}"
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


print()

for step, outcome in enumerate(
    outcomes,
    start=1,
):

    p_hat = estimator.update(
        outcome
    )

    fraction = log_optimal_fraction(
        p_hat
    )

    print(
        f"step={step:2d}  "
        f"win={str(outcome):5s}  "
        f"p_hat={p_hat:.3f}  "
        f"F*={fraction:.3f}"
    )


print()
print("=" * 65)

print(
    "Reference:"
)

print(
    "p=0.50 -> F*=",
    log_optimal_fraction(0.50),
)

print(
    "p=0.47 -> F*=",
    log_optimal_fraction(0.47),
)

print(
    "p=0.40 -> F*=",
    log_optimal_fraction(0.40),
)

print()

print(
    "Estimator-to-action test complete."
)
