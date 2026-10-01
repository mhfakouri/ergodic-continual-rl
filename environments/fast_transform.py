import numpy as np


def make_fast_transform(
    exact_transform,
    min_value=1e-3,
    max_value=20000.0,
    n_points=800,
):
    """
    Build a fast interpolation-based approximation of an
    existing positive-domain ergodicity transformation.

    The exact transform is evaluated once on a logarithmically
    spaced grid. Runtime calls then use interpolation.

    Values outside the tabulated range fall back to the exact
    transformation.
    """

    if min_value <= 0:
        raise ValueError(
            "min_value must be positive"
        )

    if max_value <= min_value:
        raise ValueError(
            "max_value must exceed min_value"
        )

    log_grid = np.linspace(
        np.log(min_value),
        np.log(max_value),
        n_points,
    )

    value_grid = np.exp(
        log_grid
    )

    transform_grid = np.array(
        [
            exact_transform(x)
            for x in value_grid
        ],
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(transform_grid)
    ):
        raise ValueError(
            "Non-finite values found in transform grid"
        )

    def fast_transform(value):

        x = float(
            np.asarray(value).reshape(-1)[0]
        )

        x = max(x, 1e-30)

        if (
            min_value
            <= x
            <= max_value
        ):
            return float(
                np.interp(
                    np.log(x),
                    log_grid,
                    transform_grid,
                )
            )

        # Rare out-of-range values retain the exact behavior.
        return float(
            exact_transform(x)
        )

    return fast_transform
