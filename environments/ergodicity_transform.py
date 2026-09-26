import numpy as np
import statsmodels.api as sm
import scipy.integrate as integrate

from scipy.interpolate import interp1d


def learn_ergodicity_transform(
    time_series_data,
    loess_span=0.2,
    min_integrate_range=0.01,
):
    """
    Modernized implementation of Baumann et al.'s TS_VS
    transformation.

    The method:
        1. Compute squared trajectory increments.
        2. Fit LOWESS to log(squared increments) versus state.
        3. Interpolate that log-variance estimate.
        4. Exponentiate to obtain the local variance function.
        5. Integrate v(x)^(-1/2) to obtain h(x).
    """

    data = np.asarray(
        time_series_data,
        dtype=np.float64,
    )

    if data.ndim != 1:
        raise ValueError(
            "time_series_data must be one-dimensional"
        )

    if len(data) < 10:
        raise ValueError(
            "time_series_data is too short"
        )

    # Squared increments.
    squared_increments = (
        np.diff(data) ** 2
    )

    # Avoid log(0).
    squared_increments = np.maximum(
        squared_increments,
        1e-30,
    )

    # Baumann et al. fit LOWESS to:
    #
    # log((X_{t+1} - X_t)^2)
    #
    # as a function of X_t.
    lowess_result = sm.nonparametric.lowess(
        np.log(squared_increments),
        data[:-1],
        frac=loess_span,
    )

    # IMPORTANT:
    # Interpolate the LOG variance.
    #
    # Do not exponentiate before interpolation.
    log_variance_function = interp1d(
        lowess_result[:, 0],
        y=lowess_result[:, 1],
        bounds_error=False,
        kind="linear",
        fill_value="extrapolate",
    )

    # Local variance estimate.
    def variance_function(x):
        return np.exp(
            log_variance_function(x)
        )

    # Ergodicity transformation.
    def transform(value):

        value = float(
            np.asarray(value).reshape(-1)[0]
        )

        result = integrate.fixed_quad(
            lambda x:
                variance_function(x)
                ** (-0.5),
            min_integrate_range,
            value,
            n=1000,
        )[0]

        return float(result)

    return transform
