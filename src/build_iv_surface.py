import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline


def svi_total_variance(k, a, b, rho, m, sigma):
    """Raw SVI total variance parameterization."""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def calibrate_svi(k, total_variance):
    """Fit SVI params (a, b, rho, m, sigma) to observed total variance via least squares."""

    # a, b, rho, m, sigma
    initial_guess = [0.01, 0.10, -0.50, 0.00, 0.10]
    lower_bounds = [-np.inf, 0, -0.999, -np.inf, 1e-8]
    upper_bounds = [np.inf, np.inf, 0.999, np.inf, np.inf]

    result = least_squares(
        lambda params: svi_total_variance(k, *params) - total_variance,
        x0=initial_guess,
        bounds=(lower_bounds, upper_bounds),
    )

    return result.x


def build_svi_mapping(options_df):
    """Calibrate SVI per quote date / maturity slice and return a params DataFrame."""

    mapping = []
    grouped = options_df.groupby(["QUOTE_DATE", "EXPIRE_DATE", "DTE"])

    for (quote_date, expire_date, dte), group in grouped:
        group = group.dropna(subset=["log_moneyness", "total_variance"])

        # need enough points for a stable 5-param fit
        if len(group) < 5:
            continue

        k = group["log_moneyness"].to_numpy()
        total_variance = group["total_variance"].to_numpy()

        try:
            params = calibrate_svi(k=k, total_variance=total_variance)
        except Exception:
            # skip slices that fail to converge instead of blowing up the whole run
            continue

        fitted = svi_total_variance(k, *params)
        train_rmse = np.sqrt(np.mean((total_variance - fitted) ** 2))

        mapping.append({
            "QUOTE_DATE": quote_date,
            "EXPIRE_DATE": expire_date,
            "DTE": dte,
            "a": params[0],
            "b": params[1],
            "rho": params[2],
            "m": params[3],
            "sigma": params[4],
            "train_count": len(group),
            "train_rmse": train_rmse,
        })

    return pd.DataFrame(mapping)


def prepare_spline_data(group):
    """Average duplicate IV observations at the same log-moneyness and sort for spline fitting."""
    return (
        group[["log_moneyness", "IV_obs"]]
        .dropna()
        .groupby("log_moneyness", as_index=False)["IV_obs"]
        .mean()
        .sort_values("log_moneyness")
    )


def fit_spline(group):
    """Fit a cubic spline of IV vs log-moneyness for a single quote date / maturity slice."""
    spline_data = prepare_spline_data(group)

    if len(spline_data) < 4:
        return None

    return CubicSpline(
        spline_data["log_moneyness"].to_numpy(),
        spline_data["IV_obs"].to_numpy(),
    )


def build_spline_mapping(options_df):
    """
    Store the training points (not the fitted spline objects) for every
    quote date / maturity slice, so splines can be reconstructed later.
    """

    mapping = []
    grouped = options_df.groupby(["QUOTE_DATE", "EXPIRE_DATE", "DTE"])

    for (quote_date, expire_date, dte), group in grouped:
        spline_data = prepare_spline_data(group)

        if len(spline_data) < 4:
            continue

        for _, row in spline_data.iterrows():
            mapping.append({
                "QUOTE_DATE": quote_date,
                "EXPIRE_DATE": expire_date,
                "DTE": dte,
                "log_moneyness": row["log_moneyness"],
                "IV_obs": row["IV_obs"],
            })

    return pd.DataFrame(mapping)


def apply_spline(k, spline_data):
    """Rebuild the cubic spline from stored training points and evaluate it at k."""
    spline = CubicSpline(
        spline_data["log_moneyness"].to_numpy(),
        spline_data["IV_obs"].to_numpy(),
    )
    return spline(k)