import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline


def svi_total_variance(k, a, b, rho, m, sigma):
    """
    Evaluate the raw SVI total-variance parameterization.
    """

    return (
        a
        + b * (
            rho * (k - m)
            + np.sqrt(
                (k - m) ** 2
                + sigma ** 2
            )
        )
    )


def calibrate_svi(k, total_variance):
    """
    Calibrate SVI parameters to observed total variance.
    """

    initial_guess = [
        0.01,   # a
        0.10,   # b
        -0.50,  # rho
        0.00,   # m
        0.10,   # sigma
    ]

    result = least_squares(
        lambda params: (
            svi_total_variance(
                k,
                *params
            ) - total_variance
        ),
        x0=initial_guess,
        bounds=(
            [-np.inf, 0, -0.999, -np.inf, 1e-8],
            [np.inf, np.inf, 0.999, np.inf, np.inf],
        ),
    )

    return result.x


def build_svi_mapping(options_df):
    """
    Calibrate SVI separately for each quote date and option maturity.
    """

    mapping = []

    grouped = options_df.groupby(
        [
            "QUOTE_DATE",
            "EXPIRE_DATE",
            "DTE",
        ]
    )

    for (quote_date, expire_date, dte), group in grouped:

        group = group.dropna(
            subset=[
                "log_moneyness",
                "total_variance",
            ]
        )

        if len(group) < 5:
            continue

        k = group["log_moneyness"].to_numpy()
        total_variance = group["total_variance"].to_numpy()

        try:
            params = calibrate_svi(
                k=k,
                total_variance=total_variance,
            )
        except Exception:
            continue

        fitted_total_variance = svi_total_variance(
            k,
            *params,
        )

        train_rmse = np.sqrt(
            np.mean(
                (
                    total_variance - fitted_total_variance
                ) ** 2
            )
        )

        mapping.append(
            {
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
            }
        )

    return pd.DataFrame(mapping)


def prepare_spline_data(group):
    """
    Prepare log-moneyness and IV observations
    """

    spline_data = (
        group[
            [
                "log_moneyness",
                "IV_obs",
            ]
        ]
        .dropna()
        .groupby("log_moneyness", as_index=False)["IV_obs"]
        .mean()
        .sort_values("log_moneyness")
    )

    return spline_data


def fit_spline(group):
    """
    Fit a cubic spline to observed implied volatility by log-moneyness.
    """

    spline_data = prepare_spline_data(group)

    if len(spline_data) < 4:
        return None

    spline = CubicSpline(
        spline_data["log_moneyness"].to_numpy(),
        spline_data["IV_obs"].to_numpy(),
    )

    return spline


def build_spline_mapping(options_df):
    """
    Build a spline mapping separately for each quote date and maturity.

    Returns a DataFrame containing the training observations needed
    to reconstruct each spline.
    """

    mapping = []

    grouped = options_df.groupby(
        [
            "QUOTE_DATE",
            "EXPIRE_DATE",
            "DTE",
        ]
    )

    for (quote_date, expire_date, dte), group in grouped:

        spline_data = prepare_spline_data(group)

        if len(spline_data) < 4:
            continue

        mapping.extend(
            [
                {
                    "QUOTE_DATE": quote_date,
                    "EXPIRE_DATE": expire_date,
                    "DTE": dte,
                    "log_moneyness": row["log_moneyness"],
                    "IV_obs": row["IV_obs"],
                }
                for _, row in spline_data.iterrows()
            ]
        )

    return pd.DataFrame(mapping)


def apply_spline(
    k,
    spline_data,
):
    """
    Reconstruct and apply a cubic spline to log-moneyness values.
    """

    spline = CubicSpline(
        spline_data["log_moneyness"].to_numpy(),
        spline_data["IV_obs"].to_numpy(),
    )

    return spline(k)