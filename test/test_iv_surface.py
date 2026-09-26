# Implied Volatility Surface Mapping

# Prepare contract-level SPX implied volatility observations
# for volatility-surface calibration and validation

# %%
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.append(
    rf"C:\Users\{os.getlogin()}\spx-options-pricing-research\src"
)

import clean_market_inputs as cmi
import clean_options_data as cod
import implied_vol as iv


# %%
# Load and prepare the cleaned SPX options data and
# corresponding market inputs

options_df_raw = cod.clean_options_data()

rates_df = cmi.load_treasury_rates()
dividends_df = cmi.load_dividend_yield()

options_df = cmi.join_market_inputs(
    options_df=options_df_raw,
    rates_df=rates_df,
    dividends_df=dividends_df,
)


# %%
# Recover contract-level implied volatility from observed
# option midpoints using vectorized Black-Scholes inversion

options_df["IV_obs"] = iv.implied_volatility_vectorized(
    S=options_df["UNDERLYING_LAST"],
    K=options_df["STRIKE"],
    T=options_df["DTE"] / 365,
    r=options_df["r"],
    q=options_df["q"],
    market_price=options_df["MID"],
    option_type=options_df["OPTION_TYPE"],
)


# %%
# Retain observations with successfully recovered implied volatility

options_df = options_df[
    options_df["IV_obs"].notna()
].copy()


# %%
# Transform contract-level implied volatility into the
# log-moneyness and total-variance inputs used by SVI

options_df["T"] = options_df["DTE"] / 365

options_df["F"] = (
    options_df["UNDERLYING_LAST"]
    * np.exp(
        (options_df["r"] - options_df["q"])
        * options_df["T"]
    )
)

options_df["log_moneyness"] = np.log(
    options_df["STRIKE"] / options_df["F"]
)

options_df["total_variance"] = (
    options_df["IV_obs"] ** 2
    * options_df["T"]
)


# %%
# Confirm that the SVI input variables contain no missing values

print(
    options_df[
        [
            "log_moneyness",
            "IV_obs",
            "total_variance",
        ]
    ].isna().sum()
)


# %%
# Inspect the resulting SVI input variables

print(
    options_df[
        [
            "T",
            "F",
            "log_moneyness",
            "IV_obs",
            "total_variance",
        ]
    ].describe()
)


# %%
# Select a reproducible trading day for the initial SVI calibration

rng = np.random.default_rng(97)

available_dates = options_df["QUOTE_DATE"].unique()

random_date = pd.Timestamp(
    rng.choice(available_dates)
)

day = options_df[
    options_df["QUOTE_DATE"] == random_date
].copy()


# %%
# Select the maturity with the largest number of valid observations
# to provide a well-populated cross-sectional smile

maturity_counts = (
    day.groupby("DTE")
    .size()
    .sort_values(ascending=False)
)

selected_dte = maturity_counts.index[0]

smile = day[
    day["DTE"] == selected_dte
].copy()

print(f"Selected date: {random_date.date()}")
print(f"Selected DTE: {selected_dte:.2f}")
print(f"Observations: {len(smile)}")


# %%
# Confirm that the selected maturity contains sufficient observations
# and valid SVI inputs

print(
    smile[
        [
            "log_moneyness",
            "IV_obs",
            "total_variance",
        ]
    ].describe()
)

print(
    smile[
        [
            "log_moneyness",
            "IV_obs",
            "total_variance",
        ]
    ].isna().sum()
)


# %%
# Define the raw SVI total-variance function

def svi_total_variance(k, a, b, rho, m, sigma):
    """
    Calculate SVI total implied variance
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


# %%
# Prepare the observed log-moneyness and total-variance inputs

k = smile["log_moneyness"].to_numpy()
total_variance = smile["total_variance"].to_numpy()


# %%
# Set the initial SVI parameter values

initial_guess = [
    0.01,   # a
    0.10,   # b
    -0.50,  # rho
    0.00,   # m
    0.10,   # sigma
]


# %%
# Fit SVI parameters by minimizing the difference between
# observed and model-implied total variance

svi_result = least_squares(
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

print("Success:", svi_result.success)
print("Message:", svi_result.message)
print("Parameters:", svi_result.x)
print(
    "Sum of squared errors:",
    np.sum(svi_result.fun ** 2)
)


# %%
# Calculate the fitted total variance for the observed contracts

smile["svi_total_variance"] = svi_total_variance(
    k,
    *svi_result.x,
)


# %%
# Calculate the in-sample SVI total-variance RMSE

rmse = np.sqrt(
    np.mean(
        (
            smile["total_variance"]
            - smile["svi_total_variance"]
        ) ** 2
    )
)

print(f"SVI total-variance RMSE: {rmse:.6f}")


# %%
# Create a smooth log-moneyness grid for the fitted SVI smile

k_grid = np.linspace(
    smile["log_moneyness"].min(),
    smile["log_moneyness"].max(),
    200,
)

svi_fit = svi_total_variance(
    k_grid,
    *svi_result.x,
)


# %%
# Plot observed total variance and the fitted SVI smile

plt.figure(figsize=(10, 6))

for option_type, group in smile.groupby("OPTION_TYPE"):
    plt.scatter(
        group["log_moneyness"],
        group["total_variance"],
        label=f"{option_type.capitalize()} observed",
        alpha=0.6,
    )

plt.plot(
    k_grid,
    svi_fit,
    label="SVI fit",
    linewidth=2,
)

plt.xlabel("Log-moneyness")
plt.ylabel("Total variance")

plt.title(
    f"SPX Total-Variance Smile with SVI Fit\n"
    f"{random_date.date()} | DTE = {selected_dte:.0f}"
)

plt.axvline(
    0,
    linestyle="--",
    linewidth=1,
)

plt.legend()
plt.grid(True)
plt.show()