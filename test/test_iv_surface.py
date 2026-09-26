# Implied Volatility Surface Mapping
#
# Prepare contract-level SPX IV observations for volatility-surface
# calibration and validation (SVI + cubic spline)

# %%
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline

sys.path.append(rf"C:\Users\{os.getlogin()}\spx-options-pricing-research\src")

import clean_market_inputs as cmi
import clean_options_data as cod
import implied_vol as iv

# %%
# load cleaned options + market inputs and join them
options_df_raw = cod.clean_options_data()
rates_df = cmi.load_treasury_rates()
dividends_df = cmi.load_dividend_yield()

options_df = cmi.join_market_inputs(
    options_df=options_df_raw, rates_df=rates_df, dividends_df=dividends_df
)

# %%
# recover IV for every contract
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
# only keep what actually solved
options_df = options_df[options_df["IV_obs"].notna()].copy()

# %%
# build the SVI inputs: forward price, log-moneyness, total variance
options_df["T"] = options_df["DTE"] / 365
options_df["F"] = options_df["UNDERLYING_LAST"] * np.exp(
    (options_df["r"] - options_df["q"]) * options_df["T"]
)
options_df["log_moneyness"] = np.log(options_df["STRIKE"] / options_df["F"])
options_df["total_variance"] = options_df["IV_obs"] ** 2 * options_df["T"]

# %%
# make sure nothing's missing before fitting anything
print(options_df[["log_moneyness", "IV_obs", "total_variance"]].isna().sum())
print(options_df[["T", "F", "log_moneyness", "IV_obs", "total_variance"]].describe())

# %%
# pick a random day to calibrate on first, just to get the fit working
# before running it across the whole dataset
rng = np.random.default_rng(97)

available_dates = options_df["QUOTE_DATE"].unique()
random_date = pd.Timestamp(rng.choice(available_dates))

day = options_df[options_df["QUOTE_DATE"] == random_date].copy()

# %%
# grab whichever maturity on that day has the most observations -
# gives the fit the best shot at a clean smile
maturity_counts = day.groupby("DTE").size().sort_values(ascending=False)
selected_dte = maturity_counts.index[0]

smile = day[day["DTE"] == selected_dte].copy()

print(f"Selected date: {random_date.date()}")
print(f"Selected DTE: {selected_dte:.2f}")
print(f"Observations: {len(smile)}")

# %%
# sanity check the slice before fitting
print(smile[["log_moneyness", "IV_obs", "total_variance"]].describe())
print(smile[["log_moneyness", "IV_obs", "total_variance"]].isna().sum())


# %%
def svi_total_variance(k, a, b, rho, m, sigma):
    """Raw SVI total variance."""
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


# %%
k = smile["log_moneyness"].to_numpy()
total_variance = smile["total_variance"].to_numpy()

# a, b, rho, m, sigma - rho bounded to (-1, 1), sigma kept strictly positive
initial_guess = [0.01, 0.10, -0.50, 0.00, 0.10]

# %%
svi_result = least_squares(
    lambda params: svi_total_variance(k, *params) - total_variance,
    x0=initial_guess,
    bounds=(
        [-np.inf, 0, -0.999, -np.inf, 1e-8],
        [np.inf, np.inf, 0.999, np.inf, np.inf],
    ),
)

print("Success:", svi_result.success)
print("Message:", svi_result.message)
print("Parameters:", svi_result.x)
print("Sum of squared errors:", np.sum(svi_result.fun ** 2))

# %%
smile["svi_total_variance"] = svi_total_variance(k, *svi_result.x)

rmse = np.sqrt(np.mean((smile["total_variance"] - smile["svi_total_variance"]) ** 2))
print(f"SVI total-variance RMSE: {rmse:.6f}")

# %%
# smooth grid for plotting the fitted curve
k_grid = np.linspace(smile["log_moneyness"].min(), smile["log_moneyness"].max(), 200)
svi_fit = svi_total_variance(k_grid, *svi_result.x)

# %%
plt.figure(figsize=(10, 6))

for option_type, group in smile.groupby("OPTION_TYPE"):
    plt.scatter(group["log_moneyness"], group["total_variance"], label=f"{option_type.capitalize()} observed", alpha=0.6)

plt.plot(k_grid, svi_fit, label="SVI fit", linewidth=2)

plt.xlabel("Log-moneyness")
plt.ylabel("Total variance")
plt.title(f"SPX Total-Variance Smile with SVI Fit\n{random_date.date()} | DTE = {selected_dte:.0f}")
plt.axvline(0, linestyle="--", linewidth=1)
plt.legend()
plt.grid(True)
plt.show()

# %%
# compare against a cubic spline as a non-parametric alternative to SVI
spline_data = (
    smile[["log_moneyness", "IV_obs"]]
    .dropna()
    .groupby("log_moneyness", as_index=False)["IV_obs"]
    .mean()
    .sort_values("log_moneyness")
)

spline = CubicSpline(spline_data["log_moneyness"].to_numpy(), spline_data["IV_obs"].to_numpy())
spline_fit = spline(k_grid)

plt.figure(figsize=(10, 6))

for option_type, group in smile.groupby("OPTION_TYPE"):
    plt.scatter(group["log_moneyness"], group["IV_obs"], label=f"{option_type.capitalize()} observed", alpha=0.6)

plt.plot(k_grid, spline_fit, label="Cubic spline fit", linewidth=2)

plt.xlabel("Log-moneyness")
plt.ylabel("Implied volatility")
plt.title(f"SPX Implied-Volatility Smile with Cubic Spline Fit\n{random_date.date()} | DTE = {selected_dte:.0f}")
plt.axvline(0, linestyle="--", linewidth=1)
plt.legend()
plt.grid(True)
plt.show()

# %%