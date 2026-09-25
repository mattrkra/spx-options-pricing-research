# Implied Volatility Recovery
#
# Recover contract-level SPX implied volatility from observed
# option midpoints using vectorized Black-Scholes inversion.
# Validate recovery and filter numerically unstable observations.
# %%
import os 
import sys
import pandas as pd
import numpy as np
sys.path.append(
    rf'C:\Users\{os.getlogin()}\spx-options-pricing-research\src'
)

import implied_vol as iv
import black_scholes as bsm
import clean_options_data as cod
import clean_market_inputs as cmi

# %%
options_df_raw = cod.clean_options_data()
rates_df = cmi.load_treasury_rates()
dividends_df = cmi.load_dividend_yield()

options_df = cmi.join_market_inputs(
    options_df=options_df_raw,
    rates_df=rates_df,
    dividends_df=dividends_df
)

# %%
# Diagnostics before running.
# Checking if r and q populated properly
print(options_df.shape)
print(options_df[["DTE", "r", "q"]].describe())
print(options_df[["r", "q"]].isna().sum())

# %%
# Running the vectorized IV solver
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
# Seeing how many failed.
print(options_df["IV_obs"].describe())

print("Missing IV:", options_df["IV_obs"].isna().sum())
print(
    "IV recovery rate:",
    options_df["IV_obs"].notna().mean()
)

# %%
# Diagnosing IV recovery failures by maturity and option price.
print(
    options_df.groupby(options_df["IV_obs"].isna())["DTE"]
    .describe()
)

print(
    options_df.groupby(options_df["IV_obs"].isna())["MID"]
    .describe()
)

# %%
# Quantifying failures by maturity buckets
# (Recovery rises almost monotonically with maturity)
options_df["DTE_BUCKET"] = pd.cut(
    options_df["DTE"],
    bins=[0, 7, 30, 60, 90, 180, 365, 1000, float("inf")]
)

print(
    options_df.groupby("DTE_BUCKET", observed=True)["IV_obs"]
    .apply(lambda x: x.notna().mean())
)

# %%
# How does the Vega cutoff affect contract retention
# and IV recovery?
for threshold in [0.01, 0.1, 1, 5, 10, 25, 50]:
    mask = options_df["vega_initial"] >= threshold

    print(
        f"Vega >= {threshold:>5}: "
        f"{mask.mean():.1%} of contracts, "
        f"{options_df.loc[mask, 'IV_obs'].notna().mean():.1%} IV recovery"
    )

# %%
# Compare the recovered IV distribution under different
# Vega cutoffs.
for threshold in [1, 5]:
    mask = (
        (options_df["vega_initial"] >= threshold)
        & options_df["IV_obs"].notna()
    )

    print(f"\nVega >= {threshold}")
    print(options_df.loc[mask, "IV_obs"].describe())

# %%
# Validate IV recovery by reconstructing option prices
# from the recovered implied volatility and comparing them
# with the observed market midpoint.

options_df["price_reconstructed"] = np.nan

valid = options_df["IV_obs"].notna()

options_df.loc[valid, "price_reconstructed"] = iv.black_scholes_price(
    S=options_df.loc[valid, "UNDERLYING_LAST"],
    K=options_df.loc[valid, "STRIKE"],
    T=options_df.loc[valid, "DTE"] / 365,
    r=options_df.loc[valid, "r"],
    sigma=options_df.loc[valid, "IV_obs"],
    q=options_df.loc[valid, "q"],
    option_type=options_df.loc[valid, "OPTION_TYPE"],
)

# Calculate reconstruction error between the model-implied
# price and the observed market midpoint.
options_df["price_error"] = (
    options_df["price_reconstructed"] - options_df["MID"]
)

print(
    options_df.loc[valid, "price_error"].describe()
)

# %%
# Retain contracts with sufficient Vega and successfully recovered IV.
options_df = options_df[
    (options_df["vega_initial"] >= 1)
    & options_df["IV_obs"].notna()
].copy()

print(options_df.shape)
print(options_df["IV_obs"].describe())
# %%
