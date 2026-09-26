# Implied Volatility Recovery
#
# Recover SPX implied vol from observed option midpoints using the
# vectorized Black-Scholes solver, then sanity-check the recovery
# and filter out anything numerically shaky.
# %%
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(rf'C:\Users\{os.getlogin()}\spx-options-pricing-research\src')

import implied_vol as iv
import black_scholes as bsm
import clean_options_data as cod
import clean_market_inputs as cmi

# %%
options_df_raw = cod.clean_options_data()
rates_df = cmi.load_treasury_rates()
dividends_df = cmi.load_dividend_yield()

options_df = cmi.join_market_inputs(
    options_df=options_df_raw, rates_df=rates_df, dividends_df=dividends_df
)

# %%
# quick check that r and q actually populated - no NaNs before we solve for IV
print(options_df.shape)
print(options_df[["DTE", "r", "q"]].describe())
print(options_df[["r", "q"]].isna().sum())

# %%
# vega at the solver's starting guess (20% vol) - low values here
# are basically a preview of which rows the solver is going to choke on
options_df["vega_initial"] = iv.black_scholes_vega(
    S=options_df["UNDERLYING_LAST"],
    K=options_df["STRIKE"],
    T=options_df["DTE"] / 365,
    r=options_df["r"],
    sigma=0.20,
    q=options_df["q"],
)

# %%
# run the vectorized solver
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
# how many failed?
print(options_df["IV_obs"].describe())
print("Missing IV:", options_df["IV_obs"].isna().sum())
print("IV recovery rate:", options_df["IV_obs"].notna().mean())

# %%
# failures aren't random - split by DTE and price to see the pattern
print(options_df.groupby(options_df["IV_obs"].isna())["DTE"].describe())
print(options_df.groupby(options_df["IV_obs"].isna())["MID"].describe())

# %%
# recovery rate by maturity bucket - rises almost monotonically with DTE,
# short-dated stuff is where most of the failures live
options_df["DTE_BUCKET"] = pd.cut(
    options_df["DTE"], bins=[0, 7, 30, 60, 90, 180, 365, 1000, float("inf")]
)
print(options_df.groupby("DTE_BUCKET", observed=True)["IV_obs"].apply(lambda x: x.notna().mean()))

# %%
# sweep a vega cutoff and see the tradeoff between how much data
# we keep vs. how reliable the recovered IV is
for threshold in [0.01, 0.1, 1, 5, 10, 25, 50]:
    mask = options_df["vega_initial"] >= threshold
    print(
        f"Vega >= {threshold:>5}: "
        f"{mask.mean():.1%} of contracts, "
        f"{options_df.loc[mask, 'IV_obs'].notna().mean():.1%} IV recovery"
    )

# %%
# 1 looked like a reasonable cutoff above - compare against 5 to see if
# it's worth being more conservative
for threshold in [1, 5]:
    mask = (options_df["vega_initial"] >= threshold) & options_df["IV_obs"].notna()
    print(f"\nVega >= {threshold}")
    print(options_df.loc[mask, "IV_obs"].describe())

# %%
# sanity check: plug the recovered IV back into BSM and see how close
# we get to the market midpoint we started from
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

options_df["price_error"] = options_df["price_reconstructed"] - options_df["MID"]
print(options_df.loc[valid, "price_error"].describe())

# %%
# keep only what's usable going forward: decent vega + IV actually recovered
options_df = options_df[
    (options_df["vega_initial"] >= 1) & options_df["IV_obs"].notna()
].copy()

print(options_df.shape)
print(options_df["IV_obs"].describe())

# %%
# how many usable quotes do we actually have per day? want to make sure
# there's enough for the surface fits later
daily_counts = options_df.groupby("QUOTE_DATE").size()
print(daily_counts.describe())
print(daily_counts.sort_values().head(20))

# %%
# spot-check one day's strike/maturity/IV coverage
day = options_df[options_df["QUOTE_DATE"] == options_df["QUOTE_DATE"].iloc[0]].copy()

print(day[["STRIKE", "DTE", "IV_obs"]].describe())
print(day[["STRIKE", "DTE", "IV_obs"]].head())

# do call/put pairs share strike-maturity coordinates, or mostly not?
day.groupby(["STRIKE", "DTE"]).size().describe()
day.groupby(["STRIKE", "DTE"])["OPTION_TYPE"].unique().head(10)

# %%
# call vs put IV at the same strike/maturity - should be close if
# put-call parity roughly holds and r/q are reasonable
pivot = day.pivot_table(index=["STRIKE", "DTE"], columns="OPTION_TYPE", values="IV_obs")
pivot["iv_diff"] = pivot["call"] - pivot["put"]
print(pivot["iv_diff"].describe())

# %%
# eyeball the smile for a random day/maturity as a final gut check
rng = np.random.default_rng(98)

random_date = pd.Timestamp(rng.choice(options_df["QUOTE_DATE"].unique()))
day = options_df[options_df["QUOTE_DATE"] == random_date].copy()

random_dte = rng.choice(day["DTE"].unique())
smile = day[day["DTE"] == random_dte].copy()

print(f"Selected date: {random_date.date()}")
print(f"Selected DTE: {random_dte:.2f}")

plt.figure(figsize=(10, 6))
for option_type, group in smile.groupby("OPTION_TYPE"):
    plt.scatter(group["STRIKE"], group["IV_obs"], label=option_type.capitalize(), alpha=0.7)

plt.xlabel("Strike")
plt.ylabel("Implied Volatility")
plt.title(f"Observed SPX Implied Volatility Smile\n{random_date.date()} | DTE = {random_dte:.0f}")
plt.legend()
plt.grid(True)
plt.show()