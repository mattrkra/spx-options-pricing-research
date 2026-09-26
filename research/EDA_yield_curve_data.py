"""
Treasury Yield Curve EDA / Reconstruction

Loads historical FRED Treasury rates, aligns them with the SPX options
sample, and reconstructs maturity-specific risk-free rates via linear
interpolation across the observed Treasury maturities.
"""
# %%
import pandas as pd
import os
import matplotlib.pyplot as plt
import sys
import numpy as np

sys.path.append(rf'C:\Users\{os.getlogin()}\spx-options-pricing-research\src')
import clean_options_data as cod

# reading in the raw FRED data - see docs/01_data_source.md for the source link
username = os.getlogin()
input_path = os.path.join(r"C:\Users", username, "spy-options-pricing", "inputs", "fredgraph.csv")

rates_df = pd.read_csv(input_path)
options_df = cod.clean_options_data()

# %%
rates_df.head(5)

# %%
# get the options date range so we can truncate the FRED data to match
options_start = options_df["QUOTE_DATE"].min()
options_end = options_df["QUOTE_DATE"].max()

# %%
rates_df["observation_date"] = pd.to_datetime(rates_df["observation_date"])

# %%
rates_df = rates_df[
    (rates_df["observation_date"] >= options_start) & (rates_df["observation_date"] <= options_end)
].copy()

# %%
rates_df = rates_df.rename(columns={"observation_date": "QUOTE_DATE"})

# %%
options_df = options_df.merge(rates_df, how="left", on="QUOTE_DATE")

# %%
rate_columns = ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS3"]
options_df[rate_columns].isna().sum()

# %%
# which dates are actually missing rates?
options_df.loc[
    options_df[rate_columns].isna().any(axis=1), ["QUOTE_DATE"] + rate_columns
].drop_duplicates()

# %%
# check the raw FRED data directly for those dates - only 6 rows missing, seems fine to drop
rates_df[
    rates_df["QUOTE_DATE"].isin(
        options_df.loc[options_df[rate_columns].isna().any(axis=1), "QUOTE_DATE"]
    )
]

# %%
options_df = options_df.dropna(subset=rate_columns).copy()
print(f"Remaining quote dates: {options_df['QUOTE_DATE'].nunique()}")

# %%
# quick look at what the yield curve looks like with simple linear
# interpolation, before building this into the actual pricing pipeline

maturities = np.array([1/12, 3/12, 6/12, 1.0, 2.0, 3.0])  # 1M, 3M, 6M, 1Y, 2Y, 3Y

random_date = rates_df["QUOTE_DATE"].dropna().sample(1).iloc[0]

date_rates = (
    rates_df.loc[rates_df["QUOTE_DATE"] == random_date, rate_columns]
    .iloc[0]
    .astype(float)
    .values
)

maturity_grid = np.linspace(maturities.min(), maturities.max(), 200)
interpolated_rates = np.interp(maturity_grid, maturities, date_rates)

plt.figure(figsize=(10, 6))
plt.plot(maturity_grid, interpolated_rates, label="Linear interpolation")
plt.scatter(maturities, date_rates, label="FRED observations", zorder=3)
plt.xlabel("Maturity (Years)")
plt.ylabel("Yield (%)")
plt.title(f"Treasury Yield Curve — {random_date.strftime('%Y-%m-%d')}")
plt.xticks(maturities, ["1M", "3M", "6M", "1Y", "2Y", "3Y"])
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
# %%