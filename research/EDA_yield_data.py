# %%
import pandas as pd
import os
import matplotlib.pyplot as plt
import sys
import numpy as np

sys.path.append(
    rf'C:\Users\{os.getlogin()}\spy-options-pricing\src'
)
import clean_options_data as cod

# Reading in dataset - 
# see docs/01_data_source.md for links
username = os.getlogin()

input_path = os.path.join(
    r"C:\Users",
    username,
    "spy-options-pricing",
    "inputs",
    "fredgraph.csv"
)

rates_df = pd.read_csv(input_path)

options_df = cod.clean_options_data()
# %%
# Inspecting fred data
rates_df.head(5)
# %%
# Getting range on options data to truncate fred data
options_end = options_df["QUOTE_DATE"].max()
options_start = options_df["QUOTE_DATE"].min()
# %%
# Convert fred observation dates to datetime
rates_df["observation_date"] = pd.to_datetime(
    rates_df["observation_date"]
)
# %%
rates_df = rates_df[
    (rates_df["observation_date"] >= options_start)
    & (rates_df["observation_date"] <= options_end)
].copy()
# %%
# Rename FRED date column to match options data
rates_df = rates_df.rename(
    columns={"observation_date": "QUOTE_DATE"}
)

# %%
# Left join Treasury rates onto options data
options_df = options_df.merge(
    rates_df,
    how="left",
    on="QUOTE_DATE",
)

# %%
# Checking for missing yields
rate_columns = [
    "DGS1MO",
    "DGS3MO",
    "DGS6MO",
    "DGS1",
    "DGS2",
    "DGS3",
]

options_df[rate_columns].isna().sum()
# %%
# Seeing what dates are missing
options_df.loc[
    options_df[rate_columns].isna().any(axis=1),
    ["QUOTE_DATE"] + rate_columns
].drop_duplicates()
# %%
# Checking in rates data directly (only 6 rows missing)
rates_df[
    rates_df["QUOTE_DATE"].isin(
        options_df.loc[
            options_df[rate_columns].isna().any(axis=1),
            "QUOTE_DATE"
        ]
    )
]
# %%
# Remove option dates without complete Treasury rate inputs

options_df = options_df.dropna(
    subset=rate_columns
).copy()

print(
    f"Remaining quote dates: "
    f"{options_df['QUOTE_DATE'].nunique()}"
)
# %%
# Example of what the yield curve would look like
# Using simple linear interpolation for now

# Treasury maturities in years
maturities = np.array([
    1/12,   # 1 month
    3/12,   # 3 months
    6/12,   # 6 months
    1.0,    # 1 year
    2.0,    # 2 years
    3.0     # 3 years
])

# Select a random date
random_date = rates_df["QUOTE_DATE"].dropna().sample(1).iloc[0]

# Get the six Treasury yields for that date
date_rates = (
    rates_df.loc[
        rates_df["QUOTE_DATE"] == random_date,
        rate_columns
    ]
    .iloc[0]
    .astype(float)
    .values
)

# Create a fine maturity grid for interpolation
maturity_grid = np.linspace(
    maturities.min(),
    maturities.max(),
    200
)

# Piecewise linear interpolation
interpolated_rates = np.interp(
    maturity_grid,
    maturities,
    date_rates
)

# Plot
plt.figure(figsize=(10, 6))

plt.plot(
    maturity_grid,
    interpolated_rates,
    label="Linear interpolation"
)

plt.scatter(
    maturities,
    date_rates,
    label="FRED observations",
    zorder=3
)

plt.xlabel("Maturity (Years)")
plt.ylabel("Yield (%)")
plt.title(
    f"Treasury Yield Curve — {random_date.strftime('%Y-%m-%d')}"
)

plt.xticks(
    maturities,
    ["1M", "3M", "6M", "1Y", "2Y", "3Y"]
)

plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
# %%
