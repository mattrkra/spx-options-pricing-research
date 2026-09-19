"""
SPX Options Pricing Research
Exploratory Data Analysis & Data Sanity Checks

Purpose:
    Perform initial exploratory analysis and sanity checks on the raw
    SPX options dataset before downstream modeling.

Checks performed:
    - Dataset dimensions and available columns
    - Key variable identification
    - Missing-value and quote completeness analysis
    - Summary statistics for pricing-related variables
    - Date parsing and dataset coverage
    - SPX underlying price consistency
    - Basic visual inspection of the SPX index level over time

This notebook is intended as a lightweight data-quality and sanity-check
step. Data cleaning and reshaping for downstream analysis are handled
separately by the reusable functions in src/data.py.

The raw dataset is not included in the repository (see docs).
"""
# %%
import pandas as pd
import os
import matplotlib.pyplot as plt

# %%
# Reading in dataset - 
# see docs/01_data_source.md for link
username = os.getlogin()

input_path = os.path.join(
    r"C:\Users",
    username,
    "spy-options-pricing",
    "inputs",
    "combined_options_data.csv"
)

options = pd.read_csv(input_path)

# %%
# Dataset overview
print(f"Rows: {options.shape[0]:,}")
print(f"Columns: {options.shape[1]:,}")

options.columns.tolist()
# %%
# Key cols to be used for modeling
key_columns = [
    "QUOTE_DATE",
    "UNDERLYING_LAST",
    "EXPIRE_DATE",
    "DTE",
    "STRIKE",
    "C_BID",
    "C_ASK",
    "C_IV",
    "C_VOLUME",
    "P_BID",
    "P_ASK",
    "P_IV",
    "P_VOLUME",
]

options[key_columns].head()
# %%
# Check for NaN on key columns (0 missing)
missing = (
    options[key_columns]
    .isna()
    .sum()
    .sort_values(ascending=False)
)

missing
# %%
print("Rows with missing call bid/ask:",
      options[["C_BID", "C_ASK"]].isna().any(axis=1).sum())

print("Rows with missing put bid/ask:",
      options[["P_BID", "P_ASK"]].isna().any(axis=1).sum())

print("Rows with complete bid/ask:",
      options[["C_BID", "C_ASK", "P_BID", "P_ASK"]].notna().all(axis=1).sum())
# %%
# Summary stats on key columns

summary_columns = [
    "UNDERLYING_LAST",
    "DTE",
    "STRIKE",
    "C_BID",
    "C_ASK",
    "C_IV",
    "C_VOLUME",
    "P_BID",
    "P_ASK",
    "P_IV",
    "P_VOLUME",
]

options[summary_columns].describe().T

# %%
# %%
# Convert date columns to datetime
#
# The dataset uses DD-MM-YYYY format.
# For example, 04-01-2010 = January 4, 2010.

options["QUOTE_DATE"] = pd.to_datetime(
    options["QUOTE_DATE"],
    format="%d-%m-%Y"
)

options["EXPIRE_DATE"] = pd.to_datetime(
    options["EXPIRE_DATE"],
    format="%d-%m-%Y"
)

# Sort chronologically
options = options.sort_values(
    ["QUOTE_DATE", "EXPIRE_DATE", "STRIKE"]
).reset_index(drop=True)


# %%
# Basic date coverage check
print("First quote date:", options["QUOTE_DATE"].min().date())
print("Last quote date:", options["QUOTE_DATE"].max().date())
print("Unique quote dates:", options["QUOTE_DATE"].nunique())
print("Unique expiration dates:", options["EXPIRE_DATE"].nunique())

# %%
# Convert date columns from DD-MM-YYYY strings to datetime

options["QUOTE_DATE"] = pd.to_datetime(
    options["QUOTE_DATE"],
    format="%d-%m-%Y"
)

options["EXPIRE_DATE"] = pd.to_datetime(
    options["EXPIRE_DATE"],
    format="%d-%m-%Y"
)
# %%
# Check conversion
print(options["QUOTE_DATE"].head())
print(options["QUOTE_DATE"].dtype)

# %%
# Price check - SPX reported time matches Yahoo
check_date = pd.Timestamp(2010, 1, 4)

date_check = (
    options.loc[
        options["QUOTE_DATE"] == check_date,
        "UNDERLYING_LAST"
    ]
    .dropna()
    .unique()
)

print(f"SPX prices reported on {check_date.date()}:")
print(date_check)
# %%
# SPX underlying price over time

spx = (
    options
    .groupby("QUOTE_DATE")["UNDERLYING_LAST"]
    .first()
    .sort_index()
)

plt.figure(figsize=(12, 6))

plt.plot(
    spx.index,
    spx.values
)

plt.title("SPX Index Level Over Time")
plt.xlabel("Date")
plt.ylabel("SPX Index Level")

plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
