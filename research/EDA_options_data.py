"""
Options EDA

Inspect the raw SPX options dataset for structure, completeness, key
variables, date coverage, and basic pricing-data consistency before
handing it off to the cleaning/reshaping step in src/data.py.
The raw dataset isn't in the repo (see docs/01_data.md).
"""
# %%
import pandas as pd
import os
import matplotlib.pyplot as plt

# %%
# reading in the raw dataset - see docs/01_data_source.md for the link
username = os.getlogin()
input_path = os.path.join(
    r"C:\Users", username, "spx-options-pricing-research", "inputs", "combined_options_data.csv"
)

options = pd.read_csv(input_path)

# %%
print(f"Rows: {options.shape[0]:,}")
print(f"Columns: {options.shape[1]:,}")
options.columns.tolist()

# %%
# the columns actually needed for modeling
key_columns = [
    "QUOTE_DATE", "UNDERLYING_LAST", "EXPIRE_DATE", "DTE", "STRIKE",
    "C_BID", "C_ASK", "C_IV", "C_VOLUME",
    "P_BID", "P_ASK", "P_IV", "P_VOLUME",
]

options[key_columns].head()

# %%
# checking for missing values on the key columns - turns out 0 missing
missing = options[key_columns].isna().sum().sort_values(ascending=False)
missing

# %%
print("Rows with missing call bid/ask:", options[["C_BID", "C_ASK"]].isna().any(axis=1).sum())
print("Rows with missing put bid/ask:", options[["P_BID", "P_ASK"]].isna().any(axis=1).sum())
print("Rows with complete bid/ask:", options[["C_BID", "C_ASK", "P_BID", "P_ASK"]].notna().all(axis=1).sum())

# %%
summary_columns = [
    "UNDERLYING_LAST", "DTE", "STRIKE",
    "C_BID", "C_ASK", "C_IV", "C_VOLUME",
    "P_BID", "P_ASK", "P_IV", "P_VOLUME",
]

options[summary_columns].describe().T

# %%
# dataset uses DD-MM-YYYY (e.g. 04-01-2010 = January 4, 2010), not the US format
options["QUOTE_DATE"] = pd.to_datetime(options["QUOTE_DATE"], format="%d-%m-%Y")
options["EXPIRE_DATE"] = pd.to_datetime(options["EXPIRE_DATE"], format="%d-%m-%Y")

options = options.sort_values(["QUOTE_DATE", "EXPIRE_DATE", "STRIKE"]).reset_index(drop=True)

# %%
print("First quote date:", options["QUOTE_DATE"].min().date())
print("Last quote date:", options["QUOTE_DATE"].max().date())
print("Unique quote dates:", options["QUOTE_DATE"].nunique())
print("Unique expiration dates:", options["EXPIRE_DATE"].nunique())

# %%
# quick check that the date parsing actually worked
print(options["QUOTE_DATE"].head())
print(options["QUOTE_DATE"].dtype)

# %%
# spot check - SPX level on a known date matches what Yahoo shows
check_date = pd.Timestamp(2010, 1, 4)

date_check = (
    options.loc[options["QUOTE_DATE"] == check_date, "UNDERLYING_LAST"]
    .dropna()
    .unique()
)

print(f"SPX prices reported on {check_date.date()}:")
print(date_check)

# %%
# SPX underlying level over the full sample, as a gut check for gaps/outliers
spx = options.groupby("QUOTE_DATE")["UNDERLYING_LAST"].first().sort_index()

plt.figure(figsize=(12, 6))
plt.plot(spx.index, spx.values)
plt.title("SPX Index Level Over Time")
plt.xlabel("Date")
plt.ylabel("SPX Index Level")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
# %%