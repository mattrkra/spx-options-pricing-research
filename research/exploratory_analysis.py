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
    "Options.csv"
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
    "[QUOTE_DATE]",
    "[UNDERLYING_LAST]",
    "[EXPIRE_DATE]",
    "[DTE]",
    "[STRIKE]",
    "[C_BID]",
    "[C_ASK]",
    "[C_IV]",
    "[C_VOLUME]",
    "[P_BID]",
    "[P_ASK]",
    "[P_IV]",
    "[P_VOLUME]",
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
# Summary stats on key columns

summary_columns = [
    "[UNDERLYING_LAST]",
    "[DTE]",
    "[STRIKE]",
    "[C_BID]",
    "[C_ASK]",
    "[C_IV]",
    "[C_VOLUME]",
    "[P_BID]",
    "[P_ASK]",
    "[P_IV]",
    "[P_VOLUME]",
]

options[summary_columns].describe().T

# %%
# Plot of SPY prices over time as quick sanity check

spy = (
    options
    .groupby("[QUOTE_DATE]")["[UNDERLYING_LAST]"]
    .first()
    .reset_index()
)


plt.figure(figsize=(12, 6))

plt.plot(
    spy["[QUOTE_DATE]"],
    spy["[UNDERLYING_LAST]"],
)

plt.title("SPY Underlying Price Over Time")
plt.xlabel("Date")
plt.ylabel("SPY Price")
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
