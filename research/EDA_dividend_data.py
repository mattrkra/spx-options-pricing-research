"""
Dividend Yield EDA

Compares point-in-time, 3-month trailing average, and 6-month trailing
average dividend-yield assumptions against the subsequent 12-month
realized S&P 500 dividend yield, using MAE to pick a baseline
assumption for the options analysis.
"""
# %%
import pandas as pd
import os
import matplotlib.pyplot as plt
import sys
import numpy as np
from sklearn.metrics import mean_absolute_error

sys.path.append(rf'C:\Users\{os.getlogin()}\spx-options-pricing-research\src')
import clean_options_data as cod

# reading in the raw dividend yield data - see docs/01_data_source.md for the source link
username = os.getlogin()
input_path = os.path.join(r"C:\Users", username, "spy-options-pricing", "inputs", "dividend_yields.csv")

dividends_df = pd.read_csv(input_path)
options_df = cod.clean_options_data()

# %%
dividends_df.head()

# %%
dividends_df["Yield"] = dividends_df["Yield"] / 100
dividends_df["Date"] = pd.to_datetime(dividends_df["Date"])

# %%
dividends_df.head()

# %%
# dividend yield assumption sensitivity - which one actually predicts
# the realized 12M forward yield best: point-in-time, 3M MA, or 6M MA?
# using 2009-2015, roughly +/-2 years around the options data window

dividends_df = dividends_df.sort_values("Date").reset_index(drop=True)

dividends_df["q_point"] = dividends_df["Yield"]
dividends_df["q_3m"] = dividends_df["Yield"].rolling(3).mean()
dividends_df["q_6m"] = dividends_df["Yield"].rolling(6).mean()

# realized average yield over the following 12 months
dividends_df["future_12m_yield"] = (
    dividends_df["Yield"].shift(-1).rolling(12).mean().shift(-11)
)

# only keep rows where every assumption and the target are populated
test_df = dividends_df.dropna(subset=["q_point", "q_3m", "q_6m", "future_12m_yield"])

sensitivity_df = pd.DataFrame({
    "Dividend-yield assumption": ["Point-in-time", "3-month MA", "6-month MA"],
    "MAE": [
        mean_absolute_error(test_df["future_12m_yield"], test_df["q_point"]),
        mean_absolute_error(test_df["future_12m_yield"], test_df["q_3m"]),
        mean_absolute_error(test_df["future_12m_yield"], test_df["q_6m"]),
    ],
})

sensitivity_df