"""
Dividend Yield EDA

Evaluates point-in-time, 3-month trailing average, and 6-month trailing
average dividend-yield assumptions against subsequent 12-month realized
S&P 500 dividend yields. The comparison uses mean absolute error (MAE) to
select a baseline dividend-yield assumption for the SPX options analysis.
"""
# %%
import pandas as pd
import os
import matplotlib.pyplot as plt
import sys
import numpy as np
from sklearn.metrics import mean_absolute_error

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
    "dividend_yields.csv"
)

dividends_df = pd.read_csv(input_path)

options_df = cod.clean_options_data()
# %%
dividends_df.head()
# %%
dividends_df["Yield"] = (
    dividends_df["Yield"] / 100
)

dividends_df["Date"] = pd.to_datetime(
    dividends_df["Date"]
)
# %%
dividends_df.head()
# %%
# Dividend-yield assumption sensitivity
# Testing MAE of point in time, 3M MA, 6M MA on 12M forward yield
# Using 2009 - 2015 (+-2 years from options data)

# Make sure observations are chronological
dividends_df = dividends_df.sort_values("Date").reset_index(drop=True)

# Historical dividend-yield assumptions
dividends_df["q_point"] = dividends_df["Yield"]

dividends_df["q_3m"] = (
    dividends_df["Yield"]
    .rolling(3)
    .mean()
)

dividends_df["q_6m"] = (
    dividends_df["Yield"]
    .rolling(6)
    .mean()
)

# Realized average dividend yield over the following 12 months
dividends_df["future_12m_yield"] = (
    dividends_df["Yield"]
    .shift(-1)
    .rolling(12)
    .mean()
    .shift(-11)
)

# Keep observations where all assumptions and the future target exist
test_df = dividends_df.dropna(
    subset=[
        "q_point",
        "q_3m",
        "q_6m",
        "future_12m_yield",
    ]
)

# Calculate MAE for each assumption
sensitivity_df = pd.DataFrame({
    "Dividend-yield assumption": [
        "Point-in-time",
        "3-month MA",
        "6-month MA",
    ],
    "MAE": [
        mean_absolute_error(
            test_df["future_12m_yield"],
            test_df["q_point"],
        ),
        mean_absolute_error(
            test_df["future_12m_yield"],
            test_df["q_3m"],
        ),
        mean_absolute_error(
            test_df["future_12m_yield"],
            test_df["q_6m"],
        ),
    ],
})

sensitivity_df