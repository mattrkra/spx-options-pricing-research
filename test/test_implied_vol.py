# %%
import os 
import sys

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
test = options_df[
    (options_df["DTE"] >= 30) &
    (options_df["DTE"] <= 90)
].copy()

test["moneyness"] = test["STRIKE"] / test["UNDERLYING_LAST"]

test = test[
    test["moneyness"].between(0.98, 1.02)
].sort_values(["QUOTE_DATE", "DTE"])

test.head()
# %%
import implied_vol as iv
import numpy as np

row = test.iloc[2]

T = row["DTE"] / 365

r = np.interp(
    T,
    [1/12, 3/12, 6/12, 1, 2, 3],
    [
        row["DGS1MO"],
        row["DGS3MO"],
        row["DGS6MO"],
        row["DGS1"],
        row["DGS2"],
        row["DGS3"],
    ],
) / 100

iv_obs = iv.implied_volatility(
    S=row["UNDERLYING_LAST"],
    K=row["STRIKE"],
    T=T,
    r=r,
    q=row["q"],
    market_price=row["MID"],
    option_type=row["OPTION_TYPE"],
)

print("T:", T)
print("r:", r)
print("q:", row["q"])
print("IV_obs:", iv_obs)
# %%
