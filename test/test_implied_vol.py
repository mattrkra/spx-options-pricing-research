# %%
import os 
import sys

sys.path.append(
    rf'C:\Users\{os.getlogin()}\spy-options-pricing\src'
)
import clean_options_data as cod
import implied_vol as iv
import black_scholes as bsm

# %%
sigma_true = 0.20

price = bsm.black_scholes_price(
    S=100,
    K=100,
    T=1,
    r=0.03,
    sigma=sigma_true,
    q=0.013,
    option_type="call"
)
# %%
sigma_estimated = iv.implied_volatility(
    price=price,
    S=100,
    K=100,
    T=1,
    r=0.03,
    q=0.013,
    option_type="call"
)
# %%
