# %%
import os 
import sys

sys.path.append(
    rf'C:\Users\{os.getlogin()}\spx-options-pricing-research\src'
)
import black_scholes as bsm
# %%
# Call
bsm.black_scholes_price(
    S=100, K=100, T=1,
    r=0.03, sigma=0.20, q=0.013,
    option_type="call"
)
# %%
# Put
bsm.black_scholes_price(
    S=100, K=100, T=1,
    r=0.03, sigma=0.20, q=0.013,
    option_type="put"
)
# %%
