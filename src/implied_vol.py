from scipy.optimize import brentq

import os 
import sys
sys.path.append(
    rf'C:\Users\{os.getlogin()}\spy-options-pricing\src'
)
import black_scholes as bsm

from black_scholes import black_scholes_price

def implied_volatility(
    price,
    S,
    K,
    T,
    r,
    q,
    option_type,
):
    def objective(vol):
        return (
            black_scholes_price(
                S=S,
                K=K,
                T=T,
                r=r,
                sigma=vol,
                q=q,
                option_type=option_type,
            )
            - price
        )

    return brentq(objective, 1e-6, 5.0)