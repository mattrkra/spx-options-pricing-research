import numpy as np
from scipy.stats import norm


def black_scholes_price(S, K, T, r, sigma, q, option_type):
    """
    Calculate Black-Scholes prices for European options.

    Inputs may be scalars or NumPy arrays. All array inputs must be
    broadcastable to a common shape.
    """

    d1 = (
        (np.log(S / K) + (r - q + 0.5 * sigma**2) * T)
        / (sigma * np.sqrt(T))
    )

    d2 = d1 - sigma * np.sqrt(T)

    call_price = (
        S * np.exp(-q * T) * norm.cdf(d1)
        - K * np.exp(-r * T) * norm.cdf(d2)
    )

    put_price = (
        K * np.exp(-r * T) * norm.cdf(-d2)
        - S * np.exp(-q * T) * norm.cdf(-d1)
    )

    return np.where(option_type == "call", call_price, put_price)