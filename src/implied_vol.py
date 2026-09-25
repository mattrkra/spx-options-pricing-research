import numpy as np
from scipy.stats import norm
import os 
import sys

sys.path.append(
    rf'C:\Users\{os.getlogin()}\spx-options-pricing-research\src'
)
from black_scholes import black_scholes_price


def black_scholes_vega(S, K, T, r, sigma, q):
    """
    Calculate Black-Scholes vega.

    Inputs may be scalars or NumPy arrays.
    """

    d1 = (
        (np.log(S / K) + (r - q + 0.5 * sigma**2) * T)
        / (sigma * np.sqrt(T))
    )

    return (
        S
        * np.exp(-q * T)
        * norm.pdf(d1)
        * np.sqrt(T)
    )


def implied_volatility(
    S,
    K,
    T,
    r,
    q,
    market_price,
    option_type,
    initial_guess=0.20,
    tolerance=1e-8,
    max_iterations=100,
    min_vega=1e-10,
):
    """
    Recover Black-Scholes implied volatility using Newton-Raphson.

    Returns NaN if the solver cannot obtain a valid solution.
    """

    # Basic input validation
    if S <= 0 or K <= 0 or T <= 0:
        return np.nan

    if market_price <= 0:
        return np.nan

    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    sigma = initial_guess

    for _ in range(max_iterations):

        price = black_scholes_price(
            S=S,
            K=K,
            T=T,
            r=r,
            sigma=sigma,
            q=q,
            option_type=option_type,
        )

        error = price - market_price

        if abs(error) < tolerance:
            return sigma

        vega = black_scholes_vega(
            S=S,
            K=K,
            T=T,
            r=r,
            sigma=sigma,
            q=q,
        )

        if vega < min_vega:
            return np.nan

        sigma_new = sigma - error / vega

        # Prevent Newton step from producing invalid volatility
        if sigma_new <= 0 or sigma_new > 5.0:
            return np.nan

        sigma = sigma_new

    return np.nan