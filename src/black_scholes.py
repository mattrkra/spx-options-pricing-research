import numpy as np
from scipy.stats import norm

def black_scholes_price(
    S,
    K,
    T,
    r,
    sigma,
    q,
    option_type,    
):
    """
    Calculate the Black-Scholes price of a European option.

    Parameters:
        S: Underlying asset price
        K: Strike price
        T: Time to expiration in years
        r: Continuously compounded risk-free rate
        sigma: Volatility
        q: Continuously compounded dividend yield
        option_type: "call" or "put"

    Returns:
        Black-Scholes option price.
    """
    d1 = (
        (np.log(S / K) + (r - q + 0.5 * sigma**2) * T)
        / (sigma * np.sqrt(T))
    )

    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":

        C = (
            (S * np.exp(-q * T) * norm.cdf(d1))
            - (K * np.exp(-r * T) * norm.cdf(d2))
        )

        return C

    elif option_type == "put":

        P = (
            (K * np.exp(-r * T) * norm.cdf(-d2))
            - (S * np.exp(-q * T) * norm.cdf(-d1))
        )

        return P

    else:

        print("Invalid option_type")