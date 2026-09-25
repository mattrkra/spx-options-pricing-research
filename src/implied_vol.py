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

def implied_volatility_vectorized(
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
    max_sigma=5.0,
):
    """
    Recover Black-Scholes implied volatility for an array of options
    using a vectorized Newton-Raphson solver.

    Parameters
    ----------
    S : array-like
        Underlying price.
    K : array-like
        Strike price.
    T : array-like
        Time to expiration in years.
    r : array-like
        Risk-free interest rate.
    q : array-like
        Dividend yield.
    market_price : array-like
        Observed market option price.
    option_type : array-like
        Option type: "call" or "put".
    initial_guess : float, default=0.20
        Initial volatility guess.
    tolerance : float, default=1e-8
        Absolute pricing-error tolerance for convergence.
    max_iterations : int, default=100
        Maximum Newton-Raphson iterations.
    min_vega : float, default=1e-10
        Minimum Vega required for a stable Newton update.
    max_sigma : float, default=5.0
        Maximum permitted implied volatility.

    Returns
    -------
    np.ndarray
        Implied volatility for each observation. Failed or invalid
        observations are returned as NaN.
    """

    # Convert numeric inputs to NumPy arrays.
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    q = np.asarray(q, dtype=float)
    market_price = np.asarray(market_price, dtype=float)
    option_type = np.asarray(option_type)

    # Validate option types.
    valid_option_types = np.isin(option_type, ["call", "put"])

    if not np.all(valid_option_types):
        raise ValueError("option_type must contain only 'call' or 'put'")

    # Initialize output.
    iv = np.full(S.shape, np.nan, dtype=float)

    # Identify observations that can enter the solver.
    active = (
        np.isfinite(S)
        & np.isfinite(K)
        & np.isfinite(T)
        & np.isfinite(r)
        & np.isfinite(q)
        & np.isfinite(market_price)
        & valid_option_types
        & (S > 0)
        & (K > 0)
        & (T > 0)
        & (market_price > 0)
    )

    # Initial volatility estimate.
    sigma = np.full(S.shape, initial_guess, dtype=float)

    for _ in range(max_iterations):

        if not np.any(active):
            break

        # Work only on observations still being solved.
        idx = np.flatnonzero(active)

        S_active = S[idx]
        K_active = K[idx]
        T_active = T[idx]
        r_active = r[idx]
        q_active = q[idx]
        market_price_active = market_price[idx]
        option_type_active = option_type[idx]
        sigma_active = sigma[idx]

        # Calculate Black-Scholes price.
        price = black_scholes_price(
            S=S_active,
            K=K_active,
            T=T_active,
            r=r_active,
            sigma=sigma_active,
            q=q_active,
            option_type=option_type_active,
        )

        # Newton-Raphson function:
        #
        # f(sigma) = BS_price(sigma) - market_price
        #
        error = price - market_price_active

        # Identify converged observations.
        converged = np.abs(error) < tolerance

        if np.any(converged):
            converged_idx = idx[converged]

            iv[converged_idx] = sigma_active[converged]

            active[converged_idx] = False

        # Stop if everything converged.
        if not np.any(active):
            break

        # Only continue Newton updates for observations
        # that have not yet converged.
        remaining = ~converged

        idx_remaining = idx[remaining]

        S_remaining = S_active[remaining]
        K_remaining = K_active[remaining]
        T_remaining = T_active[remaining]
        r_remaining = r_active[remaining]
        q_remaining = q_active[remaining]
        market_price_remaining = market_price_active[remaining]
        option_type_remaining = option_type_active[remaining]
        sigma_remaining = sigma_active[remaining]
        error_remaining = error[remaining]

        # Calculate Vega.
        vega = black_scholes_vega(
            S=S_remaining,
            K=K_remaining,
            T=T_remaining,
            r=r_remaining,
            sigma=sigma_remaining,
            q=q_remaining,
        )

        # Observations with effectively zero Vega cannot support
        # a stable Newton-Raphson update.
        valid_vega = (
            np.isfinite(vega)
            & (vega >= min_vega)
        )

        if np.any(~valid_vega):
            failed_idx = idx_remaining[~valid_vega]
            active[failed_idx] = False

        if not np.any(valid_vega):
            continue

        # Keep only observations with usable Vega.
        idx_update = idx_remaining[valid_vega]

        sigma_update = sigma_remaining[valid_vega]
        error_update = error_remaining[valid_vega]
        vega_update = vega[valid_vega]

        # Newton-Raphson update:
        #
        # sigma_new =
        #     sigma - (BS_price - market_price) / Vega
        #
        sigma_new = (
            sigma_update
            - error_update / vega_update
        )

        # Reject unstable Newton steps.
        valid_step = (
            np.isfinite(sigma_new)
            & (sigma_new > 0)
            & (sigma_new <= max_sigma)
        )

        if np.any(~valid_step):
            failed_idx = idx_update[~valid_step]
            active[failed_idx] = False

        if np.any(valid_step):
            sigma[idx_update[valid_step]] = sigma_new[valid_step]

    return iv