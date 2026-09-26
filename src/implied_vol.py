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
    Calculate Black-Scholes vega. Inputs may be scalars or NumPy arrays.
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
    Recover Black-Scholes implied volatility using Newton-Raphson. Returns 
    NaN if the solver cannot obtain a valid solution.
    """

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
    Recover Black-Scholes implied volatility using vectorized
    Newton-Raphson with Vega-based convergence checks.
    """

    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    q = np.asarray(q, dtype=float)
    market_price = np.asarray(market_price, dtype=float)
    option_type = np.asarray(option_type)

    valid_option_types = np.isin(option_type, ["call", "put"])

    if not np.all(valid_option_types):
        raise ValueError("option_type must contain only 'call' or 'put'")

    iv = np.full(S.shape, np.nan, dtype=float)

    # Identify observations that can enter the solver
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

    # Initial volatility estimate
    sigma = np.full(S.shape, initial_guess, dtype=float)

    for _ in range(max_iterations):

        if not np.any(active):
            break

        idx = np.flatnonzero(active)

        S_active = S[idx]
        K_active = K[idx]
        T_active = T[idx]
        r_active = r[idx]
        q_active = q[idx]
        market_price_active = market_price[idx]
        option_type_active = option_type[idx]
        sigma_active = sigma[idx]

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
        error = price - market_price_active

        converged = np.abs(error) < tolerance

        if np.any(converged):
            converged_idx = idx[converged]

            iv[converged_idx] = sigma_active[converged]

            active[converged_idx] = False

        if not np.any(active):
            break

        # Keep observations that have not converged
        remaining = ~converged

        idx = idx[remaining]
        S_active = S_active[remaining]
        K_active = K_active[remaining]
        T_active = T_active[remaining]
        r_active = r_active[remaining]
        q_active = q_active[remaining]
        market_price_active = market_price_active[remaining]
        option_type_active = option_type_active[remaining]
        sigma_active = sigma_active[remaining]

        # Calculate Vega
        vega = black_scholes_vega(
            S=S_active,
            K=K_active,
            T=T_active,
            r=r_active,
            sigma=sigma_active,
            q=q_active,
        )

        # Keep observations with usable Vega
        valid_vega = (
            np.isfinite(vega)
            & (vega >= min_vega)
        )

        if np.any(~valid_vega):
            failed_idx = idx[~valid_vega]
            active[failed_idx] = False

        if not np.any(valid_vega):
            continue

        idx_update = idx[valid_vega]
        sigma_update = sigma_active[valid_vega]
        error_active = error[remaining]
        error_update = error_active[valid_vega]
        vega_update = vega[valid_vega]

        # Newton-Raphson update:
        # sigma_new =
        #     sigma - (BS_price - market_price) / Vega
        sigma_new = (
            sigma_update
            - error_update / vega_update
        )

        # Reject unstable steps
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