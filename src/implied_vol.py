import numpy as np
from scipy.stats import norm
import os
import sys

sys.path.append(rf'C:\Users\{os.getlogin()}\spx-options-pricing-research\src')
from black_scholes import black_scholes_price


def black_scholes_vega(S, K, T, r, sigma, q):
    """Black-Scholes vega. Works on scalars or arrays."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def implied_volatility(
    S, K, T, r, q, market_price, option_type,
    initial_guess=0.20,
    tolerance=1e-8,
    max_iterations=100,
    min_vega=1e-10,
):
    """
    Recover IV for a single option via Newton-Raphson. Returns NaN if
    the solver stalls (vega too small) or diverges.
    """

    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    sigma = initial_guess

    for _ in range(max_iterations):
        price = black_scholes_price(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type=option_type)
        error = price - market_price

        if abs(error) < tolerance:
            return sigma

        vega = black_scholes_vega(S=S, K=K, T=T, r=r, sigma=sigma, q=q)

        # vega too flat to trust the next step
        if vega < min_vega:
            return np.nan

        sigma_new = sigma - error / vega

        # negative or absurdly high vol means the step diverged
        if sigma_new <= 0 or sigma_new > 5.0:
            return np.nan

        sigma = sigma_new

    return np.nan


def implied_volatility_vectorized(
    S, K, T, r, q, market_price, option_type,
    initial_guess=0.20,
    tolerance=1e-8,
    max_iterations=100,
    min_vega=1e-10,
    max_sigma=5.0,
):
    """
    Same solver as implied_volatility, but runs all observations in
    parallel and drops rows out of the active set as they converge or
    fail (bad inputs, unusable vega, diverging step).
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

    # rows that pass basic sanity checks get to enter the solver
    active = (
        np.isfinite(S) & np.isfinite(K) & np.isfinite(T) & np.isfinite(r)
        & np.isfinite(q) & np.isfinite(market_price) & valid_option_types
        & (S > 0) & (K > 0) & (T > 0) & (market_price > 0)
    )

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
            S=S_active, K=K_active, T=T_active, r=r_active,
            sigma=sigma_active, q=q_active, option_type=option_type_active,
        )
        error = price - market_price_active
        converged = np.abs(error) < tolerance

        if np.any(converged):
            converged_idx = idx[converged]
            iv[converged_idx] = sigma_active[converged]
            active[converged_idx] = False

        if not np.any(active):
            break

        # only keep grinding on what hasn't converged yet
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

        vega = black_scholes_vega(S=S_active, K=K_active, T=T_active, r=r_active, sigma=sigma_active, q=q_active)

        # flat vega means Newton-Raphson can't take a meaningful step
        valid_vega = np.isfinite(vega) & (vega >= min_vega)

        if np.any(~valid_vega):
            active[idx[~valid_vega]] = False

        if not np.any(valid_vega):
            continue

        idx_update = idx[valid_vega]
        sigma_update = sigma_active[valid_vega]
        error_update = error[remaining][valid_vega]
        vega_update = vega[valid_vega]

        sigma_new = sigma_update - error_update / vega_update

        # throw out anything that jumped somewhere implausible
        valid_step = np.isfinite(sigma_new) & (sigma_new > 0) & (sigma_new <= max_sigma)

        if np.any(~valid_step):
            active[idx_update[~valid_step]] = False

        if np.any(valid_step):
            sigma[idx_update[valid_step]] = sigma_new[valid_step]

    return iv