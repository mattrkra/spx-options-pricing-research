# SPX Options Pricing Research

Recovering SPX implied volatility from historical option quotes, then
comparing SVI against a cubic spline on out-of-sample accuracy.

## Research Question

Does a smooth parametric surface (SVI) generalize better than a
flexible local fit (cubic spline) when predicting SPX implied
volatility out of sample, and does the answer depend on maturity,
moneyness, or how much data is available?

## Approach

1. Clean and reshape historical SPX option quotes
2. Build market inputs (interpolated Treasury rates, dividend yields)
3. Recover implied volatility via Newton-Raphson inversion of Black-Scholes, see [docs/iv_calibration.md](docs/iv_calibration.md)
4. Construct log-moneyness and total variance per contract
5. Calibrate SVI and a cubic spline separately for each quote date and maturity, see [docs/iv_mapping.md](docs/iv_mapping.md)
6. Hold out a random 20% of each smile's observations, calibrate both models on the remaining 80%, and evaluate their predictions on the held-out observations
7. Compare out-of-sample error across maturity, sample size, and time


## Results

SVI outperforms the spline on typical-case out-of-sample RMSE across
nearly every condition tested. The spline's median error is only
moderately worse, but its mean error is inflated by a heavy right
tail of large errors, most pronounced at short and long maturities
and worsening later in the sample period, consistent with thin
liquidity and wider bid-ask noise propagating directly into the
spline's exact local interpolation.

See [research/IV_surface_mapping.ipynb](research/IV_surface_mapping.ipynb)
for the full analysis, plots, and discussion.

## Repository Structure

```text
src/          Reusable pricing, IV, data, and modeling functions
research/     EDA, testing, calibration, and model comparison notebooks
tests/        Unit and model validation tests
docs/         Data and methodology documentation
inputs/       Raw data, excluded from version control
outputs/      Generated analysis outputs, excluded from version control
```

## Data

Options data covers January 2010 through August 2013. Treasury rates
are sourced from FRED; historical S&P 500 dividend yields from Multpl.

See [docs/data.md](docs/data.md) for data sources and market-input construction.

## Limitations

Bid-ask midpoints are used as a proxy for transaction prices. Risk-free
rates are linearly interpolated CMT Treasury yields rather than a
bootstrapped zero curve. Low-vega and non-converged IV observations
are excluded. Holdouts are random within each smile, so results reflect
within-surface generalization rather than performance on entirely
unseen maturities or strike regions.

See [docs/limitations.md](docs/limitations.md) for further discussion.