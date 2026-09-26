# Implied Volatility Mapping

## Overview

The recovered implied volatilities are used to model the SPX volatility smile separately by quote date and option maturity. Two approaches are compared: SVI, a structured parametric model, and a cubic spline, a more flexible nonparametric model. The goal is to compare how well each approach captures the observed IV relationship across log-moneyness.

## SVI

The Stochastic Volatility Inspired (SVI) parameterization models total implied variance as a function of log-moneyness:

$$
w(k)
=
a+b\left[
\rho(k-m)+\sqrt{(k-m)^2+\sigma^2}
\right]
$$

where $k=\ln(K/F)$ is log-moneyness and $w(k)=IV^2T$ is total variance.

The five SVI parameters are calibrated to the observed total variance for each quote date and maturity. SVI provides a compact representation of the volatility smile using only five parameters.

## Cubic Spline

A cubic spline models the observed implied volatility directly as a smooth function of log-moneyness. The spline is fitted using the available training observations, with unique log-moneyness values used as knots.

Unlike SVI, the spline does not impose a fixed parametric shape, allowing it to follow the observed smile more closely. This makes it a useful flexible benchmark against the more structured SVI model.

## Comparison

SVI and spline models are calibrated using the same training observations and evaluated on the same held-out observations. For each split, the fitted models generate predicted IVs for the test observations, which are compared with their independently recovered observed IVs.

Model performance is evaluated using out-of-sample prediction error, primarily MAE and RMSE. The comparison is repeated across multiple random 80/20 splits to assess whether the results are consistent rather than dependent on a single train-test split.

The analysis also considers the stability of SVI parameters across splits. This helps distinguish a model that fits well consistently from one that achieves similar errors with unstable parameter estimates.
