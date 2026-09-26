# Implied Volatility Calibration

## Overview

Implied volatility is recovered from observed SPX option prices using the Black-Scholes-Merton (BSM) model. BSM provides the pricing relationship needed to solve for volatility, while Newton-Raphson provides an efficient numerical inversion. The recovered IVs are used as the observed volatility input for the surface modeling stage.

## Black-Scholes-Merton

For calls and puts:

$$
C = S e^{-qT}N(d_1) - K e^{-rT}N(d_2)
$$

$$
P = K e^{-rT}N(-d_2) - S e^{-qT}N(-d_1)
$$

where:

$$
d_1 =
\frac{
\ln(S/K) + (r-q+\frac{1}{2}\sigma^2)T
}{
\sigma\sqrt{T}
}
$$

$$
d_2 = d_1-\sigma\sqrt{T}
$$

The observed option price is represented by the bid-ask midpoint:

$$
P_{\text{market}} = \frac{\text{Bid}+\text{Ask}}{2}
$$

For each option, the remaining inputs are held fixed and volatility is solved such that:

$$
P_{BSM}(\sigma)=P_{\text{market}}
$$

## Newton-Raphson

Because the BSM pricing equation cannot be directly inverted for volatility, Newton-Raphson is used:

$$
\sigma_{n+1}
=
\sigma_n
-
\frac{
P_{BSM}(\sigma_n)-P_{\text{market}}
}{
Vega(\sigma_n)
}
$$

Vega provides the derivative needed for each update. The solver iterates until the pricing error falls below the specified tolerance or the observation fails a numerical validity check.

The vectorized implementation applies the updates across the full option dataset and removes observations with unusable Vega or unstable volatility estimates.

## Validation

Recovered IVs are substituted back into BSM to reconstruct the option price:

$$
\text{Price Error}
=
P_{BSM}(\widehat{\sigma})
-
P_{\text{market}}
$$

Small reconstruction errors confirm that the numerical inversion is working as intended. The recovered IVs are then used in the volatility-surface analysis.
