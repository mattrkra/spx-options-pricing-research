# Data

## Data Sources

Historical SPX options data is sourced from the [S&P 500 Options 2010–2023 EOD dataset](https://www.kaggle.com/datasets/shubhamcodez/s-and-p-500-daily-options-data-2010-2023) on Kaggle. The provided data covers January 2010 through August 2013.

Historical Treasury constant-maturity rates used for the risk-free rate \(r\) are sourced from the [St. Louis Fed](https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS1MO,DGS3MO,DGS6MO,DGS1,DGS2,DGS3).

[Historical S&P 500 dividend yields](https://www.multpl.com/s-p-500-dividend-yield/table/by-month) are used for the dividend yield \(q\).

## Market Inputs

Treasury rates are linearly interpolated across the 1M, 3M, 6M, 1Y, 2Y, and 3Y maturities to obtain a contract-specific \(r\). Linear interpolation provides a simple approximation across the available maturity grid.

The dividend yield is observed monthly and carried forward to option quote dates. A monthly yield is used because higher-frequency historical dividend expectations are not available in the selected data sources.

Dividend yield is an approximation to the dividend input used in the forward price. Actual dividends may differ from the market's expected future dividends, so the resulting forward is not perfectly aligned with the market forward. Dividend futures data would provide a more market-based input but is outside the scope of this project due to data limitations.

The forward price is approximated as:

$$
F = S e^{(r-q)T}
$$

where \(S\) is the observed SPX level, \(r\) is the interpolated risk-free rate, \(q\) is the dividend yield, and \(T\) is time to expiration.

## Option Data

Call and put quotes are reshaped into long format. The option price is represented by the bid-ask midpoint:

$$
\text{MID} = \frac{\text{BID} + \text{ASK}}{2}
$$

Invalid observations and observations with missing required pricing inputs are removed.
