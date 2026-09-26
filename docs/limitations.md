# Limitations

The analysis uses end-of-day option quotes, with the bid-ask midpoint used as the observed option price. This does not capture the actual transaction price or the information contained in the spread, but provides a consistent price measure across the dataset.

Direct SPX forward prices are not available in the selected data. Instead, forwards are approximated using the observed SPX level, Treasury rates interpolated from FRED, and a monthly historical S&P 500 dividend yield. This provides a consistent point-in-time approximation, although the historical dividend yield may differ from the market's expected future dividends.

Implied volatility recovery is less reliable for low-Vega options, particularly deep in-the-money or out-of-the-money options and very short-dated contracts. Small differences in option prices can produce large changes in implied volatility, and Newton-Raphson may fail to converge. These observations are excluded when Vega is below the analysis threshold or the solver does not obtain a valid solution.

The SVI and spline models are evaluated using repeated random 80/20 contract-level splits. Because nearby strikes and maturities can appear in both training and test sets, the results measure out-of-sample prediction within an observed surface rather than performance on entirely unseen regions. The same splits are used for both models to keep the comparison consistent.

Finally, the provided options data covers January 2010 through August 2013 despite the source dataset being labeled 2010–2023. The results therefore do not capture later market regimes.