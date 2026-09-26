import os
import pandas as pd
import numpy as np


def load_dividend_yield(
    filepath: str = None,
    start_date: pd.Timestamp = None,
    end_date: pd.Timestamp = None,
) -> pd.DataFrame:
    """
    Load historical S&P 500 dividend-yield data.

    If filepath is None, the default project input file is used.
    start_date/end_date optionally filter the returned range.
    Returns a DataFrame with QUOTE_DATE and q.
    """

    if filepath is None:
        username = os.getlogin()
        filepath = os.path.join(
            r"C:\Users", username, "spx-options-pricing-research", "inputs", "dividend_yields.csv"
        )

    dividends = pd.read_csv(filepath)

    dividends["Date"] = pd.to_datetime(dividends["Date"])
    dividends["Yield"] = pd.to_numeric(dividends["Yield"], errors="coerce")
    dividends["q"] = dividends["Yield"] / 100

    dividends = dividends[["Date", "q"]].rename(columns={"Date": "QUOTE_DATE"})
    dividends = dividends.dropna(subset=["QUOTE_DATE", "q"])
    dividends = dividends.sort_values("QUOTE_DATE").reset_index(drop=True)

    if start_date is not None:
        dividends = dividends[dividends["QUOTE_DATE"] >= start_date]

    if end_date is not None:
        dividends = dividends[dividends["QUOTE_DATE"] <= end_date]

    return dividends.reset_index(drop=True)


def load_treasury_rates(
    filepath: str = None,
    start_date: pd.Timestamp = None,
    end_date: pd.Timestamp = None,
) -> pd.DataFrame:
    """
    Load historical FRED Treasury constant-maturity rates.

    If filepath is None, the default project input file is used.
    Returns QUOTE_DATE plus rate columns for 1M/3M/6M/1Y/2Y/3Y maturities,
    keeping only dates where all six maturities are present.
    """

    if filepath is None:
        username = os.getlogin()
        filepath = os.path.join(
            r"C:\Users", username, "spx-options-pricing-research", "inputs", "fredgraph.csv"
        )

    rates = pd.read_csv(filepath)

    rates["observation_date"] = pd.to_datetime(rates["observation_date"])
    rates = rates.rename(columns={"observation_date": "QUOTE_DATE"})

    rate_columns = ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS3"]

    rates = rates[["QUOTE_DATE"] + rate_columns].copy()
    rates[rate_columns] = rates[rate_columns].apply(pd.to_numeric, errors="coerce")
    rates = rates.sort_values("QUOTE_DATE").reset_index(drop=True)

    if start_date is not None:
        rates = rates[rates["QUOTE_DATE"] >= start_date]

    if end_date is not None:
        rates = rates[rates["QUOTE_DATE"] <= end_date]

    # drop any date where at least one maturity is missing - partial
    # curves aren't usable for the interpolation step later
    rates = rates.dropna(subset=rate_columns).reset_index(drop=True)

    return rates


def join_market_inputs(
    options_df: pd.DataFrame,
    rates_df: pd.DataFrame = None,
    dividends_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Join Treasury rates and dividend yields onto cleaned SPX option
    observations by quote date.

    Rates are kept at their observed maturities for later per-contract
    interpolation (see _interpolate_risk_free_rate). If rates_df /
    dividends_df aren't passed in, they're loaded automatically over
    the date range spanned by options_df.
    """

    options = options_df.copy()

    options_start = options["QUOTE_DATE"].min()
    options_end = options["QUOTE_DATE"].max()

    if rates_df is None:
        rates_df = load_treasury_rates(start_date=options_start, end_date=options_end)

    if dividends_df is None:
        dividends_df = load_dividend_yield(start_date=options_start, end_date=options_end)

    rate_columns = ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS3"]

    rates = rates_df[["QUOTE_DATE"] + rate_columns].copy()
    dividends = dividends_df[["QUOTE_DATE", "q"]].copy()

    # should already be unique per date, but guard against dupes just in case
    rates = rates.drop_duplicates(subset="QUOTE_DATE")
    dividends = dividends.drop_duplicates(subset="QUOTE_DATE")

    options = options.merge(rates, how="left", on="QUOTE_DATE", validate="many_to_one")
    options = options.merge(dividends, how="left", on="QUOTE_DATE", validate="many_to_one")

    # dividend yields are reported monthly, so forward-fill across the
    # days in between rather than dropping them
    options = options.sort_values("QUOTE_DATE").reset_index(drop=True)
    options["q"] = options["q"].ffill()

    # can't price without a full set of market inputs
    options = options.dropna(subset=rate_columns + ["q"]).copy()

    options = _interpolate_risk_free_rate(options)

    options = options.sort_values(
        ["QUOTE_DATE", "EXPIRE_DATE", "STRIKE", "OPTION_TYPE"]
    ).reset_index(drop=True)

    return options


def _interpolate_risk_free_rate(options_df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate a contract-specific risk-free rate from the Treasury
    curve joined to each option, using DTE/365 as time to expiration.

    Note: this is not a perfect capture of future market expectations -
    see docs/A1_limitations.md.

    Maturity grid: DGS1MO=1mo, DGS3MO=3mo, DGS6MO=6mo, DGS1=1y, DGS2=2y, DGS3=3y
    """

    options = options_df.copy().reset_index(drop=True)

    rate_columns = ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS3"]
    maturity_grid = np.array([1 / 12, 3 / 12, 6 / 12, 1.0, 2.0, 3.0])

    option_maturities = options["DTE"].to_numpy(dtype=float) / 365
    interpolated_rates = np.full(len(options), np.nan)

    # rates are joined per quote date, so every row in a group shares the
    # same curve - just grab it once and interpolate for that day's options
    for quote_date, index in options.groupby("QUOTE_DATE").groups.items():
        date_rates = options.loc[index, rate_columns].iloc[0].to_numpy(dtype=float) / 100

        interpolated_rates[index] = np.interp(
            option_maturities[index], maturity_grid, date_rates
        )

    options["r"] = interpolated_rates

    return options