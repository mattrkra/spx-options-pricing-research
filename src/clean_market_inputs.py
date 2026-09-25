import os
import pandas as pd
import numpy as np


def load_dividend_yield(
    filepath: str = None,
    start_date: pd.Timestamp = None,
    end_date: pd.Timestamp = None,
) -> pd.DataFrame:
    """
    Load and prepare historical S&P 500 dividend-yield data.

    The dividend yield is converted from percentage points to decimal
    form and the historical point-in-time yield is retained as the
    project's baseline dividend-yield assumption.

    Parameters:
        filepath: Path to the dividend-yield CSV. If None, the default
            project input file is used.
        start_date: Optional earliest date to retain.
        end_date: Optional latest date to retain.

    Returns:
        DataFrame containing QUOTE_DATE and q.
    """

    if filepath is None:
        username = os.getlogin()

        filepath = os.path.join(
            r"C:\Users",
            username,
            "spx-options-pricing-research",
            "inputs",
            "dividend_yields.csv",
        )

    dividends = pd.read_csv(filepath)

    dividends["Date"] = pd.to_datetime(
        dividends["Date"]
    )

    dividends["Yield"] = pd.to_numeric(
        dividends["Yield"],
        errors="coerce",
    )

    # Convert percentage yield to decimal form.
    dividends["q"] = dividends["Yield"] / 100

    dividends = dividends[
        ["Date", "q"]
    ].rename(
        columns={"Date": "QUOTE_DATE"}
    )

    dividends = dividends.dropna(
        subset=["QUOTE_DATE", "q"]
    )

    dividends = dividends.sort_values(
        "QUOTE_DATE"
    ).reset_index(drop=True)

    if start_date is not None:
        dividends = dividends[
            dividends["QUOTE_DATE"] >= start_date
        ]

    if end_date is not None:
        dividends = dividends[
            dividends["QUOTE_DATE"] <= end_date
        ]

    return dividends.reset_index(drop=True)


def load_treasury_rates(
    filepath: str = None,
    start_date: pd.Timestamp = None,
    end_date: pd.Timestamp = None,
) -> pd.DataFrame:
    """
    Load and prepare historical FRED Treasury constant-maturity rates.

    The six Treasury maturities used in the project are retained as
    inputs for later maturity-specific risk-free-rate interpolation.

    Parameters:
        filepath: Path to the FRED Treasury-rate CSV. If None, the
            default project input file is used.
        start_date: Optional earliest date to retain.
        end_date: Optional latest date to retain.

    Returns:
        DataFrame containing QUOTE_DATE and Treasury rates for 1M,
        3M, 6M, 1Y, 2Y, and 3Y maturities.
    """

    if filepath is None:
        username = os.getlogin()

        filepath = os.path.join(
            r"C:\Users",
            username,
            "spx-options-pricing-research",
            "inputs",
            "fredgraph.csv",
        )

    rates = pd.read_csv(filepath)

    rates["observation_date"] = pd.to_datetime(
        rates["observation_date"]
    )

    rates = rates.rename(
        columns={
            "observation_date": "QUOTE_DATE"
        }
    )

    rate_columns = [
        "DGS1MO",
        "DGS3MO",
        "DGS6MO",
        "DGS1",
        "DGS2",
        "DGS3",
    ]

    rates = rates[
        ["QUOTE_DATE"] + rate_columns
    ].copy()

    rates[rate_columns] = rates[
        rate_columns
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    rates = rates.sort_values(
        "QUOTE_DATE"
    ).reset_index(drop=True)

    if start_date is not None:
        rates = rates[
            rates["QUOTE_DATE"] >= start_date
        ]

    if end_date is not None:
        rates = rates[
            rates["QUOTE_DATE"] <= end_date
        ]

    # Retain only dates with a complete Treasury curve.
    rates = rates.dropna(
        subset=rate_columns
    ).reset_index(drop=True)

    return rates

def join_market_inputs(
    options_df: pd.DataFrame,
    rates_df: pd.DataFrame = None,
    dividends_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Join historical Treasury rates and dividend yields onto cleaned
    SPX option observations.

    Treasury rates are joined by quote date and retained at their
    observed maturities for later maturity-specific interpolation.
    The selected point-in-time dividend yield is joined as q.

    Parameters:
        options_df: Cleaned SPX option observations returned by
            clean_options_data().
        rates_df: Treasury rate data returned by load_treasury_rates().
            If None, rates are loaded automatically.
        dividends_df: Dividend-yield data returned by
            load_dividend_yield(). If None, dividend yields are loaded
            automatically.

    Returns:
        DataFrame containing cleaned SPX option observations with
        Treasury rates and dividend yields joined by quote date.
    """

    options = options_df.copy()

    # Determine the date range of the options dataset.
    options_start = options["QUOTE_DATE"].min()
    options_end = options["QUOTE_DATE"].max()

    # Load market inputs if they were not supplied.
    if rates_df is None:
        rates_df = load_treasury_rates(
            start_date=options_start,
            end_date=options_end,
        )

    if dividends_df is None:
        dividends_df = load_dividend_yield(
            start_date=options_start,
            end_date=options_end,
        )

    # Treasury maturities used for risk-free-rate interpolation.
    rate_columns = [
        "DGS1MO",
        "DGS3MO",
        "DGS6MO",
        "DGS1",
        "DGS2",
        "DGS3",
    ]

    # Keep only the market-input fields required downstream.
    rates = rates_df[
        ["QUOTE_DATE"] + rate_columns
    ].copy()

    dividends = dividends_df[
        ["QUOTE_DATE", "q"]
    ].copy()

    # Ensure one market-input observation per quote date.
    rates = rates.drop_duplicates(
        subset="QUOTE_DATE"
    )

    dividends = dividends.drop_duplicates(
        subset="QUOTE_DATE"
    )

    # Join Treasury rates onto option observations.
    options = options.merge(
        rates,
        how="left",
        on="QUOTE_DATE",
        validate="many_to_one",
    )

    # Join dividend yield onto option observations.
    options = options.merge(
        dividends,
        how="left",
        on="QUOTE_DATE",
        validate="many_to_one",
    )

    # Remove option observations from dates without a complete
    # Treasury curve or dividend-yield input.
    options = options.dropna(
        subset=rate_columns + ["q"]
    ).copy()

    # Convert the date-level Treasury curve into a
    # contract-specific risk-free rate.
    options = _interpolate_risk_free_rate(options)

    # Sort consistently after joining market inputs.
    options = options.sort_values(
        [
            "QUOTE_DATE",
            "EXPIRE_DATE",
            "STRIKE",
            "OPTION_TYPE",
        ]
    ).reset_index(drop=True)

    return options


def _interpolate_risk_free_rate(options_df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate a contract-specific risk-free rate from the Treasury
    constant-maturity curve joined to each option observation.

    Treasury inputs:
        DGS1MO = 1 month
        DGS3MO = 3 months
        DGS6MO = 6 months
        DGS1   = 1 year
        DGS2   = 2 years
        DGS3   = 3 years

    The Treasury rates are assumed to be in percentage points and are
    converted to decimal form for use in Black-Scholes.

    The option's time to expiration is calculated as DTE / 365.

    Parameters
    ----------
    options_df : pd.DataFrame
        Options data containing DTE and the joined Treasury rate columns.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with a contract-specific `r` column.
    """

    options = options_df.copy()

    rate_columns = [
        "DGS1MO",
        "DGS3MO",
        "DGS6MO",
        "DGS1",
        "DGS2",
        "DGS3",
    ]

    # Treasury maturities expressed in years.
    maturity_grid = np.array([
        1 / 12,
        3 / 12,
        6 / 12,
        1.0,
        2.0,
        3.0,
    ])

    # Convert option time to expiration from days to years.
    option_maturities = options["DTE"].to_numpy(dtype=float) / 365

    # Store interpolated rates here.
    interpolated_rates = np.full(len(options), np.nan)

    # Interpolate separately for each quote date because the Treasury
    # curve changes from day to day.
    for quote_date, index in options.groupby("QUOTE_DATE").groups.items():

        date_rates = options.loc[index, rate_columns].iloc[0].to_numpy(dtype=float)

        # Convert percentage-point Treasury yields to decimals.
        date_rates = date_rates / 100

        # Interpolate each option's maturity using that day's curve.
        interpolated_rates[index] = np.interp(
            option_maturities[index],
            maturity_grid,
            date_rates,
        )

    options["r"] = interpolated_rates

    return options