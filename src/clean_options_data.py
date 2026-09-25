import os
import pandas as pd


def clean_options_data(
    filepath: str = None,
) -> pd.DataFrame:
    """
    Load, clean, and reshape raw SPX options data for downstream analysis.

    If no filepath is provided, the default input file is loaded from the
    project's inputs directory.

    Processing steps:
        1. Load the raw options dataset.
        2. Keep the underlying, date, expiration, strike, and quote fields
           required for option pricing and implied-volatility estimation.
        3. Convert quote and expiration dates from DD-MM-YYYY strings
           to pandas datetime objects.
        4. Convert required numeric fields to numeric dtype.
        5. Reshape call and put quotes from wide format into long format,
           creating one observation per option contract.
        6. Calculate the bid-ask midpoint for each option.
        7. Remove observations with missing required quote information.
        8. Remove observations with invalid values required for pricing.
        9. Sort observations by quote date, expiration date, strike,
           and option type.

    Parameters:
        filepath: Path to the raw options CSV. If None, the default
            project input file is used.

    Returns:
        A cleaned long-format DataFrame containing individual call and
        put option observations suitable for pricing and IV analysis.
    """

    if filepath is None:
        username = os.getlogin()

        filepath = os.path.join(
            r"C:\Users",
            username,
            "spx-options-pricing-research",
            "inputs",
            "combined_options_data.csv",
        )

    options = pd.read_csv(filepath)

    required_columns = [
        "QUOTE_DATE",
        "UNDERLYING_LAST",
        "EXPIRE_DATE",
        "DTE",
        "STRIKE",
        "C_BID",
        "C_ASK",
        "P_BID",
        "P_ASK",
    ]

    options = options[required_columns].copy()

    # Convert dates
    options["QUOTE_DATE"] = pd.to_datetime(
        options["QUOTE_DATE"],
        format="%d-%m-%Y",
    )

    options["EXPIRE_DATE"] = pd.to_datetime(
        options["EXPIRE_DATE"],
        format="%d-%m-%Y",
    )

    # Convert numeric fields
    numeric_columns = [
        "UNDERLYING_LAST",
        "DTE",
        "STRIKE",
        "C_BID",
        "C_ASK",
        "P_BID",
        "P_ASK",
    ]

    options[numeric_columns] = options[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Create call observations
    calls = options[
        [
            "QUOTE_DATE",
            "UNDERLYING_LAST",
            "EXPIRE_DATE",
            "DTE",
            "STRIKE",
            "C_BID",
            "C_ASK",
        ]
    ].copy()

    calls = calls.rename(
        columns={
            "C_BID": "BID",
            "C_ASK": "ASK",
        }
    )

    calls["OPTION_TYPE"] = "call"

    # Create put observations
    puts = options[
        [
            "QUOTE_DATE",
            "UNDERLYING_LAST",
            "EXPIRE_DATE",
            "DTE",
            "STRIKE",
            "P_BID",
            "P_ASK",
        ]
    ].copy()

    puts = puts.rename(
        columns={
            "P_BID": "BID",
            "P_ASK": "ASK",
        }
    )

    puts["OPTION_TYPE"] = "put"

    # Combine calls and puts
    options = pd.concat(
        [calls, puts],
        ignore_index=True,
    )

    # Calculate market midpoint
    options["MID"] = (
        options["BID"] + options["ASK"]
    ) / 2

    # Remove observations missing required pricing information
    options = options.dropna(
        subset=[
            "QUOTE_DATE",
            "UNDERLYING_LAST",
            "EXPIRE_DATE",
            "DTE",
            "STRIKE",
            "BID",
            "ASK",
            "MID",
        ]
    )

    # Remove invalid observations
    options = options[
        (options["UNDERLYING_LAST"] > 0)
        & (options["STRIKE"] > 0)
        & (options["DTE"] > 0)
        & (options["BID"] >= 0)
        & (options["ASK"] > 0)
        & (options["ASK"] >= options["BID"])
        & (options["MID"] > 0)
    ]

    # Sort consistently
    options = options.sort_values(
        [
            "QUOTE_DATE",
            "EXPIRE_DATE",
            "STRIKE",
            "OPTION_TYPE",
        ]
    ).reset_index(drop=True)

    return options
