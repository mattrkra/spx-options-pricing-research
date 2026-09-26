import os
import pandas as pd


def clean_options_data(filepath: str = None) -> pd.DataFrame:
    """
    Load raw SPX options data and reshape it into long format (one row
    per call/put observation), dropping anything unusable for pricing.

    If filepath is None, the default project input file is used.
    """

    if filepath is None:
        username = os.getlogin()
        filepath = os.path.join(
            r"C:\Users", username, "spx-options-pricing-research", "inputs", "combined_options_data.csv"
        )

    options = pd.read_csv(filepath)

    required_columns = [
        "QUOTE_DATE", "UNDERLYING_LAST", "EXPIRE_DATE", "DTE", "STRIKE",
        "C_BID", "C_ASK", "P_BID", "P_ASK",
    ]
    options = options[required_columns].copy()

    options["QUOTE_DATE"] = pd.to_datetime(options["QUOTE_DATE"], format="%d-%m-%Y")
    options["EXPIRE_DATE"] = pd.to_datetime(options["EXPIRE_DATE"], format="%d-%m-%Y")

    numeric_columns = ["UNDERLYING_LAST", "DTE", "STRIKE", "C_BID", "C_ASK", "P_BID", "P_ASK"]
    options[numeric_columns] = options[numeric_columns].apply(pd.to_numeric, errors="coerce")

    # split into separate call/put rows, then stack them into one long table
    calls = options[["QUOTE_DATE", "UNDERLYING_LAST", "EXPIRE_DATE", "DTE", "STRIKE", "C_BID", "C_ASK"]].copy()
    calls = calls.rename(columns={"C_BID": "BID", "C_ASK": "ASK"})
    calls["OPTION_TYPE"] = "call"

    puts = options[["QUOTE_DATE", "UNDERLYING_LAST", "EXPIRE_DATE", "DTE", "STRIKE", "P_BID", "P_ASK"]].copy()
    puts = puts.rename(columns={"P_BID": "BID", "P_ASK": "ASK"})
    puts["OPTION_TYPE"] = "put"

    options = pd.concat([calls, puts], ignore_index=True)

    # use the bid-ask midpoint as the option's market price
    options["MID"] = (options["BID"] + options["ASK"]) / 2

    required_fields = ["QUOTE_DATE", "UNDERLYING_LAST", "EXPIRE_DATE", "DTE", "STRIKE", "BID", "ASK", "MID"]
    options = options.dropna(subset=required_fields)

    # filter out quotes that can't be real (zero/negative prices,
    # expired contracts, crossed markets, etc.)
    options = options[
        (options["UNDERLYING_LAST"] > 0)
        & (options["STRIKE"] > 0)
        & (options["DTE"] > 0)
        & (options["BID"] >= 0)
        & (options["ASK"] > 0)
        & (options["ASK"] >= options["BID"])
        & (options["MID"] > 0)
    ]

    options = options.sort_values(
        ["QUOTE_DATE", "EXPIRE_DATE", "STRIKE", "OPTION_TYPE"]
    ).reset_index(drop=True)

    return options