"""
Provides functions to load and query the merged CO₂-per-capita dataset.
"""

from pathlib import Path

import pandas as pd

# Module‐level cache for the merged DataFrame
_cached_df = None

# Resolve the dataset relative to this file so the page loads regardless of the
# working directory Streamlit happens to be started from.
DATA_PATH = Path(__file__).resolve().parent / "assets" / "All main data (1998 to 2023).xlsx"


def load_merged_data() -> pd.DataFrame:
    """Load and cache the merged CO₂-per-capita data from Excel.

    Expects:
      - File: assets/All main data (1998 to 2023).xlsx
      - Sheet name: "merged"
      - Columns:
          * State (abbrev, e.g. "WY", "AK")
          * Year (int)
          * co2 per capita (float)

    Returns:
        pd.DataFrame: The full merged dataset, cached after first load.
    """
    global _cached_df

    if _cached_df is None:
        # engine="openpyxl" ensures pandas uses the correct reader
        _cached_df = pd.read_excel(DATA_PATH, sheet_name="merged", engine="openpyxl")
    return _cached_df


def get_state_co2_series(state_code: str) -> pd.DataFrame:
    """Retrieve the CO₂-per-capita time series for a single state.

    Args:
        state_code (str): Two-letter state abbreviation, case‐insensitive.

    Returns:
        pd.DataFrame: A DataFrame with columns ["Year", "co2 per capita"],
                      sorted by Year ascending.
    """
    df = load_merged_data()
    # Filter to the requested state
    df_sub = df[df["State"] == state_code.upper()]

    # Select and sort the relevant columns
    return df_sub[["Year", "co2 per capita"]].sort_values("Year")
