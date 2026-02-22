import pytest
import pandas as pd
import numpy as np
from src.analyse.utils import normalize_dates_for_lr


def test_normalize_dates_for_lr():
    """
    Test that normalize_dates_for_lr correctly normalizes dates to months since 2000-01-01.
    
    Expected behavior:
    - 2000-01-01 should give ~0 months
    - 2001-01-01 should give ~12 months
    - 2010-01-01 should give ~120 months
    """
    # Create test dataframe with dates
    df = pd.DataFrame({
        "date": pd.to_datetime(["2000-01-01", "2001-01-01", "2010-01-01", "2020-01-01"]).to_period("M").to_timestamp()
    })
    
    # Normalize dates
    result = normalize_dates_for_lr(df, date_col="date")
    
    # Extract values
    val_2000 = result[0][0]
    val_2001 = result[1][0]
    val_2010 = result[2][0]
    val_2020 = result[3][0]
    
    print(f"\n2000-01-01: {val_2000:.2f} months")
    print(f"2001-01-01: {val_2001:.2f} months")
    print(f"2010-01-01: {val_2010:.2f} months")
    print(f"2020-01-01: {val_2020:.2f} months")
    
    # Check if values are as expected (with tolerance of 1 month)
    assert abs(val_2000 - 0) < 1, f"Expected 2000-01-01 to be ~0 months, got {val_2000:.2f}"
    assert abs(val_2001 - 12) < 1, f"Expected 2001-01-01 to be ~12 months, got {val_2001:.2f}"
    assert abs(val_2010 - 120) < 1, f"Expected 2010-01-01 to be ~120 months, got {val_2010:.2f}"
    assert abs(val_2020 - 240) < 1, f"Expected 2020-01-01 to be ~240 months, got {val_2020:.2f}"
