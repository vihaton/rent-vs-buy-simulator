from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import pandas as pd
import numpy as np
from datetime import datetime

def get_straight_line_distance(address1, address2, manhattan_distance_multiplier = 1.2):
    """
    Calculate straight-line distance (as the crow flies)
    Not actual walking distance, but useful for quick estimates
    """
    geolocator = Nominatim(user_agent="rent-vs-buy-simulator")
    
    loc1 = geolocator.geocode(address1)
    loc2 = geolocator.geocode(address2)
    
    coords1 = (loc1.latitude, loc1.longitude)
    coords2 = (loc2.latitude, loc2.longitude)
    
    distance = geodesic(coords1, coords2).kilometers
    
    # Rough estimate: walking time = distance / 5 km/h
    walking_time_min = (distance / 5) * 60 * manhattan_distance_multiplier
    
    return {
        'distance_km': distance,
        'estimated_walking_min': walking_time_min
    }

def wrangle_actuals(df_actuals: pd.DataFrame, size_brackets = [0, 50, 80, 110, np.inf]) -> pd.DataFrame:
    df_actuals["date"] = pd.to_datetime(df_actuals["Koopdatum"], format="mixed")
    df_actuals["year_quarter"] = df_actuals["date"].dt.to_period("Q")
    df_actuals["year_month"] = df_actuals["date"].dt.to_period("M")
    df_actuals["year_quarter"] = df_actuals["year_quarter"].dt.to_timestamp()
    df_actuals["year_month"] = df_actuals["year_month"].dt.to_timestamp()

    # rename
    df_actuals.rename(columns={
        "Koopsom_EUR": "price",
        "Koopsom_per_m2_EUR": "price_per_sqm",
        "Postcode": "postal_code",
        "Huisnr": "house_number",
        }, inplace=True
    )
    
    # get floor area
    df_actuals["floor_area"] = round(df_actuals["price"] / df_actuals["price_per_sqm"], 1)
    df_actuals["floor_area"] = df_actuals["floor_area"].ffill()
    df_actuals["price_per_sqm"] = df_actuals["price"] / df_actuals["floor_area"]

    df_actuals["size_bracket"] = pd.cut(
        df_actuals["floor_area"],
        bins=size_brackets,
        labels=["Small", "Medium", "Large", "Extra Large"]
    )

    df_actuals["status"] = "sold"
    df_actuals.sort_values("date", inplace=True)

    return df_actuals


def normalize_dates_for_lr(df, date_col="date"):
    X = df[date_col].astype(np.int64).values.reshape(-1, 1)  # Convert to timestamps (units depend on dtype)
    # datetime64 can be in different units (ns, us, ms, s) - convert to seconds
    # For datetime64[ns]: divide by 1e9, for datetime64[us]: divide by 1e6
    time_unit = df[date_col].dtype
    if 'datetime64[ns]' in str(time_unit):
        X = X / 1e9  # nanoseconds to seconds
    elif 'datetime64[us]' in str(time_unit):
        X = X / 1e6  # microseconds to seconds
    elif 'datetime64[ms]' in str(time_unit):
        X = X / 1e3  # milliseconds to seconds
    # else: assume already in seconds
    
    X = X / 86400 / 30.44  # Convert seconds to days to months
    ts_2k = datetime(2000, 1, 1)
    ts_shift = ts_2k.timestamp() / 86400 / 30.44  # Convert seconds to days to months since epoch
    X = X - ts_shift  # Normalize to start from year 2000
    return X
