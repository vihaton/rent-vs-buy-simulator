from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import pandas as pd

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

def wrangle_actuals(df_actuals: pd.DataFrame) -> pd.DataFrame:
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
        }, inplace=True
    )
    
    # get floor area
    df_actuals["floor_area"] = round(df_actuals["price"] / df_actuals["price_per_sqm"], 1)
    df_actuals["floor_area"] = df_actuals["floor_area"].ffill()
    df_actuals["price_per_sqm"] = df_actuals["price"] / df_actuals["floor_area"]
    df_actuals["status"] = "sold"
    df_actuals.sort_values("date", inplace=True)

    return df_actuals
