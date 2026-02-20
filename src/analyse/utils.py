from geopy.geocoders import Nominatim
from geopy.distance import geodesic

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
