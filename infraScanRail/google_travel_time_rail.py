import time
import logging
import pandas as pd
from shapely.geometry import LineString
from googlemaps import Client as GoogleMapsClient


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Assuming get_google_travel_time is defined to retrieve travel time from Google Maps API
def get_google_travel_time(origin_coords, destination_coords, api_key):
    """
    Fetch the travel time using Google Maps API.
    
    :param origin_coords: str, formatted as 'lat,lng' for origin
    :param destination_coords: str, formatted as 'lat,lng' for destination
    :param api_key: str, your Google API key
    :return: travel time in minutes or None if request fails
    """
    gmaps = GoogleMapsClient(key=api_key)
    
    try:
        # Request travel time via train (transit mode)
        directions = gmaps.directions(origin_coords, destination_coords, mode="transit", transit_mode="train")
        
        if directions:
            # Extract travel time from the API response
            travel_time = directions[0]['legs'][0]['duration']['value'] / 60  # Convert from seconds to minutes
            return travel_time
        else:
            logger.warning(f"No directions found for {origin_coords} to {destination_coords}")
            return None
    except Exception as e:
        logger.error(f"Error fetching travel time for {origin_coords} to {destination_coords}: {e}")
        return None

# Function to calculate travel times for all rows in od_matrix
def calculate_travel_times(od_matrix, api_key):
    """
    Calculate and update travel times in the GeoDataFrame using Google Maps API.

    :param od_matrix: pandas DataFrame, GeoDataFrame containing LineString geometries
    :param api_key: str, Google API key
    """
    od_matrix['GoogleTravelTime'] = None  # Initialize the column for travel times
    
    # Iterate through each row to calculate travel time
    for idx, row in od_matrix.iterrows():
        line_geom = row['Geometry']  # Geometry column, ensure correct name
        
        if isinstance(line_geom, LineString):
            # Extract the origin (start) and destination (end) coordinates
            origin_coords = f"{line_geom.coords[0][1]},{line_geom.coords[0][0]}"  # Format as 'lat,lng'
            destination_coords = f"{line_geom.coords[-1][1]},{line_geom.coords[-1][0]}"  # Format as 'lat,lng'
            
            # Fetch the travel time using the Google Maps API
            travel_time = get_google_travel_time(origin_coords, destination_coords, api_key)
            
            # Update the GeoDataFrame with the calculated travel time if available
            if travel_time is not None:
                od_matrix.at[idx, 'GoogleTravelTime'] = travel_time
            else:
                logger.warning(f"Travel time not available for row {idx} (origin: {origin_coords}, destination: {destination_coords})")
        
        # Adjust sleep to prevent hitting API rate limits (Google recommends 1 request per second)
        time.sleep(1)
    
    return od_matrix

# Example usage
if __name__ == "__main__":
    # Assuming `od_matrix` is a pandas DataFrame containing your data
    api_key = 'AIzaSyCFByVXpNNrVY_HATr7NaJk2m3Tuix1u2Y'  # Ensure you use a valid API key

    # Calculate the travel times and update od_matrix
    od_matrix = calculate_travel_times(od_matrix, api_key)
    
    # Print the updated GeoDataFrame to check results
    print(od_matrix[['OriginStation', 'DestinationStation', 'GoogleTravelTime']].head())
