import requests

def get_station_info(station_id):
    """
    Get information about a specific weather station including its coordinates
    """
    headers = {'User-Agent': '(your-app-name, your-email@example.com)'}
    
    try:
        url = f"https://api.weather.gov/stations/{station_id}"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        props = data['properties']
        coords = data['geometry']['coordinates']
        
        print(f"Station {station_id} Information:")
        print(f"Name: {props['name']}")
        print(f"Coordinates: {coords[1]:.4f}, {coords[0]:.4f}")
        
        return coords[1], coords[0]  # lat, lon
        
    except Exception as e:
        print(f"Error getting station info: {e}")
        return None, None
    

print(get_station_info("KLAX"))