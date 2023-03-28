import math
import json

def point_to_square(point: dict, km: float = 1):
    """
    Takes a GeoJSON point feature and returns a GeoJSON polygon feature
    representing a square of size `km` centered at the point.

    Args:
        point: A GeoJSON point feature containing the coordinates of the point.
        km: A float indicating the size of the square in kilometers (default 1).

    Returns:
        A GeoJSON polygon feature representing a square of size `km` centered at the point.
    """
    # Get the coordinates of the point
    coordinate = point['geometry']['coordinates']
    # Extract the longitude and latitude values
    long, lat = coordinate[0], coordinate[1]
    # Calculate the change in longitude and latitude required to generate the square
    change_long = 360 / (math.cos(math.radians(lat)) * 40075) * km
    change_lat = 360 / (math.cos(math.radians(long)) * 40075) * km
    # Create a GeoJSON polygon feature representing the square
    square = {
        "type": "Feature",
        "properties": {},
        "geometry": {
            "coordinates": [
                [
                    [long - change_long, lat + change_lat],
                    [long + change_long, lat + change_lat],
                    [long + change_long, lat - change_lat],
                    [long - change_long, lat - change_lat],
                    [long - change_long, lat + change_lat]
                ]
            ],
            "type": "Polygon"
        }
    }
    return square

def convert_to_squares():
    # Load the input GeoJSON file containing point features
    with open('coordinates/points.geojson', 'r') as file:
        points = json.load(file)
    # Convert each point feature to a square feature
    squares = {
        "type": "FeatureCollection",
        "features": [point_to_square(point) for point in points['features']]
    }
    # Write the output GeoJSON file containing square features
    with open('coordinates/squares.geojson', 'x') as file:
        json.dump(squares, file)

def get_coordinates_from_points():
    # Open the GeoJSON file containing point features
    with open('coordinates/points.geojson', 'r') as file:
        # Load the file as a JSON object
        points = json.load(file)

    # Extract the coordinates from each point feature and add to a list
    coordinates = []
    for point in points['features']:
        point_coords = point['geometry']['coordinates']
        coordinates.append(point_coords)

    # Return the list of coordinates
    return coordinates

if __name__ == "__main__":
    # Use a tool to create a list of points such as 'geojson.io' and then copy the list into 'points.geojson'
    convert_to_squares()