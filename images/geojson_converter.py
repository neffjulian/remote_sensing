import math
import json

def point_to_square(point: dict, km: float = 1) -> dict:
    """
    Takes a GeoJSON point feature and returns a GeoJSON polygon feature
    representing a square of size `km` centered at the point.

    Args:
        point (dict): A GeoJSON point feature containing the coordinates of the point.
        km (float): A float indicating the size of the square in kilometers (default 1).

    Returns:
        dict: A GeoJSON polygon feature representing a square of size `km` centered at the point.
    """
    coordinate = point['geometry']['coordinates']
    long, lat = coordinate[0], coordinate[1]
    change_long = 360 / (math.cos(math.radians(lat)) * 40075) * km
    change_lat = 360 / (math.cos(math.radians(long)) * 40075) * km
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
    """
    Converts a GeoJSON point feature collection into a GeoJSON polygon feature
    collection, with each point converted into a square polygon of a given size.

    Reads points from the 'points.geojson' file, and writes squares to the
    'squares.geojson' file.
    """
    with open('../coordinates/points.geojson', 'r') as file:
        points = json.load(file)
    squares = {
        "type": "FeatureCollection",
        "features": [point_to_square(point) for point in points['features']]
    }
    with open('../coordinates/squares.geojson', 'x') as file:
        json.dump(squares, file)

if __name__ == "__main__":
    # Use a tool to create a list of points such as 'geojson.io' and then copy the list into 'points.geojson'
    convert_to_squares()