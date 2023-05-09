import json
import math
from pathlib import Path

import pandas as pd
from datetime import datetime

COORDINATE_DIR = Path().absolute().parent.parent.joinpath('data', 'coordinates')

MONTHS = {"jan": "01", "feb": "02", "mar": "03", "apr": "04", 
          "may": "05", "jun": "06", "jul": "07", "aug": "08", 
          "sep": "09", "oct": "10", "nov": "11", "dec": "12"}

DAY_IN_MONTH = {"jan": "31", "feb": "28", "mar": "31", "apr": "30", 
          "may": "31", "jun": "30", "jul": "31", "aug": "31", 
          "sep": "30", "oct": "31", "nov": "30", "dec": "31"}

def get_dates(year: str, month: str) -> tuple:
    """
    Compute the start and end dates for a given year and month.

    Parameters:
    - year (str): A string representation of the year (e.g., '2022').
    - month (str): A string representation of the month (e.g., '01' or 'January').

    Returns:
    - A tuple with two datetime objects: The start date and end date for the given month and year.

    This function takes a year and a month as input and returns the corresponding start and end dates as datetime objects.
    The function uses a dictionary to map month names and abbreviations to their corresponding integer values, and the
    DAY_IN_MONTH dictionary to determine the last day of the month.
    """
    start_date = datetime(int(year), int(MONTHS[month]), 1)
    end_date = datetime(int(year), int(MONTHS[month]), int(DAY_IN_MONTH[month]))
    return start_date, end_date

def get_point_coordinates(coordinate_file: str) -> list:
    """
    Extract the coordinates of points of interest from a geojson file.

    Parameters:
    - coordinate_file (str): The name of the geojson file containing the coordinates.

    Returns:
    - A list of tuples: Each tuple contains the (longitude, latitude) coordinates of a point of interest.

    This function reads the specified geojson file and returns a list of the coordinates of the points of interest
    specified in the file. The coordinates are returned as (longitude, latitude) tuples in a list.
    """
    point_location = COORDINATE_DIR.joinpath(coordinate_file)
    point_file = point_location.open()
    point_geojson = json.load(point_file)
    point_coordinates = []

    for point in point_geojson["features"]:
        coordinates = point["geometry"]["coordinates"]
        point_coordinates.append(coordinates)

    return point_coordinates

def create_dirs(dir_directory: str, year: str, month: str):
    """
    Creates the necessary directory structure for storing data for a specific year and month.

    Parameters:
    - dir_directory (str): The directory in which to create the subdirectories.
    - year (str): The year for which data is being downloaded.
    - month (str): The month for which data is being downloaded.

    Returns:
    - data_dir (Path): The path to the 'data' subdirectory.
    - plot_dir (Path): The path to the 'plot' subdirectory.
    - metadata_dir (Path): The path to the 'metadata' subdirectory.
    """
    satellite_dir = Path(dir_directory).joinpath(f'{year}', f'{MONTHS[month]}_{month}')
    data_dir = satellite_dir.joinpath('data')
    plot_dir = satellite_dir.joinpath('plot')
    metadata_dir = satellite_dir.joinpath('metadata')

    data_dir.mkdir(exist_ok=False, parents=True)
    plot_dir.mkdir(exist_ok=False, parents=True)
    metadata_dir.mkdir(exist_ok=False, parents=True)

    return data_dir, plot_dir, metadata_dir

def write_date_to_csv(metadata_dir: Path, index: int, date: str, coordinate_x: float, coordinate_y) -> None:
    """
    Appends a new row with index, date, x and y coordinates to a CSV file stored in the metadata directory.

    Parameters:
    - metadata_dir (Path): Path object for the metadata directory.
    - index (int): Index of the new row to be added.
    - date (str): Date for the new row to be added.
    - coordinate_x (float): x-coordinate for the new row to be added.
    - coordinate_y (float): y-coordinate for the new row to be added.

    Returns:
    - None: The function appends a new row to a CSV file and does not return any value.
    """
    file_location = metadata_dir.joinpath("dates.csv")
    
    try:
        df = pd.read_csv(file_location)
    except:
        df = pd.DataFrame(columns=['index', 'date', 'x', 'y'])

    new_row = pd.DataFrame({
        'index': [index],
        'date': [date],
        'x': [coordinate_x],
        'y': [coordinate_y]
    })

    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(file_location, header=True, index=False)


def point_to_square(point: dict, length: float = 2) -> dict:
    """
    Takes a GeoJSON point feature and returns a GeoJSON polygon feature
    representing a square of size `km` centered at the point.

    Args:
        point (dict): A GeoJSON point feature containing the coordinates of the point.
        length (float): A float indicating the length of the square in kilometers (default 2) for a 2x2 km square.

    Returns:
        dict: A GeoJSON polygon feature representing a square of size `km` centered at the point.
    """
    coordinate = point['geometry']['coordinates']
    longitude, latitude = coordinate[0], coordinate[1]

    change_longitude = 360 / (math.cos(math.radians(latitude)) * 40075) * (length / 2)
    change_latitude = 360 / (math.cos(math.radians(longitude)) * 40075) * (length / 2)

    square = {
        "type": "Feature",
        "properties": {},
        "geometry": {
            "coordinates": [
                [
                    [longitude - change_longitude, latitude + change_latitude],
                    [longitude + change_longitude, latitude + change_latitude],
                    [longitude + change_longitude, latitude - change_latitude],
                    [longitude - change_longitude, latitude - change_latitude],
                    [longitude - change_longitude, latitude + change_latitude]
                ]
            ],
            "type": "Polygon"
        }
    }

    return square

def convert_to_squares(name: str, length: int = 2) -> None:
    """
    Converts a GeoJSON point feature collection into a GeoJSON polygon feature
    collection, with each point converted into a square polygon of a given size.

    Reads points from the 'points.geojson' file, and writes squares to the
    'squares.geojson' file.
    """

    source_dir = COORDINATE_DIR.joinpath(name)

    with open(source_dir, 'r') as source_file:
        points = json.load(source_file)

    squares = {
        "type": "FeatureCollection",
        "features": [point_to_square(point, length) for point in points['features']]
    }

    return squares

def get_coordinates(file_name: str) -> list:
    """
    Retrieves a list of coordinates from a GeoJSON file.

    Args:
        file_name: The filename of the coordinate file (e.g. "squares.geojson")

    Returns:
        List: A list of lists of coordinates.
    """    
    file_location = COORDINATE_DIR.joinpath(file_name)
    coordinate_file = file_location.open()
    coordinate_geojson = json.load(coordinate_file)
    
    coordinates = []
    for feature in coordinate_geojson["features"]:
        coordinate = feature["geometry"]["coordinates"]
        coordinates.append(coordinate)

    return coordinates