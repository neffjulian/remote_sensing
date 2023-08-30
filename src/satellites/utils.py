"""
Collection of utility functions used for downloading and processing
satellite data from Sentinel-2 and PlanetScope.

@date: 2023-08-30
@author: Julian Neff, ETH Zurich

Copyright (C) 2023 Julian Neff

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import json
import math
from pathlib import Path

# Set the base directory for coordinates
COORDINATE_DIR = Path(__file__).parent.parent.parent.joinpath('data', 'coordinates')

# Mapping of month abbreviations to month numbers
MONTHS = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "may": "05", "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12"
}

def create_dirs(dir_directory: str, year: str, month: str) -> tuple:
    """
    Create the necessary directories for data, plots, and metadata.
    
    Args:
        dir_directory (str): The base directory path.
        year (str): The year.
        month (str): The month abbreviation.

    Returns:
        tuple: The paths of the created directories (data, plot, metadata).
    """
    # Create the paths for the satellite, data, plot, and metadata directories
    satellite_dir = Path(dir_directory).joinpath(f'{year}', f'{MONTHS[month]}_{month}')
    data_dir = satellite_dir.joinpath('data')
    plot_dir = satellite_dir.joinpath('plot')
    metadata_dir = satellite_dir.joinpath('metadata')

    # Create the directories if they don't exist
    data_dir.mkdir(exist_ok=False, parents=True)
    plot_dir.mkdir(exist_ok=False, parents=True)
    metadata_dir.mkdir(exist_ok=False, parents=True)

    # Return the paths of the created directories
    return data_dir, plot_dir, metadata_dir

def point_to_square(point: dict, length: float = 2) -> dict:
    """
    Convert a point to a square polygon.

    Args:
        point (dict): The point geometry with coordinates.
        length (float): The length of the square's sides (default: 2).

    Returns:
        dict: The square polygon feature.
    """
    # Extract the longitude and latitude coordinates from the point
    coordinate = point['geometry']['coordinates']
    properties = point['properties']
    longitude, latitude = coordinate[0], coordinate[1]

    # Calculate the change in longitude and latitude to create the square
    change_longitude = 360 / (math.cos(math.radians(latitude)) * 40075) * (length / 2)
    change_latitude = 360 / (math.cos(math.radians(longitude)) * 40075) * (length / 2)

    # Create the square polygon feature
    square = {
        "type": "Feature",
        "properties": properties,
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [longitude - change_longitude, latitude + change_latitude],
                    [longitude + change_longitude, latitude + change_latitude],
                    [longitude + change_longitude, latitude - change_latitude],
                    [longitude - change_longitude, latitude - change_latitude],
                    [longitude - change_longitude, latitude + change_latitude]
                ]
            ]
        }
    }

    return square

def convert_to_squares(name: str, length: int = 2) -> dict:
    """
    Convert a set of points to square polygons.

    Args:
        name (str): The name of the source file containing the points.
        length (int): The length of the square's sides (default: 2).

    Returns:
        dict: The feature collection of square polygons.
    """
    # Get the full path of the source file
    source_dir = COORDINATE_DIR.joinpath(name)

    # Load the points from the source file
    with open(source_dir, 'r') as source_file:
        points = json.load(source_file)

    # Convert each point to a square using point_to_square function
    squares = {
        "type": "FeatureCollection",
        "features": [point_to_square(point, length) for point in points['features']]
    }

    return squares