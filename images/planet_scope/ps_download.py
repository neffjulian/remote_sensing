import os
import json
from pathlib import Path
from datetime import datetime, date, timedelta

import numpy as np
import geopandas as gpd
from eodal.downloader.planet_scope import PlanetAPIClient
from eodal.config import get_settings
from dotenv import load_dotenv

load_dotenv('.env')
settings = get_settings()
settings.PLANET_API_KEY = os.getenv('PLANET_API_KEY')
settings.USE_STAC = True

COORDINATE_PATH = Path("images/coordinates/squares_1.geojson")

def create_folder(month: str, year: str, index: int = None) -> Path:
    """
    Creates a new folder for storing PlanetScope images.

    Args:
        month (str): The month of the images to be stored, e.g. "aug".
        year (str): The year of the images to be stored, e.g. "22".
        index (int, optional): An optional index number for the folder.
            If provided, the folder will be named "{year}_{month}/{index:04d}".
            If not provided, the folder will be named "{year}_{month}".

    Returns:
        Path: A Path object representing the path to the newly created folder.
    """
    folder_location = f"images/planet_scope/{year}_{month}"
    folder_path = Path(folder_location)

    if index is not None:
        folder_path = folder_path.joinpath(f"{index:04d}")

    if not folder_path.exists():
        folder_path.mkdir()

    return folder_path

def create_stats(month: str, year: str) -> None:
    """Create a CSV file with empty statistics data for PlanetScope images.

    Args:
        month (str): The month of the images to be stored, e.g. "aug".
        year (str): The year of the images to be stored, e.g. "22".

    Returns:
        None

    """
    stats_location = f"images/planet_scope/stats/{year}_{month}_stats.csv"
    stats_path = Path(stats_location)

    header = "index, date, coordinate, B02, B03, B04, B05, B8A"
    data = np.empty((296, 8), dtype=object)
    data[:, 0] = np.arange(0, 296)
    np.savetxt(stats_path, data, header=header, delimiter=',', fmt='%s', encoding="utf-8")

def write_date_to_stats(month: str, year: str, date: str, index: int) -> None:
    """Write a date to the statistics CSV file for a specified index.

    Args:
        month (str): The month of the images to be stored, e.g. "aug".
        year (str): The year of the images to be stored, e.g. "22".
        date (str): A string representing the date in YYYY-MM-DD format, e.g. '2022-01-01' for January 1st, 2022.
        index (int): An integer representing the index of the row to update in the CSV file.

    Returns:
        None

    """
    stats_location = f"images/planet_scope/stats/{year}_{month}_stats.csv"
    stats_path = Path(stats_location)

    header = "index, date, coordinate, B02, B03, B04, B05, B8A"
    data = np.loadtxt(stats_path, delimiter = ',', dtype=object)

    data[index, 1] = date

    np.savetxt(stats_path, data, header=header, delimiter=',', fmt='%s', encoding="utf-8")

def get_coordinates() -> list:
    """
    Retrieves a list of coordinates from a GeoJSON file.

    Returns:
        List: A list of lists of coordinates.
    """    
    with open(COORDINATE_PATH, "r") as f:
        geojson_coordinates = json.load(f)
    
    coordinates = []
    for feature in geojson_coordinates["features"]:
        gdf = gpd.GeoDataFrame.from_features([feature])
        gdf = gdf.set_crs(epsg=4326)
        coordinates.append(gdf)

    return coordinates

def get_s2_dates(month: str, year: str) -> np.ndarray:
    """
    Load Sentinel-2 dates from a CSV file for the specified month and year.

    Parameters:
        month (str): A string representing the month in the format of "jan", "feb", etc.
        year (str): A string representing the year in the format of "yy".

    Returns:
        s2_dates (numpy.ndarray): A numpy array containing the index and date of Sentinel-2 data for the given month and year.

    Raises:
        Exception: If the CSV file for the given month and year does not exist in the specified directory.
    """
    s2_stats_location = f"images/sentinel_2/stats/{year}_{month}_stats.csv"
    s2_stats_path = Path(s2_stats_location)

    if not s2_stats_path.exists():
        raise Exception("Stats currently do not exist. Check if the data is already downloaded")
    
    s2_stats = np.loadtxt(s2_stats_path, delimiter = ',', dtype=object)
    s2_dates = s2_stats[:, :2] # Gives the index and date of the data
    return s2_dates

def filter_coordinates(s2_dates: list, coordinates: list) -> list:
    """
    Filters coordinates such that we only take values where we have existing s2 data.

    Parameters:
        s2_dates (List[List[str]]): A list of s2_dates, where each row contains an index and a date in the format of "yyyy-mm-dd".
        coordinates (List[Tuple[float, float]]): A list of coordinates, where each coordinate is a tuple of latitude and longitude.

    Returns:
        filtered_coordinates (List[Tuple[int, datetime.date, Tuple[float, float]]]): A list of filtered coordinates, where each row contains an index, date, and a tuple of latitude and longitude.
    """
    filtered_coordinates = []
    for index, coordinate in enumerate(coordinates):
        try:
            current_row = s2_dates[s2_dates[:, 0] == str(index)]
            current_index = int(current_row[0][0])
            current_date = datetime.strptime(current_row[0][1], '%Y-%m-%d').date()

            filtered_coordinates.append((current_index, current_date, coordinate))
        except:
            continue
    return filtered_coordinates

def get_processing_tools() -> list:
    """
    Reads a JSON file containing feature geometries and creates a list of coordinates such that we can use it for clipping.

    Returns:
        processing_tools (list): A list of processing tools, where each processing tool is a list containing a dictionary with a single key "clip" and its corresponding value.
    """
    with open(COORDINATE_PATH, 'r') as f:
        file = json.load(f)

    processing_tools = []
    for feature in file['features']:
        processing_tools.append(
            [
                {
                "clip": {
                    "aoi": feature['geometry']
                }
            }
        ]
    )
    return processing_tools

def get_order_name(month: str, year: str, index: str) -> str:
    return f'{date.today()}_{year}_{month}_{int(index):04d}'

def place_ps_order(month: str, year: str) -> None:
    """
    Places an order for PlanetScope imagery for a given month and year.

    Parameters:
        month (str): A string representing the month in the format of "jan", "feb", etc.
        year (str): A string representing the year in the format of "yy".

    Returns:
        None.
    """
    create_folder(month, year)
    coordinates = get_coordinates()
    s2_dates = get_s2_dates(month, year)
    filtered_coordinates = filter_coordinates(s2_dates, coordinates)
    processing_tools = get_processing_tools()

    orders = np.empty((0, 3), dtype=np.dtype([("index", np.int), 
                                              ("order_name", "U100"), 
                                              ("order_url", "U100")]))
    for i, date, gdf in filtered_coordinates:
        index = int(i)

        added_days = 0
        could_place_order = False
        while not could_place_order:
            try:
                client = PlanetAPIClient.query_planet_api(
                    start_date = date - timedelta(days=added_days),
                    end_date = date + timedelta(days=added_days),
                    bounding_box = gdf,
                    cloud_cover_threshold = 5.
                )

                order_name = get_order_name(month, year, index)
                order_url = client.place_order(
                    order_name=order_name,
                    processing_tools=processing_tools[index]
                )
                could_place_order = True
            except:
                added_days += 1
        print(f"Added {added_days} days to order {i}")

        current_order = np.array([(i, order_name, order_url)], dtype=orders.dtype)
        orders = np.append(orders, current_order)
    
    order_path = Path(f'images/planet_scope/{year}_{month}/orders.csv')
    np.savetxt(order_path, orders, delimiter=',', fmt='%d,%s,%s', header="index, order_name, order_url")

def download_ps_data(month: str, year: str) -> None:
    """
    Downloads PlanetScope imagery data for a given month and year.

    Parameters:
        month (str): A string representing the month in the format of "jan", "feb", etc.
        year (str): A string representing the year in the format of "yy".

    Returns:
        None.
    """
    orders_path = Path(f'images/planet_scope/{year}_{month}/orders.csv')
    orders = np.loadtxt(orders_path, delimiter=',', dtype=np.dtype([("index", np.int), 
                                                                    ("order_name", "U100"), 
                                                                    ("order_url", "U100")]))
    
    # TODO: This fails currently. Show Lukas 
    client = PlanetAPIClient
    for index, order_name, order_url in orders:
        download_dir = create_folder(month, year, index)
        status = client.check_order_status(
            order_url=order_url
        )
        if status == "Failed":
            print(f"Order {index} failed.")
            continue

        client.download_order(
            download_dir=download_dir,
            order_name=order_name
        )