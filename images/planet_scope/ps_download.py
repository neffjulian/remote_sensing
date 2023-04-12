# Based on "scripts/planet_download.py" from eodal repo
import os
import geopandas as gpd
from eodal.downloader.planet_scope import PlanetAPIClient
from eodal.config import get_settings
from datetime import date
from dotenv import load_dotenv
import json
from pathlib import Path

load_dotenv('.env')
settings = get_settings()
settings.PLANET_API_KEY = os.getenv('PLANET_API_KEY')
settings.USE_STAC = True

def create_folder(month: str, year: str) -> Path:
    """
    Create a folder for the downloaded files with the given name if it does not exist.

    Args:
        month (str): Month during which the files are taken from.
        year (str): Year during which the files are taken from.

    Returns:
        path to the folder
    """
    if not os.path.exists(f"images/planet_scope/{year}_{month}"):
        os.mkdir(f"images/planet_scope/{year}_{month}")

    return Path(f"images/planet_scope/{year}_{month}")

def get_orders(filename: str) -> list:
    """
    Uses a geojson file and returns a list of each "feature"

    Args:
        filename (str): Name of the geojson file
    """
    with open(f'images/coordinates/{filename}.geojson', 'r') as f:
        file = json.load(f)
    orders = []
    for feature in file['features']: 
        gdf = gpd.GeoDataFrame.from_features([feature])
        print(gdf.to_json())
        gdf = gdf.set_crs(epsg=4326)
        orders.append(gdf)
    return orders

def get_coordinates(filename: str) -> list:
    """
    Uses a geojson file and returns a list of each polygon coordinate

    Args:
        filename (str): Name of the geojson file
    """
    with open(f'images/coordinates/{filename}.geojson', 'r') as f:
        file = json.load(f)
        f.close()
    coordinates = []
    for feature in file['features']: 
        coordinate = {
            "type": "Polygon",
            "coordinates": feature['geometry']['coordinates']
        }
        coordinates.append(coordinate)
    return coordinates

def get_order_name(month: str, year: str, index: int) -> str:
    return f'{date.today()}_Julian_{year}_{month}_{index:04d}'

def query_data_api(month: str, year: int, order: gpd.GeoDataFrame) -> PlanetAPIClient:
    MONTHS = {
        'january': (1, 31),
        'february': (2, 28),
        'march': (3, 31),
        'april': (4, 30),
        'may': (5, 31),
        'june': (6, 30),
        'july': (7, 31),
        'august': (8, 31),
        'september': (9, 30),
        'october': (10, 31),
        'november': (11, 30),
        'december': (12, 31)
    }
    month = MONTHS[month]
    client = PlanetAPIClient.query_planet_api(
        start_date = date(year, month[0], 1),
        end_date = date(year, month[0], month[1]),
        bounding_box = order,
        cloud_cover_threshold=50.
    )
    return client 

def download_ps_data(month: str, year: int) -> int: # E.g. month = "may", year = "22"
    download_dir = create_folder(month, year)
    orders = get_orders("test_coordinates")
    coordinates = get_coordinates("test_coordinates")

    for index, order in enumerate(orders):
        client = query_data_api(month, year, order)
        order_name = get_order_name(month, year, index)
        order_url = client.place_order(
            order_name = order_name,
            processing_tools = [coordinates[index]])
        status = client.check_order_status(
            order_url = order_url, 
            loop = True)
        print(index, status)
        client.download_order(
            download_dir = download_dir,
            order_name = order_name
        )