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

MONTHS = {'january': (1, 31), 'february': (2, 28), 'march': (3, 31), 'april': (4, 30), 'may': (5, 31), 'june': (6, 30), 
              'july': (7, 31), 'august': (8, 31), 'september': (9, 30), 'october': (10, 31), 'november': (11, 30),'december': (12, 31)}

def create_folder(month: str, year: str, index: int = None) -> Path:
    """
    Create a folder for the downloaded files with the given name if it does not exist.

    Args:
        month (str): Month during which the files are taken from.
        year (str): Year during which the files are taken from.

    Returns:
        path to the folder
    """
    if index is None:
        if not os.path.exists(f"images/planet_scope/{year}_{month}"):
            os.mkdir(f"images/planet_scope/{year}_{month}")
        return Path(f"images/planet_scope/{year}_{month}")

    if not os.path.exists(f"images/planet_scope/{year}_{month}/{index:04d}"):
        os.mkdir(f"images/planet_scope/{year}_{month}/{index:04d}")

    return Path(f"images/planet_scope/{year}_{month}/{index:04d}")

def get_orders(filename: str):
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
        coordinates.append(
            [
                {
                "clip": {
                    "aoi": feature['geometry']
                }
            }
        ]
    )
    return coordinates

def get_order_name(month: str, year: str, index: int) -> str:
    return f'{date.today()}_{year}_{month}_{index:04d}'

def query_data_api(month: str, year: int, order):
    month = MONTHS[month]
    client = PlanetAPIClient.query_planet_api(
        start_date = date(year, month[0], 1),
        end_date = date(year, month[0], month[1]),
        bounding_box = order,
        cloud_cover_threshold=10.
    )
    return client 

def download_ps_data(month: str, year: str) -> int:
    dates = MONTHS[month]
    create_folder(month, year)

    with open('images/coordinates/squares_1.geojson', 'r') as f:
        file = json.load(f)

    for i, feature in enumerate(file['features']):
        gdf = gpd.GeoDataFrame.from_features([feature])
        gdf = gdf.set_crs(epsg=4326)

        client = PlanetAPIClient.query_planet_api(
            start_date=date(year, dates[0], 1),
            end_date=date(year, dates[0], dates[1]),
            bounding_box=gdf,
            cloud_cover_threshold=10.
        )

        order_name = get_order_name(month, year, i)
        order_url = client.place_order(
            order_name=order_name,
            processing_tools=[
                {
                    "clip": {
                        "aoi": feature['geometry']
                    }
                }
            ]
        )

        client.check_order_status(
            order_url=order_url,
            loop=True
        )
        
        client.download_order(
            download_dir=create_folder(month, year, i),
            order_name=order_name
        )
  

def test(): # Code works
    with open('images/coordinates/squares_1.geojson', 'r') as f:
        file = json.load(f)

    coords = []
    for feature in file['features']:
        gdf = gpd.GeoDataFrame.from_features([feature])
        gdf = gdf.set_crs(epsg=4326)
        coords.append(gdf)
    
    for coord in coords:
        print(coord)
        client = PlanetAPIClient.query_planet_api(
            start_date = date(2022, 4, 1),
            end_date = date(2022, 4, 28),
            bounding_box = coord,
            cloud_cover_threshold=10.
        )

        order_name = "Julian_test"
        download_dir = Path("images/planet_scope/22_may")
        order_url = client.place_order(
            order_name="Julian_test", 
            processing_tools=[
                {
                "clip": {
                    "aoi": file['features'][0]['geometry']
                    }
                }
            ]
        )

        client.check_order_status(
            order_url=order_url, 
            loop=True
        )

        client.download_order(
            download_dir=download_dir, 
            order_name=order_name
        )

    