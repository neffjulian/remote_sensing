import requests
import json
import os
from pathlib import Path
import geopandas as gpd


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
    with open(f'images/coordinates/{filename}.geojson', 'r') as f:
            file = json.load(f)
    return [feature['geometry']['coordinates'][0] for feature in file['features']]

def download_si_data(month:str, year:int) -> int:
    pass