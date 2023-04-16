import requests
import json
import os
from pathlib import Path

url = "http://data.geo.admin.ch/api/stac/v0.9/"

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
    download_dir = create_folder(month, year)
    orders = get_orders("test_coordinates")

    response = requests.get("https://data.geo.admin.ch/api/stac/v0.9/collections")
    print(response.json())
    # 

    # for order in orders:
    #     params = {
    #         "format": "tiff",
    #         "lang": "en",
    #         "bbox": f"{order[0]},{order[1]},{order[2]},{order[3]}",
    #         "layers": "ch.swisstopo.images-swissimage.metadata",
    #         "geometryFormat": "geojson",
    #         "sr": "2056"
    #     }

    #     response = requests.get(url, params=params)
    #     print(response)