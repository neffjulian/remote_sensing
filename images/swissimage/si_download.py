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
    download_dir = create_folder(month, year)
    # orders = get_orders("zhr_coordinates")
    gdf = gpd.read_file('images/coordinates/zhr_coordinates.geojson')
    bbox = gdf.total_bounds
    image_size = '500,500'
    image_format = 'png'
    layers = 'ch.swisstopo.swissimage-product'

    url = "https://api3.geo.admin.ch/rest/services/api/MapServer"


    params = {
         'bbox': ','.join(str(x) for x in bbox),
         'size': image_size,
         'format': image_format,
         'layers': layers
    }

    response = requests.get(url, params=params)

    # print(response.content)
    with open('swissimage.png', 'wb') as f:
         f.write(response.content)
    

    