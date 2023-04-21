# Based on "scripts/planet_download.py" from eodal repo
import os
import geopandas as gpd
from eodal.downloader.planet_scope import PlanetAPIClient
from eodal.config import get_settings
from datetime import date
from dotenv import load_dotenv
import json
from pathlib import Path

def test(): # Code works
    with open('images/coordinates/squares_1.geojson', 'r') as f:
        file = json.load(f)

    coords = []
    for feature in file['features']:
        gdf = gpd.GeoDataFrame.from_features([feature])
        gdf = gdf.set_crs(epsg=4326)
        coords.append(gdf)
    
    coord = coords[0]
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