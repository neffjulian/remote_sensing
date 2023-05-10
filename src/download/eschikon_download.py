import os
import argparse
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import List
import urllib

from dotenv import load_dotenv
from eodal.downloader.planet_scope import PlanetAPIClient
from eodal.config import get_settings
import geopandas as gpd
import planetary_computer
from eodal.config import get_settings
from eodal.core.raster import RasterCollection
from eodal.core.sensors.sentinel2 import Sentinel2
from eodal.mapper.feature import Feature
from eodal.mapper.filter import Filter
from eodal.mapper.mapper import Mapper, MapperConfigs

from .utils import (
    MONTHS,
    DAY_IN_MONTH
)

COORDINATES_PATH = Path().absolute().parent.joinpath("data", "coordinates", "bounding_box_eth_centrum_eschikon.geojson")

DATA_DIR = Path().absolute().parent.joinpath("data", "raw", "eschikon")
DOTENV_PATH = Path().absolute().parent.joinpath(".env")
load_dotenv(DOTENV_PATH)

settings = get_settings()
settings.PLANET_API_KEY = os.getenv('PLANET_API_KEY')
settings.USE_STAC = True

def create_dirs(year: str, month: str) -> None:
    curr_dir = DATA_DIR.joinpath(f"{year}", f"{MONTHS[month]}_{month}")
    curr_dir.mkdir(parents=True, exist_ok=True)
    return curr_dir

def sentinel_eschikon(curr_dir: Path, start_date: datetime, end_date: datetime) -> None:
    metadata_filters: List[Filter] = [
        Filter('cloudy_pixel_percentage', '<=', 25),
        Filter('processing_level', '==', 'Level-2A')]
    
    feature = Feature.from_geoseries(gpd.read_file(COORDINATES_PATH).geometry)
    
    mapper_configs = MapperConfigs(
        collection="sentinel2-msi",
        time_start=start_date,
        time_end=end_date,
        feature=feature,
        metadata_filters=metadata_filters
    )

    mapper_configs.to_yaml(curr_dir.joinpath('sample_mapper_call.yaml'))

    mapper = Mapper(mapper_configs)
    mapper.query_scenes()
    mapper.metadata

    mapper.metadata = mapper.metadata[
        mapper.metadata.cloudy_pixel_percentage ==
        mapper.metadata.cloudy_pixel_percentage.min()].copy()
    
    scene_kwargs = {
        'scene_constructor': Sentinel2.from_safe,
        'scene_constructor_kwargs': {
            'band_selection': ["B02", "B03", "B04", "B05", "B8A"]}}
    
    mapper.load_scenes(scene_kwargs=scene_kwargs)
    scene = mapper.data[mapper.data.timestamps[0]]

    scene_10m = RasterCollection()
    for band in ['blue', 'green', 'red']:
        scene_10m.add_band(scene[band])
    scene_10m.to_rasterio(curr_dir.joinpath(f'scene_10m.tif'))

    scene_20m = RasterCollection()
    for band in ['red_edge_1', 'nir_2']:
        scene_20m.add_band(scene[band])
    scene_20m.to_rasterio(curr_dir.joinpath(f'scene_20m.tif'))

    # save metadata xml to disk
    href_xml = mapper.metadata.assets.iloc[0]['granule-metadata']['href']
    response = urllib.request.urlopen(
        planetary_computer.sign_url(href_xml)).read()
    fpath_xml = curr_dir.joinpath(href_xml.split('/')[-1])
    with open(fpath_xml, 'wb') as dst:
        dst.write(response)

    # plot scene
    fig = scene_10m.plot_multiple_bands(band_selection=['blue', 'green', 'red'])
    fig.savefig(curr_dir.joinpath(f'scene_10m.png'), dpi=300)
    
    # date
    return mapper.data.timestamps[0]

def planetscope_eschikon(curr_dir: Path, start_date: datetime, end_date: datetime) -> None:
    client = PlanetAPIClient.query_planet_api(
        start_date=start_date,
        end_date=end_date,
        bounding_box=COORDINATES_PATH,
        cloud_cover_threshold=25
    )

    order_name = f"eschikon_{start_date.year}_{start_date.month}"
    order_url = client.place_order(
        order_name=order_name
    )

    client.check_order_status(order_url, loop=True)
    client.download_order(curr_dir, order_url)

def main(year: str, month: str):
    if not (2017 <= int(year) <= 2022):
        raise ValueError(f"Year invalid ('{year}'). Use a value between '2017'  and '2022'.")
    
    if month not in MONTHS:
        raise ValueError(f"Month invalid ('{month}'). Use one out of {list(MONTHS)}.")
    
    start_date = datetime(year, int(MONTHS[month]), 1)
    end_date = datetime(year, int(MONTHS[month]), int(DAY_IN_MONTH[month]))
    curr_dir = create_dirs(year, month)

    date = sentinel_eschikon(curr_dir, start_date, end_date)
    curr_date = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
    
    start_date = curr_date + timedelta(hours = 12)
    end_date = curr_date - timedelta(hours = 12)

    planetscope_eschikon(curr_dir, start_date, end_date)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", required=True, type=str)
    parser.add_argument("--month", required=True, type=str)

    args = parser.parse_args()
    main(**vars(args))