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

from sentinel_download import download_eschikon

from utils import (
    MONTHS,
    DAY_IN_MONTH
)
DOTENV_PATH = Path().absolute().parent.joinpath(".env")
load_dotenv(DOTENV_PATH)

settings = get_settings()
settings.PLANET_API_KEY = os.getenv('PLANET_API_KEY')
settings.USE_STAC = True

DATA_DIR = Path().absolute().parent.parent.joinpath("data")
COORDINATE_DIR = DATA_DIR.joinpath("coordinates")

BROATEFAELD_SHP = COORDINATE_DIR.joinpath("Arenenberg", "Broatefaeld.shp")
BRAMENWIES_SHP = COORDINATE_DIR.joinpath("Strickhof", "Bramenwies.shp")
FLUEGENRAIN_SHP = COORDINATE_DIR.joinpath("Strickhof", "Fluegenrain.shp")
HOHRUETI_SHP = COORDINATE_DIR.joinpath("Strickhof", "Hohrueti.shp")
ALTKLOSTER_SHP = COORDINATE_DIR.joinpath("SwissFutureFarm", "Altkloster.shp")
RUETTELI_SHP = COORDINATE_DIR.joinpath("SwissFutureFarm", "Ruetteli.shp")
WITZWIL_SHP = COORDINATE_DIR.joinpath("Witzwil", "Parzelle35.shp")
ESCHIKON_GEOJSON = COORDINATE_DIR.joinpath("bounding_box_eth_centrum_eschikon.geojson")
STRICKHOF_GEOJSON = COORDINATE_DIR.joinpath("bounding_box_strickhof.geojson")

FIELD_DIR = DATA_DIR.joinpath("fields")
FIELD_DIR.mkdir(exist_ok=True)

def planetscope_eschikon(shp_file: Path, curr_dir: Path, date: datetime) -> None:
    field_name = shp_file.name[:-4].lower()

    client = PlanetAPIClient.query_planet_api(
        start_date = date + timedelta(hours = 24),
        end_date = date + timedelta(hours = 24),
        bounding_box = Feature.from_geoseries(gpd.read_file(shp_file).geometry),
        cloud_cover_threshold = 25
    )

    order_name = f"{field_name}"
    order_url = client.place_order(
        order_name=order_name
    )

    client.check_order_status(order_url, loop=True)
    client.download_order(curr_dir, order_url)

def download_sentinel_data(shp_file: Path, year: str, month: str, curr_dir: Path):

    data_dir = curr_dir.joinpath("data")
    metadata_dir = curr_dir.joinpath("metadata")
    plot_dir = curr_dir.joinpath("plot")

    field_name = shp_file.name[:-4].lower()
    print(field_name)

    start_date = datetime(int(year), int(MONTHS[month]), 1)
    end_date = datetime(int(year), int(MONTHS[month]), int(DAY_IN_MONTH[month]))

    metadata_filters: List[Filter] = [
        Filter('cloudy_pixel_percentage', '<=', 25),
        Filter('processing_level', '==', 'Level-2A')]

    feature = Feature.from_geoseries(gpd.read_file(shp_file).geometry)
    mapper_configs = MapperConfigs(
        collection="sentinel2-msi",
        time_start=start_date,
        time_end=end_date,
        feature=feature,
        metadata_filters=metadata_filters
    )
    mapper_configs.to_yaml(metadata_dir.joinpath('sample_mapper_call.yaml'))
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
    scene_10m.to_rasterio(data_dir.joinpath(f'{field_name}_10m.tif'))

    scene_20m = RasterCollection()
    for band in ['red_edge_1', 'nir_2']:
        scene_20m.add_band(scene[band])
    scene_20m.to_rasterio(data_dir.joinpath(f'{field_name}_20m.tif'))

    # save metadata xml to disk
    href_xml = mapper.metadata.assets.iloc[0]['granule-metadata']['href']
    response = urllib.request.urlopen(
        planetary_computer.sign_url(href_xml)).read()
    fpath_xml = metadata_dir.joinpath(href_xml.split('/')[-1])
    with open(metadata_dir.joinpath(field_name), 'wb') as dst:
        dst.write(response)

    # plot scene
    fig = scene_10m.plot_multiple_bands(band_selection=['blue', 'green', 'red'])
    fig.savefig(plot_dir.joinpath(f'{field_name}_10m.png'), dpi=300)
    
    # date
    return mapper.data.timestamps[0]

def main(year: str, month: str):
    # Create dirs 
    sentinel_dir = FIELD_DIR.joinpath("sentinel", year, f"{MONTHS[month]}_{month}")
    planetscope_dir = FIELD_DIR.joinpath("planetscope", year, f"{MONTHS[month]}_{month}")

    sentinel_dir.mkdir(exist_ok=True, parents=True)
    planetscope_dir.mkdir(exist_ok=True, parents=True)

    sentinel_dir.joinpath("data").mkdir(exist_ok=True, parents=True)
    sentinel_dir.joinpath("metadata").mkdir(exist_ok=True, parents=True)
    sentinel_dir.joinpath("plot").mkdir(exist_ok=True, parents=True)

    planetscope_dir.joinpath("data").mkdir(exist_ok=True, parents=True)
    planetscope_dir.joinpath("metadata").mkdir(exist_ok=True, parents=True)
    planetscope_dir.joinpath("plot").mkdir(exist_ok=True, parents=True)


    # Download Data
    broatefaeld_date = download_sentinel_data(BROATEFAELD_SHP, year, month, sentinel_dir)
    bramenwies_date = download_sentinel_data(BRAMENWIES_SHP, year, month, sentinel_dir)
    fluegenrain_date = download_sentinel_data(FLUEGENRAIN_SHP, year, month, sentinel_dir)
    hohrueti_date = download_sentinel_data(HOHRUETI_SHP, year, month, sentinel_dir)
    altkloster_date =download_sentinel_data(ALTKLOSTER_SHP, year, month, sentinel_dir)
    ruetteli_date = download_sentinel_data(RUETTELI_SHP, year, month, sentinel_dir)
    witzwil_date = download_sentinel_data(WITZWIL_SHP, year, month, sentinel_dir)
    eschikon_date = download_sentinel_data(ESCHIKON_GEOJSON, year, month, sentinel_dir)
    strickhof_date = download_sentinel_data(STRICKHOF_GEOJSON, year, month, sentinel_dir)
        
    planetscope_eschikon(BROATEFAELD_SHP, sentinel_dir, broatefaeld_date)
    planetscope_eschikon(BRAMENWIES_SHP, bramenwies_date, broatefaeld_date)
    planetscope_eschikon(FLUEGENRAIN_SHP, fluegenrain_date, broatefaeld_date)
    planetscope_eschikon(HOHRUETI_SHP, hohrueti_date, broatefaeld_date)
    planetscope_eschikon(ALTKLOSTER_SHP, altkloster_date, broatefaeld_date)
    planetscope_eschikon(RUETTELI_SHP, ruetteli_date, broatefaeld_date)
    planetscope_eschikon(WITZWIL_SHP, witzwil_date, broatefaeld_date)
    planetscope_eschikon(ESCHIKON_GEOJSON, eschikon_date, broatefaeld_date)
    planetscope_eschikon(STRICKHOF_GEOJSON, strickhof_date, broatefaeld_date)

def download_sentinel_data():
    download_eschikon()

if __name__ == "__main__":
    download_sentinel_data()
    # main("2022", "mar")
    # main("2022", "apr")
    # main("2022", "may")
    # main("2022", "jun")
    # main("2022", "jul")
    # main("2022", "aug")
    # main("2022", "sep")