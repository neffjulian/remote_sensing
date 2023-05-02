from pathlib import Path
from typing import List
from datetime import datetime
import json

import pandas as pd

import urllib
import geopandas as gpd
import planetary_computer
from eodal.config import get_settings
from eodal.core.raster import RasterCollection
from eodal.core.sensors.sentinel2 import Sentinel2
from eodal.mapper.feature import Feature
from eodal.mapper.filter import Filter
from eodal.mapper.mapper import Mapper, MapperConfigs

from .utils import (
    convert_to_squares,
    create_dirs,
    write_date_to_csv,
    get_dates,
    get_point_coordinates,
)

Settings = get_settings()
Settings.USE_STAC = True

DATA_DIR = Path().absolute().parent.joinpath("data", "raw", "sentinel")
COORDINATE_DIR = Path().absolute().parent.joinpath("data", "coordinates")
COLLECTION = "sentinel2-msi"

def get_features(coordinates_location: Path) -> list:
    """
    Extracts geographic features from a geojson file.

    Parameters:
    - coordinates_location (Path): Path to the geojson file containing the geographic coordinates.

    Returns:
    - features (list): A list of features extracted from the input geojson file.

    This function reads the input geojson file and extracts geographic features from it. The function uses the `convert_to_squares`
    function to process the input geojson file and extract the coordinates. It then creates a GeoDataFrame from the coordinates
    and creates a `Feature` object for each feature in the GeoDataFrame. The resulting list of features is returned as the output of
    the function.
    """
    coordinates = convert_to_squares(coordinates_location)

    features = []
    for feature in coordinates['features']:
        gdf = {"type": "FeatureCollection", "features": [feature]}
        feat = Feature("geometry", gpd.GeoDataFrame.from_features(gdf)['geometry'].iloc[0], 4326, {})
        features.append(feat)
    return features

def download_sentinel_data(coordinate_file: str, year: str, month: str) -> None:
    """
    Downloads Sentinel-2 data for a specific year and month, based on coordinates specified in a geojson file.

    Parameters:
    - coordinate_file (str): Path to the geojson file containing the coordinates of the area of interest.
    - year (str): The year for which data should be downloaded.
    - month (str): The month for which data should be downloaded.

    Returns:
    - None: The function saves downloaded data to disk and does not return any value.

    This function uses Sentinel-2 data accessed through the Planetary Computer API to download data for a specific year and
    month. The function first extracts the geographic coordinates from the input geojson file and downloads the least
    cloudy scene available from the STAC API for each set of coordinates. The downloaded data is then saved to disk in
    both GeoTIFF and XML formats. Additionally, a PNG image is created for each downloaded scene, and the dates and coordinates about
    the downloaded data is saved to a CSV file.
    """
    data_dir, plot_dir, metadata_dir = create_dirs(DATA_DIR, year, month)
    start_date, end_date = get_dates(year, month)
    features = get_features(coordinate_file)
    point_coordinates = get_point_coordinates(coordinate_file)
    
    metadata_filters: List[Filter] = [
        Filter('cloudy_pixel_percentage', '<=', 25),
        Filter('processing_level', '==', 'Level-2A')]
    
    for i, feature in enumerate(features):
        try:
            mapper_configs = MapperConfigs(
                collection=COLLECTION,
                time_start=start_date,
                time_end=end_date,
                feature=feature,
                metadata_filters=metadata_filters)

            mapper_configs.to_yaml(metadata_dir.joinpath('sample_mapper_call.yaml'))

            mapper = Mapper(mapper_configs)
            mapper.query_scenes()
            mapper.metadata

            # get the least cloudy scene
            mapper.metadata = mapper.metadata[
                mapper.metadata.cloudy_pixel_percentage ==
                mapper.metadata.cloudy_pixel_percentage.min()].copy()
            
            
            # load the least cloudy scene available from STAC
            scene_kwargs = {
                'scene_constructor': Sentinel2.from_safe,
                'scene_constructor_kwargs': {
                    'band_selection': ["B02", "B03", "B04", "B05", "B8A"]}}

            mapper.load_scenes(scene_kwargs=scene_kwargs)
            # the data loaded into `mapper.data` as a EOdal SceneCollection
            # it should now contain only a single scene
            scene = mapper.data[mapper.data.timestamps[0]]

            # save the scene to disk
            # we distinguish the 10 and 20m bands by the suffixes _10m and _20m
            scene_10m = RasterCollection()
            for band in ['blue', 'green', 'red']:
                scene_10m.add_band(scene[band])
            scene_10m.to_rasterio(data_dir.joinpath(f'{i:04d}_scene_10m.tif'))

            scene_20m = RasterCollection()
            for band in ['red_edge_1', 'nir_2']:
                scene_20m.add_band(scene[band])
            scene_20m.to_rasterio(data_dir.joinpath(f'{i:04d}_scene_20m.tif'))

            # save metadata xml to disk
            href_xml = mapper.metadata.assets.iloc[0]['granule-metadata']['href']
            response = urllib.request.urlopen(
                planetary_computer.sign_url(href_xml)).read()
            fpath_xml = metadata_dir.joinpath(f'{i:04d}_' + href_xml.split('/')[-1])
            with open(fpath_xml, 'wb') as dst:
                dst.write(response)

            # plot scene
            fig = scene_10m.plot_multiple_bands(band_selection=['blue', 'green', 'red'])
            fig.savefig(plot_dir.joinpath(f'{i:04d}_scene_10m.png'), dpi=300)
            
            coordinate_x, coordinate_y = point_coordinates[i]
            date = mapper.data.timestamps[0]
            write_date_to_csv(metadata_dir, i, date, coordinate_x, coordinate_y)
        except:
            continue
