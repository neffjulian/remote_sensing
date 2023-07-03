from pathlib import Path
from typing import List
import argparse
import urllib
from datetime import datetime, timedelta

import json
import pandas as pd
import geopandas as gpd
import planetary_computer
from eodal.config import get_settings
from eodal.core.raster import RasterCollection
from eodal.core.sensors.sentinel2 import Sentinel2
from eodal.mapper.feature import Feature
from eodal.mapper.filter import Filter
from eodal.mapper.mapper import Mapper, MapperConfigs

from utils import (
    convert_to_squares,
    create_dirs,
    MONTHS
)

Settings = get_settings()
Settings.USE_STAC = True

# Define the directories for storing data, metadata, and plots
DATA_DIR = Path(__file__).parent.parent.parent.joinpath("data", "raw", "sentinel")
COORDINATE_DIR = Path(__file__).parent.parent.parent.joinpath("data", "coordinates")
COLLECTION = "sentinel2-msi"

DAY_IN_MONTH = {"jan": "31", "feb": "28", "mar": "31", "apr": "30", 
          "may": "31", "jun": "30", "jul": "31", "aug": "31", 
          "sep": "30", "oct": "31", "nov": "30", "dec": "31"}

def get_point_names(coordinate_file: str) -> list:
    point_location = COORDINATE_DIR.joinpath(coordinate_file)
    point_file = point_location.open()
    point_geojson = json.load(point_file)
    point_names = []

    for point in point_geojson["features"]:
        coordinates = point["properties"]["name"]
        point_names.append(coordinates)
        
    return point_names


def get_point_coordinates(coordinate_file: str) -> list:
    """Get the names of the points from a coordinate file.

    Args:
        coordinate_file (str): The path to the coordinate file.

    Returns:
        List[str]: A list of point names.
    """
    point_location = COORDINATE_DIR.joinpath(coordinate_file)
    point_file = point_location.open()
    point_geojson = json.load(point_file)
    point_coordinates = []

    for point in point_geojson["features"]:
        coordinates = point["geometry"]["coordinates"]
        point_coordinates.append(coordinates)

    return point_coordinates

def get_dates(year: str, month: str) -> tuple:
    """Get the coordinates of the points from a coordinate file.

    Args:
        coordinate_file (str): The path to the coordinate file.

    Returns:
        List: A list of point coordinates.
    """
    start_date = datetime(int(year), int(MONTHS[month]), 1)
    end_date = datetime(int(year), int(MONTHS[month]), int(DAY_IN_MONTH[month]))
    return start_date, end_date

def write_date_to_csv(metadata_dir: Path, index: int, date: str, coordinate_x: float, coordinate_y) -> None:
    """Get the start and end dates of a given month and year.

    Args:
        year (str): The year.
        month (str): The month.

    Returns:
        tuple: A tuple containing the start date and end date.
    """
    file_location = metadata_dir.joinpath("dates.csv")
    
    try:
        df = pd.read_csv(file_location)
    except:
        df = pd.DataFrame(columns=['index', 'date', 'x', 'y'])

    new_row = pd.DataFrame({
        'index': [index],
        'date': [date],
        'x': [coordinate_x],
        'y': [coordinate_y]
    })

    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(file_location, header=True, index=False)

def get_features(coordinates_location: Path) -> list:
    """Convert square features to polygons.

    Args:
        coordinates_location (Path): The path to the coordinates.

    Returns:
        List: A list of features.
    """
    coordinates = convert_to_squares(coordinates_location)

    features = []
    for feature in coordinates['features']:
        gdf = {"type": "FeatureCollection", "features": [feature]}
        feat = Feature("geometry", gpd.GeoDataFrame.from_features(gdf)['geometry'].iloc[0], 4326, {})
        features.append(feat)
    return features

def download_sentinel_data(coordinate_file: str, year: str, month: str) -> None:
    """Download Sentinel-2 data for a given coordinate file, year, and month.

    Args:
        coordinate_file (str): The path to the coordinate file.
        year (str): The year. e.g. "2022"
        month (str): The month. e.g. "aug"
    """
    data_dir, plot_dir, metadata_dir = create_dirs(DATA_DIR, year, month)
    start_date, end_date = get_dates(year, month)
    features = get_features(coordinate_file)
    point_coordinates = get_point_coordinates(coordinate_file)
    
    metadata_filters: List[Filter] = [
        Filter('cloudy_pixel_percentage', '<=', 25),
        Filter('processing_level', '==', 'Level-2A')]
    
    errors = []
    
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
                    'band_selection': ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]}}

            mapper.load_scenes(scene_kwargs=scene_kwargs)
            # the data loaded into `mapper.data` as a EOdal SceneCollection
            # it should now contain only a single scene
            scene = mapper.data[mapper.data.timestamps[0]]

            # save the scene to disk
            # we distinguish the 10 and 20m bands by the suffixes _10m and _20m
            scene_10m = RasterCollection()
            for band in ['blue', 'green', 'red', 'nir_1']:
                scene_10m.add_band(scene[band])
            scene_10m.to_rasterio(data_dir.joinpath(f'{i:04d}_scene_10m.tif'))

            scene_20m = RasterCollection()
            for band in ['red_edge_1', 'red_edge_2', 'red_edge_3', 'nir_2', 'swir_1', 'swir_2']:
                scene_20m.add_band(scene[band])
            scene_20m.to_rasterio(data_dir.joinpath(f'{i:04d}_scene_20m.tif'))

            scene_60m = RasterCollection()
            for band in ['ultra_blue', 'nir_3']:
                scene_60m.add_band(scene[band])
            scene_60m.to_rasterio(data_dir.joinpath(f'{i:04d}_scene_60m.tif'))

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
            errors.append(i)

    print(f"In total {len(errors)} errors occured. Namely {errors}")

def download_eschikon() -> None:
    """Download Eschikon data with available in-situ measurements

    Args:
        coordinate_file (str): The path to the coordinate file.
        year (str): The year. E.g. "2022"
        month (str): The month. E.g. "aug"
    """
    field_dir = DATA_DIR.parent.parent.joinpath("fields", "sentinel")
    data_dir = field_dir.joinpath("data")
    metadata_dir = field_dir.joinpath("metadata")
    plot_dir = field_dir.joinpath("plot")

    data_dir.mkdir(exist_ok=True, parents=True)
    metadata_dir.mkdir(exist_ok=True, parents=True)
    plot_dir.mkdir(exist_ok=True, parents=True)

    coordinate_file = DATA_DIR.parent.parent.joinpath("coordinates", "field_parcels.geojson")

    # TODO: Change dates to when in-situ measurements were taken
    dates = [
        (datetime(2022, 1, 1), datetime(2022, 12, 31)), # broatefaeld
        (datetime(2022, 1, 1), datetime(2022, 12, 31)), # bramenwies
        (datetime(2022, 1, 1), datetime(2022, 12, 31)), # fluegenrain
        (datetime(2022, 1, 1), datetime(2022, 12, 31)), # hohrueti
        (datetime(2022, 1, 1), datetime(2022, 12, 31)), # altkloster
        (datetime(2022, 1, 1), datetime(2022, 12, 31)), # ruetteli
        (datetime(2022, 1, 1), datetime(2022, 12, 31)), # witzwil
        (datetime(2022, 1, 1), datetime(2022, 12, 31)), # strickhof
        (datetime(2022, 1, 1), datetime(2022, 12, 31)), # eschikon
    ]

    field_names = get_point_names("field_parcels.geojson")
    features = get_features(coordinate_file)
    point_coordinates = get_point_coordinates(coordinate_file)

    metadata_filters: List[Filter] = [
        Filter('cloudy_pixel_percentage', '<=', 25),
        Filter('processing_level', '==', 'Level-2A')]
    
    errors = []
    
    for i, feature in enumerate(features):
        start_date, end_date = dates[i]
        field_name = field_names[i]
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
                    'band_selection': ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]}}

            mapper.load_scenes(scene_kwargs=scene_kwargs)
            # the data loaded into `mapper.data` as a EOdal SceneCollection
            # it should now contain only a single scene
            scene = mapper.data[mapper.data.timestamps[0]]

            # save the scene to disk
            # we distinguish the 10 and 20m bands by the suffixes _10m and _20m
            scene_10m = RasterCollection()
            for band in ['blue', 'green', 'red', 'nir_1']:
                scene_10m.add_band(scene[band])
            scene_10m.to_rasterio(data_dir.joinpath(f'{field_name}_scene_10m.tif'))

            scene_20m = RasterCollection()
            for band in ['red_edge_1', 'red_edge_2', 'red_edge_3', 'nir_2', 'swir_1', 'swir_2']:
                scene_20m.add_band(scene[band])
            scene_20m.to_rasterio(data_dir.joinpath(f'{field_name}_scene_20m.tif'))

            scene_60m = RasterCollection()
            for band in ['ultra_blue', 'nir_3']:
                scene_60m.add_band(scene[band])
            scene_60m.to_rasterio(data_dir.joinpath(f'{field_name}_scene_60m.tif'))

            # save metadata xml to disk
            href_xml = mapper.metadata.assets.iloc[0]['granule-metadata']['href']
            response = urllib.request.urlopen(
                planetary_computer.sign_url(href_xml)).read()
            fpath_xml = metadata_dir.joinpath(f'{field_name}_' + href_xml.split('/')[-1])
            with open(fpath_xml, 'wb') as dst:
                dst.write(response)

            # plot scene
            fig = scene_10m.plot_multiple_bands(band_selection=['blue', 'green', 'red'])
            fig.savefig(plot_dir.joinpath(f'{field_name}_scene_10m.png'), dpi=300)
            
            coordinate_x, coordinate_y = point_coordinates[i]
            date = mapper.data.timestamps[0]
            write_date_to_csv(metadata_dir, i, date, coordinate_x, coordinate_y)
        except:
            errors.append(i)

    print(f"In total {len(errors)} occured for Eschikon downloads. Namely {errors}")

def download_in_situ() -> None:
    folder = DATA_DIR.parent.joinpath("in_situ", "sentinel")
    data_dir, metadata_dir, plot_dir = create_dirs(folder, "2022", "mar")
    
    coordinates = DATA_DIR.parent.parent.joinpath("coordinates", "field_parcels.geojson")
    with open(coordinates, 'r') as source_file:
        squares = json.load(source_file)

    orders = []

    metadata_filters: List[Filter] = [
        Filter('cloudy_pixel_percentage', '<=', 25),
        Filter('processing_level', '==', 'Level-2A')
    ]
    
    print("Start placing orders for in situ data.")
    for i, feature in enumerate(squares["features"]):
        added_hours = 6
        index = f"{i:04d}"
        date = datetime.strptime(feature["properties"]["date"], "%Y-%m-%d %H:%M:%S")
        lai = feature["properties"]["lai"]
        gdf = {"type": "FeatureCollection", "features": [feature]}
        feature = Feature("geometry", gpd.GeoDataFrame.from_features(gdf)['geometry'].iloc[0], 4326, {})
        while( added_hours < 48):
            try:
                start_date = date - timedelta(hours=added_hours)
                end_date = date + timedelta(hours=added_hours)

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
                        'band_selection': ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]}}

                mapper.load_scenes(scene_kwargs=scene_kwargs)
                # the data loaded into `mapper.data` as a EOdal SceneCollection
                # it should now contain only a single scene
                scene = mapper.data[mapper.data.timestamps[0]]

                # save the scene to disk
                # we distinguish the 10 and 20m bands by the suffixes _10m and _20m
                scene_10m = RasterCollection()
                for band in ['blue', 'green', 'red', 'nir_1']:
                    scene_10m.add_band(scene[band])
                scene_10m.to_rasterio(data_dir.joinpath(f'{i:04d}_scene_10m.tif'))

                scene_20m = RasterCollection()
                for band in ['red_edge_1', 'red_edge_2', 'red_edge_3', 'nir_2', 'swir_1', 'swir_2']:
                    scene_20m.add_band(scene[band])
                scene_20m.to_rasterio(data_dir.joinpath(f'{i:04d}_scene_20m.tif'))

                scene_60m = RasterCollection()
                for band in ['ultra_blue', 'nir_3']:
                    scene_60m.add_band(scene[band])
                scene_60m.to_rasterio(data_dir.joinpath(f'{i:04d}_scene_60m.tif'))

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
                
                date = mapper.data.timestamps[0]
                write_date_to_csv(metadata_dir, i, date)
            except:
                added_hours += 6

def main(year: str, month: str, in_situ: bool = False) -> None:
    if not (2017 <= int(year) <= 2022):
        raise ValueError(f"Year invalid ('{year}'). Use a value between '2017'  and '2022'.")
    
    if month not in MONTHS:
        raise ValueError(f"Month invalid ('{month}'). Use one out of {list(MONTHS)}.")
    
    if in_situ is True:
        download_in_situ()
        return

    download_sentinel_data('points_ch.geojson', year, month)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", required=True, type=str)
    parser.add_argument("--month", required=True, type=str)
    parser.add_argument("--in_situ", type=bool)

    args = parser.parse_args()
    main(**vars(args))