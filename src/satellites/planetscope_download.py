import os
import json
import argparse
from pathlib import Path
from datetime import datetime, date, timedelta
import xml.etree.ElementTree as ET
import shutil

import pandas as pd
import geopandas as gpd
from dotenv import load_dotenv
from eodal.downloader.planet_scope import PlanetAPIClient
from eodal.config import get_settings

from utils import (
    convert_to_squares,
    create_dirs,
    MONTHS
)

DOTENV_PATH = Path(__file__).parent.parent.parent.joinpath(".env")
DATA_DIR = Path(__file__).parent.parent.parent.joinpath("data", "raw", "planetscope")
load_dotenv(DOTENV_PATH)

settings = get_settings()
settings.PLANET_API_KEY = os.getenv('PLANET_API_KEY')
settings.USE_STAC = True

def get_sentinel_dates(year: str, month: str) -> list:
    sentinel_metadata = DATA_DIR.parent.joinpath("sentinel", f"{year}", f"{MONTHS[month]}_{month}", "metadata")

    dates = {}
    for file in sentinel_metadata.iterdir():
        if file.name.endswith(".xml"):
            root = ET.parse(file).getroot()
            sensing_times = root.findall(".//SENSING_TIME")
            sensing_times_datetime = [datetime.strptime(sensing_time.text, '%Y-%m-%dT%H:%M:%S.%fZ') for sensing_time in sensing_times]

            dates[int(file.name[:4])] = sensing_times_datetime[0]

    return dates

def get_start_end_dates(date: datetime, added_hours: int) -> tuple:
    """
    Given a datetime object and a number of hours to add, returns the start and end dates.

    Parameters:
    - date (datetime): The datetime object representing the date for which the start and end dates will be computed.
    - added_hours (int): The number of hours to add to the start and end dates.

    Returns:
    - tuple: A tuple containing the start and end dates.
    """
    start_date = date - timedelta(hours=added_hours)
    end_date = date + timedelta(hours=added_hours)
    return start_date, end_date


def filter_coordinates(sentinel_dates: dict, features: list) -> list:
    """
    Filters a list of geojson features based on whether their index is present in a given dataframe of Sentinel-2 dates.
    
    Args:
    - sentinel_dates (pd.DataFrame): A dataframe containing a list of Sentinel-2 dates with their corresponding index values.
    - features (list): A list of geojson features to be filtered.
    
    Returns:
    - filtered_coordinates (list): A list of tuples containing the filtered features along with their index, date, processing tools, and GeoDataFrame object.
    """
    filtered_coordinates = []
    for index, feature in enumerate(features["features"]):
        if index in sentinel_dates:
            processing_tool = [{"clip": {"aoi": feature['geometry']}}]
            gdf = gpd.GeoDataFrame.from_features([feature])
            gdf = gdf.set_crs(epsg=4326)
            filtered_coordinate = (index, sentinel_dates[index], processing_tool, gdf)
            filtered_coordinates.append(filtered_coordinate)
    return filtered_coordinates

def get_order_name(month: str, year: str, index: str) -> str:
    """
    Generates a string representing the name of an order based on the month, year, and index provided, along with the current date.

    Args:
    - month (str): A string representing the month of the order.
    - year (str): A string representing the year of the order.
    - index (str): A string representing the index of the order.

    Returns:
    - order_name (str): A string representing the name of the order, with the format 'current_date_year_month_index'.
    """
    return f'{date.today()}_{year}_{month}_{int(index):04d}'

def place_planetscope_orders(coordinate_file: str, year: str, month: str) -> None:
    """
    Places orders for PlanetScope imagery for a given month and year based on the coordinates provided in a file.

    Args:
    - coordinate_file (str): A string representing the path to the file containing the coordinates to be used for the orders.
    - year (str): A string representing the year for which the orders are to be placed.
    - month (str): A string representing the month for which the orders are to be placed.

    Returns:
    - None
    """
    _, _, metadata_dir = create_dirs(DATA_DIR, year, month)
    sentinel_dates = get_sentinel_dates(year, month)

    coordinates = convert_to_squares(coordinate_file)
    filtered_coordinates = filter_coordinates(sentinel_dates, coordinates)

    orders = []
    print("Start placing orders:")
    for index, date, proc_tool, gdf in filtered_coordinates:
        added_hours = 0
        print("Trying to place order ", index)
        while(added_hours <= 48):
            try:
                start_date, end_date = get_start_end_dates(date, added_hours)
                client = PlanetAPIClient.query_planet_api(
                    start_date = start_date,
                    end_date = end_date,
                    bounding_box = gdf,
                    cloud_cover_threshold = 25
                )
                order_name = get_order_name(month, year, index)
                order_url = client.place_order(
                    order_name=order_name,
                    processing_tools=proc_tool
                )
                orders.append((index, order_name, order_url, added_hours))
                print(f"Placed order: {order_name}")
                break
            except:
                added_hours += 12                

    orders_csv = pd.DataFrame(orders, columns=["index", "order_name", "order_url", "added_hours"])
    orders_location = metadata_dir.joinpath("orders.csv")
    orders_csv.to_csv(orders_location, index=False)

def download_planetscope_orders(year: str, month: str) -> None:
    """
    Downloads PlanetScope imagery orders for a given month and year.

    Args:
    - year (str): A string representing the year for which the orders are to be downloaded.
    - month (str): A string representing the month for which the orders are to be downloaded.

    Returns:
    - None
    """
    current_dir = DATA_DIR.joinpath(f"{year}", f"{MONTHS[month]}_{month}") # e.g. 2022/04_apr
    orders_location = current_dir.joinpath("metadata", "orders.csv")
    download_dir = current_dir.joinpath("data")

    if not orders_location.exists():
        raise Exception(f"No matching orders found for {orders_location}")
    
    client = PlanetAPIClient()
    client.authenticate(url=get_settings().ORDERS_URL)
    orders = pd.read_csv(orders_location)
    
    for _, row in orders.iterrows():
        order_name = row["order_name"]
        order_url = row["order_url"]
        index = row["index"]

        curr_download_dir = download_dir.joinpath(f"{index:04d}")
        curr_download_dir.mkdir(exist_ok=True)

        try:
            status = client.check_order_status(
                order_url=order_url
            )

            if status == "Failed":
                print(f"Order '{order_name}' failed to be ordered.")
                continue
        
            client.download_order(
                download_dir=curr_download_dir,
                order_url=order_url
            )
        except:
            print(f"Order '{order_name}' failed while downloading.")

def copy_planetscope_data(year: str, month: str) -> None:
    """
    Copy PlanetScope data from the raw directory to the filtered directory,
    filtering out non-Analytic files and copying each image and its metadata file
    to the data and metadata subdirectories, respectively.

    Args:
        year (str): The year of the data to copy.
        month (str): The month of the data to copy.

    Returns:
        None.
    """

    source_dir = DATA_DIR.joinpath("raw", "planetscope", year, f"{MONTHS[month]}_{month}", "data")
    target_dir = DATA_DIR.joinpath("filtered", "planetscope", year, f"{MONTHS[month]}_{month}")

    if not target_dir.exists():
        data_dir = target_dir.joinpath("data")
        metadata_dir = target_dir.joinpath("metadata")
        data_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        for root, dirs, _ in os.walk(str(source_dir)):
            curr_index = Path(root).name
            if not curr_index.isdigit():
                continue

            sizes = []
            for dir in dirs:
                curr_dir = Path(root).joinpath(dir)
                curr_files = [os.path.getsize(filename) for filename in curr_dir.iterdir() if "Analytic" in filename.name and filename.name.endswith(".tif")]
                if len(curr_files) > 1:
                    raise Exception("Error in dir ", curr_dir, "too many options to choose from")
                
                sizes.append(curr_files[0])
            max_size = max(sizes)
            max_index = sizes.index(max_size)

            if max_size < 4000000:
                print(f"Index {curr_index} is too small ({max_size}).")
                continue

            curr_dir = Path(root).joinpath(dirs[max_index])
            for file in curr_dir.iterdir():
                if not "Analytic" in file.name:
                    continue

                if file.name.endswith(".tif"):
                    shutil.copy(src=file,
                        dst=data_dir.joinpath(f"{curr_index}.tif"),
                        follow_symlinks=False
                    )
                elif file.name.endswith(".xml"):
                    shutil.copy(
                        src=file,
                        dst=metadata_dir.joinpath(f"{curr_index}.xml"),
                        follow_symlinks=False
                    )
                else:
                    raise Exception("Error with file: ", file, " should be either a .tif file or .xml")
                
def download_in_situ(place_order: bool) -> None:
    coordinates = DATA_DIR.parent.parent.joinpath("coordinates", "field_parcels.geojson")
    data_dir = DATA_DIR.parent.joinpath("in_situ", "planetscope")

    if place_order is True:
        create_dirs(data_dir, "2022", "mar")

        with open(coordinates, 'r') as source_file:
            squares = json.load(source_file)

        orders = []
        print("Start placing orders for in situ data.")
        for i, feature in enumerate(squares["features"]):
            index = f"{i:04d}"
            date = datetime.strptime(feature["properties"]["date"], "%Y-%m-%d %H:%M:%S")
            lai = feature["properties"]["lai"]

            processing_tool = [{"clip": {"aoi": feature['geometry']}}]
            gdf = gpd.GeoDataFrame.from_features([feature])
            gdf = gdf.set_crs(epsg=4326)
            
            added_hours = 6
            while(added_hours < 48):
                try:
                    start_date, end_date = get_start_end_dates(date, added_hours)
                    client = PlanetAPIClient.query_planet_api(
                        start_date = start_date,
                        end_date = end_date,
                        bounding_box = gdf,
                        cloud_cover_threshold = 10
                    )
                    order_name = f"in_situ_{index}"
                    order_url = client.place_order(
                        order_name=order_name,
                        processing_tools=processing_tool
                    )
                    orders.append((index, order_name, order_url, added_hours, lai))
                    print(f"Placed order: {order_name} - Addded hours: ", added_hours)
                    break
                except:
                    added_hours += 12
        
        orders_csv = pd.DataFrame(orders, columns=["index", "order_name", "order_url", "added_hours", "lai"])
        orders_location = data_dir.joinpath("orders.csv")
        orders_csv.to_csv(orders_location, index=False)
    
    else:
        orders_location = data_dir.joinpath("orders.csv")
        download_dir = data_dir.joinpath("2022", "03_mar", "data")
        client = PlanetAPIClient()
        client.authenticate(url=get_settings().ORDERS_URL)
        orders = pd.read_csv(orders_location)
        
        for _, row in orders.iterrows():
            index = row["index"]
            order_name = row["order_name"]
            order_url = row["order_url"]

            curr_download_dir = download_dir.joinpath(f"{index:04d}")
            curr_download_dir.mkdir(exist_ok=True)

            try:
                status = client.check_order_status(
                    order_url=order_url
                )

                if status == "Failed":
                    print(f"Order '{order_name}' failed to be ordered.")
                    continue
            
                client.download_order(
                    download_dir=curr_download_dir,
                    order_url=order_url
                )
            except:
                print(f"Order '{order_name}' failed while downloading.")
        source_dir = download_dir
        target_dir = data_dir.parent.parent.parent.joinpath("filtered", "in_situ", "planetscope_in_situ")

        data_dir = target_dir.joinpath("data")
        metadata_dir = target_dir.joinpath("metadata")
        
        for root, dirs, _ in os.walk(str(source_dir)):
            curr_index = Path(root).name
            if not curr_index.isdigit():
                continue

            sizes = []
            for dir in dirs:
                curr_dir = Path(root).joinpath(dir)
                curr_files = [os.path.getsize(filename) for filename in curr_dir.iterdir() if "Analytic" in filename.name and filename.name.endswith(".tif")]
                if len(curr_files) > 1:
                    raise Exception("Error in dir ", curr_dir, "too many options to choose from")
                
                sizes.append(curr_files[0])
            max_size = max(sizes)
            max_index = sizes.index(max_size)

            if max_size < 4000000:
                print(f"Index {curr_index} is too small ({max_size}).")
                continue

            curr_dir = Path(root).joinpath(dirs[max_index])
            for file in curr_dir.iterdir():
                if not "Analytic" in file.name:
                    continue

                if file.name.endswith(".tif"):
                    shutil.copy(src=file,
                        dst=data_dir.joinpath(f"{curr_index}.tif"),
                        follow_symlinks=False
                    )
                elif file.name.endswith(".xml"):
                    shutil.copy(
                        src=file,
                        dst=metadata_dir.joinpath(f"{curr_index}.xml"),
                        follow_symlinks=False
                    )
                else:
                    raise Exception("Error with file: ", file, " should be either a .tif file or .xml")

def main(year: str = "mar", month: str = "2022", place_order: bool = False, download_order: bool = False, in_situ: bool = False) -> None:
    if in_situ is True:
        download_in_situ(place_order=place_order)
        return

    if not (2017 <= int(year) <= 2022):
        raise ValueError(f"Year invalid ('{year}'). Use a value between '2017'  and '2022'.")
    
    if month not in MONTHS:
        raise ValueError(f"Month invalid ('{month}'). Use one out of {list(MONTHS)}.")

    if place_order is True:
        print(f"Placing order for {year} {month}...")
        place_planetscope_orders('points_ch.geojson', year, month)
    elif download_order is True:
        print(f"Download order for {year} {month}...")
        download_planetscope_orders(year, month)
        copy_planetscope_data(year, month)
    else:
        raise ValueError("Either select 'place_order' or 'download_orders'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=str)
    parser.add_argument("--month", type=str)
    parser.add_argument("--place_order", type=bool)
    parser.add_argument("--download_order", type=bool)
    parser.add_argument("--in_situ", type=bool)

    args = parser.parse_args()
    main(**vars(args))