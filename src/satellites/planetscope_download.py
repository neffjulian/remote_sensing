"""
All code related to downloading PlanetScope data. This includes placing orders,
downloading orders, and copying the data to the filtered directory.

@date: 2023-08-30
@author: Julian Neff, ETH Zurich

Copyright (C) 2023 Julian Neff

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

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
    """
    Looks at the dates when the corresponding Sentinel-2 images were taken and returns a list of the indices of the images that were taken in the same month and year.

    Parameters:
    - year (str): The year for which the Sentinel-2 images were taken.
    - month (str): The month for which the Sentinel-2 images were taken.

    Returns:
    - dates (list): A list of the indices of the images that were taken in the same month and year.
    """

    # Dir to sentinel metadata files
    sentinel_metadata = DATA_DIR.parent.joinpath("sentinel", f"{year}", f"{MONTHS[month]}_{month}", "metadata")

    # Create empty list to store dates
    dates = {}

    # Iterate through all files in the sentinel metadata dir
    for file in sentinel_metadata.iterdir():
        if file.name.endswith(".xml"):

            # Get the sensing times from the xml file
            root = ET.parse(file).getroot()
            sensing_times = root.findall(".//SENSING_TIME")
            sensing_times_datetime = [datetime.strptime(sensing_time.text, '%Y-%m-%dT%H:%M:%S.%fZ') for sensing_time in sensing_times]

            # Add the index and date to the list
            dates[int(file.name[:4])] = sensing_times_datetime[0]

    # Return the list of dates
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

    # Create empty list to store filtered coordinates
    filtered_coordinates = []

    # Iterate through all features
    for index, feature in enumerate(features["features"]):
        if index in sentinel_dates:

            # Get the processing tools and GeoDataFrame object
            processing_tool = [{"clip": {"aoi": feature['geometry']}}]

            # Create a GeoDataFrame object from the feature
            gdf = gpd.GeoDataFrame.from_features([feature])
            gdf = gdf.set_crs(epsg=4326)

            # Add the index, date, processing tools, and GeoDataFrame object to the list
            filtered_coordinate = (index, sentinel_dates[index], processing_tool, gdf)
            filtered_coordinates.append(filtered_coordinate)

    # Return the list of filtered coordinates
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

    # Create directories for the data and metadata
    _, _, metadata_dir = create_dirs(DATA_DIR, year, month)

    # Get the dates for the Sentinel-2 images
    sentinel_dates = get_sentinel_dates(year, month)

    # Read the coordinates from the file and convert them to squares
    coordinates = convert_to_squares(coordinate_file)

    # Filter the coordinates based on the Sentinel-2 dates
    filtered_coordinates = filter_coordinates(sentinel_dates, coordinates)

    # Create empty list to store orders
    orders = []

    # Iterate through all filtered coordinates
    print("Start placing orders:")
    for index, date, proc_tool, gdf in filtered_coordinates:

        # Try to place the order, adding 12 hours to the time window each time if it fails
        added_hours = 0

        print("Trying to place order ", index)
        while(added_hours <= 48):
            try:
                # Get the start and end dates
                start_date, end_date = get_start_end_dates(date, added_hours)

                # Query the Planet API
                client = PlanetAPIClient.query_planet_api(
                    start_date = start_date,
                    end_date = end_date,
                    bounding_box = gdf,
                    cloud_cover_threshold = 25
                )

                # Collect the order name and url
                order_name = get_order_name(month, year, index)

                # Place the order, this fails if no images are found
                order_url = client.place_order(
                    order_name=order_name,
                    processing_tools=proc_tool
                )

                # Add the order to the list
                orders.append((index, order_name, order_url, added_hours))
                print(f"Placed order: {order_name}")
                break
            except:
                added_hours += 12                

    # Write the orders to a csv file
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

    # Create directories for the data and metadata
    current_dir = DATA_DIR.joinpath(f"{year}", f"{MONTHS[month]}_{month}") # e.g. 2022/04_apr
    orders_location = current_dir.joinpath("metadata", "orders.csv")
    download_dir = current_dir.joinpath("data")

    # Check if the orders file exists
    if not orders_location.exists():
        raise Exception(f"No matching orders found for {orders_location}")
    
    # Read the orders from the csv file
    client = PlanetAPIClient()
    client.authenticate(url=get_settings().ORDERS_URL)
    orders = pd.read_csv(orders_location)
    
    # Iterate through all orders
    for _, row in orders.iterrows():

        # Get the order name and url
        order_name = row["order_name"]
        order_url = row["order_url"]
        index = row["index"]

        # Create a directory for the order
        curr_download_dir = download_dir.joinpath(f"{index:04d}")
        curr_download_dir.mkdir(exist_ok=True)

        # Try to download the order, skipping if it fails
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

    # Create directories for the data and metadata
    source_dir = DATA_DIR.joinpath("raw", "planetscope", year, f"{MONTHS[month]}_{month}", "data")
    target_dir = DATA_DIR.joinpath("filtered", "planetscope", year, f"{MONTHS[month]}_{month}")

    # Check if the target directory exists
    if not target_dir.exists():
        data_dir = target_dir.joinpath("data")
        metadata_dir = target_dir.joinpath("metadata")
        data_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        # Iterate through all subdirectories in the source directory
        for root, dirs, _ in os.walk(str(source_dir)):
            curr_index = Path(root).name
            if not curr_index.isdigit():
                continue
            
            # Select file with the largest size as it likely contains the most information.
            sizes = []
            for dir in dirs:
                curr_dir = Path(root).joinpath(dir)
                curr_files = [os.path.getsize(filename) for filename in curr_dir.iterdir() if "Analytic" in filename.name and filename.name.endswith(".tif")]
                if len(curr_files) > 1:
                    raise Exception("Error in dir ", curr_dir, "too many options to choose from")
                
                sizes.append(curr_files[0])
            max_size = max(sizes)
            max_index = sizes.index(max_size)

            # Check if the file is large enough, usually files smaller than the threshold are not usable.
            if max_size < 4000000:
                print(f"Index {curr_index} is too small ({max_size}).")
                continue
            
            # Copy the image (.tif) and metadata (.xml) file to the target directory
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
    """
    Downloads PlanetScope data from in-situ locations. If place_order is True, then
    the function will place orders for the data. Otherwise, it will download the
    data from the orders that have already been placed.

    Args:
        place_order (bool): Whether to place orders for the data or download the data from existing orders.
    """

    # Create directories for the data and metadata
    coordinates = DATA_DIR.parent.parent.joinpath("coordinates", "field_parcels.geojson")
    data_dir = DATA_DIR.parent.joinpath("in_situ", "planetscope")

    # Check download step
    if place_order is True:
        create_dirs(data_dir, "2022", "mar")

        with open(coordinates, 'r') as source_file:
            squares = json.load(source_file)

        # Create empty list to store orders
        orders = []

        # Iterate through all filtered coordinates
        print("Start placing orders for in situ data.")
        for i, feature in enumerate(squares["features"]):

            # Get the processing tools and GeoDataFrame object
            index = f"{i:04d}"
            date = datetime.strptime(feature["properties"]["date"], "%Y-%m-%d %H:%M:%S")
            lai = feature["properties"]["lai"]

            processing_tool = [{"clip": {"aoi": feature['geometry']}}]
            gdf = gpd.GeoDataFrame.from_features([feature])
            gdf = gdf.set_crs(epsg=4326)
            
            # Try to place the order, adding 12 hours to the time window each time if it fails
            added_hours = 6
            while(added_hours < 48):
                try:
                    # Get the start and end dates
                    start_date, end_date = get_start_end_dates(date, added_hours)

                    # Query the Planet API
                    client = PlanetAPIClient.query_planet_api(
                        start_date = start_date,
                        end_date = end_date,
                        bounding_box = gdf,
                        cloud_cover_threshold = 10
                    )

                    # Collect the order name and url, this fails if no images are found
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
        
        # Write the orders to a csv file
        orders_csv = pd.DataFrame(orders, columns=["index", "order_name", "order_url", "added_hours", "lai"])
        orders_location = data_dir.joinpath("orders.csv")
        orders_csv.to_csv(orders_location, index=False)
    
    else:
        # Read the orders from the csv file
        orders_location = data_dir.joinpath("orders.csv")
        download_dir = data_dir.joinpath("2022", "03_mar", "data")
        client = PlanetAPIClient()
        client.authenticate(url=get_settings().ORDERS_URL)
        orders = pd.read_csv(orders_location)

        # Iterate through all orders
        for _, row in orders.iterrows():
            # Get the order name and url
            index = row["index"]
            order_name = row["order_name"]
            order_url = row["order_url"]

            # Create a directory for the order
            curr_download_dir = download_dir.joinpath(f"{index:04d}")
            curr_download_dir.mkdir(exist_ok=True)

            try:
                # Download the order
                status = client.check_order_status(
                    order_url=order_url
                )

                # Check if the order failed
                if status == "Failed":
                    print(f"Order '{order_name}' failed to be ordered.")
                    continue
                
                # Download the order
                client.download_order(
                    download_dir=curr_download_dir,
                    order_url=order_url
                )
            except:
                print(f"Order '{order_name}' failed while downloading.")

        # Copy the data to the filtered directory
        source_dir = download_dir
        target_dir = data_dir.parent.parent.parent.joinpath("filtered", "in_situ", "planetscope_in_situ")

        # Check if the target directory exists
        data_dir = target_dir.joinpath("data")
        metadata_dir = target_dir.joinpath("metadata")
        
        # Iterate through all subdirectories in the source directory
        for root, dirs, _ in os.walk(str(source_dir)):
            curr_index = Path(root).name
            if not curr_index.isdigit():
                continue
            
            # Select file with the largest size as it likely contains the most information.
            sizes = []
            for dir in dirs:
                curr_dir = Path(root).joinpath(dir)
                curr_files = [os.path.getsize(filename) for filename in curr_dir.iterdir() if "Analytic" in filename.name and filename.name.endswith(".tif")]
                if len(curr_files) > 1:
                    raise Exception("Error in dir ", curr_dir, "too many options to choose from")
                
                sizes.append(curr_files[0])
            max_size = max(sizes)
            max_index = sizes.index(max_size)

            # Check if the file is large enough, usually files smaller than the threshold are not usable.
            if max_size < 4000000:
                print(f"Index {curr_index} is too small ({max_size}).")
                continue
            
            # Copy the image (.tif) and metadata (.xml) file to the target directory
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
    """
    Main function for downloading PlanetScope data. If place_order is True, then
    the function will place orders for the data. Otherwise, it will download the
    data from the orders that have already been placed.

    Args:
        year (str): The year of the data to download.
        month (str): The month of the data to download.
        place_order (bool): Whether to place orders for the data or download the data from existing orders.
        download_order (bool): Whether to download the data from existing orders.
        in_situ (bool): Whether to download in-situ data.
    """


    # Check if in-situ data is to be downloaded
    if in_situ is True:
        download_in_situ(place_order=place_order)
        return

    # Check if the year and month are valid, there are no orders available before 2017 and after 2022
    if not (2017 <= int(year) <= 2022):
        raise ValueError(f"Year invalid ('{year}'). Use a value between '2017'  and '2022'.")
    
    # Check if the month exists
    if month not in MONTHS:
        raise ValueError(f"Month invalid ('{month}'). Use one out of {list(MONTHS)}.")

    # Check if the order is to be placed or downloaded
    if place_order is True:
        print(f"Placing order for {year} {month}...")
        place_planetscope_orders('points_ch.geojson', year, month)

    # Check if the order is to be downloaded
    elif download_order is True:
        print(f"Download order for {year} {month}...")
        download_planetscope_orders(year, month)
        copy_planetscope_data(year, month)

    # Raise an error if neither is selected
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