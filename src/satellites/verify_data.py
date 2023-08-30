"""
This file creates a CSV of all available data. It is used to verify where data is available, 
what coordinates it has, and what the time difference between PlanetScope and Sentinel-2 data is.

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

from pathlib import Path
import json
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent.parent.joinpath("data")
MONTH_MAPS = {"March": "03_mar", "April": "04_apr", "May": "05_may", "June": "06_jun", "July": "07_jul", "August": "08_aug", "September": "09_sep"}

def get_planetscope_date(index: int, month: str) -> SystemError:
    """
    Collects the time of capture of PlanetScope data for a given index and month.

    Args:
        index (int): The index of the point.
        month (str): The month of the point.

    Returns:
        str: The time of capture of the PlanetScope data.
    """

    # Get the path of the data and metadata files
    data = DATA_DIR.joinpath("filtered", "planetscope", "2022", MONTH_MAPS[month], "lai", f"{index:04d}_lai_4bands.tif")
    file = DATA_DIR.joinpath("filtered", "planetscope", "2022", MONTH_MAPS[month], "metadata", f"{index:04d}.xml")

    # Check if the files exist
    if file.exists() and data.exists():

        # Parse the XML file
        tree = ET.parse(file)
        root = tree.getroot()

        # Return the time of capture 
        return root[0][0][4][0][1].text
    else:
        return "-"

def get_sentinel_date(index: int, month: str) -> str:
    """
    Checks the time of capture of Sentinel-2 data for a given index and month.

    Args:
        index (int): The index of the point.
        month (str): The month of the point.

    Returns:
        str: The time of capture of the Sentinel-2 data.
    """

    # Get the path of the data and metadata files
    data = DATA_DIR.joinpath("filtered", "sentinel", "2022", MONTH_MAPS[month], "lai", f"{index:04d}_scene_10m_lai.tif")
    file = DATA_DIR.joinpath("filtered", "sentinel", "2022", MONTH_MAPS[month], "metadata", f"{index:04d}_MTD_TL.xml")

    # Check if the files exist
    if file.exists() and data.exists():

        # Parse the XML file
        tree = ET.parse(file)
        root = tree.getroot()

        # Return the time of capture
        return root[0][4].text
    else:
        return "-"
    
def parse_timestamp(timestamp_str):
    """
    Parse a timestamp string into a datetime object.

    Args:
        timestamp_str (str): The timestamp string to parse.
    """
    try:
        # Attempt to parse the timestamp
        if timestamp_str.endswith('Z'):
            timestamp_str = timestamp_str[:-1] + '+00:00'
        return datetime.fromisoformat(timestamp_str)
    except ValueError:
        # Return None if parsing fails
        return None
    
def get_time_diff(timestamp1: str, timestamp2: str) -> int:
    """
    Get the time difference between two timestamps in hours.

    Args:
        timestamp1 (str): The first timestamp.
        timestamp2 (str): The second timestamp.

    Returns:
        int: The time difference in hours.
    """

    # Parse the timestamps
    dt1 = parse_timestamp(timestamp1)
    dt2 = parse_timestamp(timestamp2)

    # Return None if parsing fails
    if dt1 is None or dt2 is None:
        return "-"
    
    # Calculate the time difference in hours
    difference = dt2 - dt1
    hours, remainder = divmod(difference.seconds, 3600)

    # Return the time difference in hours
    return hours
    
def verify_files() -> None:
    """
    Creates a CSV file with the following columns:
    "month", "index", "latitude", "longitude", "planetscope_capture_date", "sentinel_capture_date", "difference_hours", "area_sq_km"
    """
    # read points from geojson
    points_ch = DATA_DIR.joinpath("coordinates", "points_ch.geojson")
    with open(points_ch) as f:
        points = json.load(f)
    
    # create dataframe
    df = pd.DataFrame(index = range(2072), columns=["month", "index", "latitude", "longitude", "planetscope_capture_date", "sentinel_capture_date", "difference_hours", "area_sq_km"])

    # iterate through months and points
    for i, month in enumerate(list(MONTH_MAPS.keys())):
        for index in range(296):
            # fill dataframe
            df.iloc[i*296 + index]["month"] = month
            df.iloc[i*296 + index]["index"] = index
            df.iloc[i*296 + index]["latitude"] = points["features"][index]["geometry"]["coordinates"][1]
            df.iloc[i*296 + index]["longitude"] = points["features"][index]["geometry"]["coordinates"][0]
            s2_time = get_sentinel_date(index, month)
            ps_time = get_planetscope_date(index, month)
            df.iloc[i*296 + index]["sentinel_capture_date"] = s2_time
            df.iloc[i*296 + index]["planetscope_capture_date"] = ps_time
            df.iloc[i*296 + index]["difference_hours"] = get_time_diff(s2_time, ps_time)

            if s2_time != "-" and ps_time != "-":
                df.iloc[i*296 + index]["area_sq_km"] = 4

    df.to_csv(DATA_DIR.joinpath("filtered", "verify.csv"), index=False)

if __name__ == "__main__":
    verify_files()