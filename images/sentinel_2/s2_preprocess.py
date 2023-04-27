import json
import csv
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

# Define the bands to retrieve
bands = ["B02", "B03", "B04", "B05", "B8A"]
bands_10m = ["B02", "B03", "B04"]
bands_20m = ["B05", "B8A"]

def load_stats(month: str, year: str) -> np.ndarray:
    """
    Load the statistics data for a specified month and year from a CSV file.

    Args:
        month (str): A string representing the month, e.g. 'jan' for January.
        year (str): A string representing the year, e.g. '22' for 2022.

    Returns:
        numpy.ndarray: A NumPy array containing the statistics data.

    """
    stats_location = f"images/sentinel_2/stats/{year}_{month}_stats.csv"
    stats_path = Path(stats_location)

    stats = np.loadtxt(stats_path, delimiter = ',', dtype=object)
    return stats

def save_stats(month: str, year: str, data: np.ndarray) -> None:
    """
    Save statistics data for a specified month and year to a CSV file.

    Args:
        month (str): A string representing the month, e.g. 'jan' for January.
        year (str): A string representing the year, e.g. '22' for 2022.
        data (numpy.ndarray): A NumPy array containing the statistics data to be saved.
# 
    Returns:
        None.

    """
    stats_location = f"images/sentinel_2/stats/{year}_{month}_stats.csv"
    stats_path = Path(stats_location)

    header = "index, date, coordinate, B02, B03, B04, B05, B8A"
    np.savetxt(stats_path, data, header=header, delimiter=',', fmt='%s', encoding="utf-8")

def remove_date_from_stats(month: str, year: str, index: int) -> None:
    """
    Remove the date value from a specified row in a statistics CSV file.

    Args:
        month (str): A string representing the month, e.g. 'jan' for January.
        year (str): A string representing the year, e.g. '22' for 2022.
        index (int): An integer representing the index of the row to update in the CSV file.

    Returns:
        None.

    """
    stats_location = f"images/sentinel_2/stats/{year}_{month}_stats.csv"
    stats_path = Path(stats_location)
    header = "index, date, coordinate, B02, B03, B04, B05, B8A"

    stats = np.loadtxt(stats_path, delimiter = ',', dtype=object)
    stats[index, 1] = None
    np.savetxt(stats_path, stats, header=header, delimiter=',', fmt='%s', encoding="utf-8")

def remove_outliers(month: str, year: str) -> None:
    """
    Remove outliers from the data for a specified month and year. This corresponds to files that do not have the correct shape

    Args:
        month (str): A string representing the month, e.g. 'jan' for January.
        year (str): A string representing the year, e.g. '22' for 2022.
        
    Returns:
        None.

    """
    folder_location = f"images/sentinel_2/{year}_{month}"
    folder = Path(folder_location)

    for file_path in folder.iterdir():
        file = dict(np.load(file_path.joinpath('data.npz')))
        index = int(file_path.parts[-1])

        for key in file.keys():
            _, length, width = np.shape(file[key])

            if key in bands_10m:
                if min(length, width) < 200:
                    remove_date_from_stats(month, year, index)
                    shutil.rmtree(file_path)
                    break

            elif key in bands_20m:
                if min(length, width) < 100:
                    remove_date_from_stats(month, year, index)
                    shutil.rmtree(file_path)
                    break
            else:
                raise Exception(f"Invalid key {key} in file {index} for {month}-{year}")
    print(f"Outliers removed for dataset {month} {year}")

def crop_array(arr: np.ndarray, z: int) -> np.ndarray:
    _, x, y = arr.shape
    start_x = (x - z) // 2
    start_y = (y - z) // 2
    
    end_x = start_x + z
    end_y = start_y + z
    
    new_arr = np.zeros((1, z, z))
    new_arr[0,:,:] = arr[0,start_x:end_x,start_y:end_y]
    return new_arr

def crop_data(month: str, year: str) -> None:
    folder_location = f"images/sentinel_2/{year}_{month}"
    folder_path = Path(folder_location)
    min_10, min_20 = 200, 100

    for file_path in folder_path.iterdir():
        file = dict(np.load(file_path.joinpath('data.npz')))
        index = int(file_path.parts[-1])

        band_data = []
        for key in file.keys():
            band = file[key]
            _, length, width = np.shape(band)

            if key in bands_10m:
                if max(length, width) > 200:
                    band = crop_array(band, min_10)
            elif key in bands_20m:
                if max(length, width) > 100:
                    band = crop_array(band, min_20)
            else:
                raise Exception(f"Invalid key {key} in file {index} for {month}-{year}")
            
            band_data.append(band)
            
        np.savez(
            file_path.joinpath('data.npz'),
            **{band: data for band, data in zip(bands, band_data)}
        )
    print(f"Croped the data for dataset {month} {year}")

def get_coordinates_from_locations() -> list:
    coordinates_path = Path("images/coordinates/points.geojson")
    with open(coordinates_path, "r") as f:
        file = json.load(f)

    coordinates = []
    for feature in file['features']:
        coordinate = feature['geometry']['coordinates']
        coordinates.append(coordinate)

    return coordinates

def write_stats_to_csv(month: str, year: str) -> None:
    folder_location = f"images/sentinel_2/{year}_{month}"
    folder_path = Path(folder_location)
    coordinates = get_coordinates_from_locations()
    stats = load_stats(month, year)

    for file_path in folder_path.iterdir():
        file = dict(np.load(file_path.joinpath('data.npz')))
        index = int(file_path.parts[-1])

        _, b02_len, b02_wid = np.shape(file["B02"])
        _, b03_len, b03_wid = np.shape(file["B03"])
        _, b04_len, b04_wid = np.shape(file["B04"])
        _, b05_len, b05_wid = np.shape(file["B05"])
        _, b8A_len, b8A_wid = np.shape(file["B8A"])

        stats[index][2] = f"{coordinates[index][0]}-{coordinates[index][1]}"
        stats[index][3] = f"({b02_len}-{b02_wid})"
        stats[index][4] = f"({b03_len}-{b03_wid})"
        stats[index][5] = f"({b04_len}-{b04_wid})"
        stats[index][6] = f"({b05_len}-{b05_wid})"
        stats[index][7] = f"({b8A_len}-{b8A_wid})"

    # clean_stats = stats[~np.isnan(stats).any(axis=1)]
    pd_stats = pd.DataFrame(stats)
    clean_stats = pd_stats.mask(pd_stats.eq("None")).dropna().to_numpy()
    
    save_stats(month, year, clean_stats)
    print(f"Saved the stats of dataset {month} {year}")

def preprocess_s2_data(month: str, year: str) -> None:
    remove_outliers(month, year)
    # TODO: Leave out for now as it might be helpful later when georeferencing
    crop_data(month, year)
    write_stats_to_csv(month, year)