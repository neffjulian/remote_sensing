import json
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image
import planetary_computer
import rasterio
from rasterio import features, warp, windows
import pystac_client
from pystac.extensions.eo import EOExtension as eo

SENTINEL_BANDS = ["B02", "B03", "B04", "B05", "B8A"]
MONTHS = {"jan": "01", "feb": "02", "mar": "03", "apr": "04", 
          "may": "05", "jun": "06", "jul": "07", "aug": "08", 
          "sep": "09", "oct": "10", "nov": "11", "dec": "12"}

COORDINATE_PATH = Path("images/coordinates/squares.geojson")

def create_folder(month: str, year: str, index: int = None) -> Path:
    """
    Creates a new folder for storing Sentinel-2 images.

    Args:
        month (str): The month of the images to be stored, e.g. "aug".
        year (str): The year of the images to be stored, e.g. "22".
        index (int, optional): An optional index number for the folder.
            If provided, the folder will be named "{year}_{month}/{index:04d}".
            If not provided, the folder will be named "{year}_{month}".

    Returns:
        Path: A Path object representing the path to the newly created folder.
    """
    folder_location = f"images/sentinel_2/{year}_{month}"
    folder_path = Path(folder_location)

    if index is not None:
        folder_path = folder_path.joinpath(f"{index:04d}")

    if not folder_path.exists():
        folder_path.mkdir()

    return folder_path

def create_stats(month: str, year: str) -> None:
    """Create a CSV file with empty statistics data for Sentinel-2 images.

    Args:
        month (str): The month of the images to be stored, e.g. "aug".
        year (str): The year of the images to be stored, e.g. "22".

    Returns:
        None

    """
    stats_location = f"images/sentinel_2/stats/{year}_{month}_stats.csv"
    stats_path = Path(stats_location)

    header = "index, date, coordinate, B02, B03, B04, B05, B8A"
    data = np.empty((296, 8), dtype=object)
    data[:, 0] = np.arange(0, 296)
    np.savetxt(stats_path, data, header=header, delimiter=',', fmt='%s', encoding="utf-8")

def write_date_to_stats(month: str, year: str, date: str, index: int) -> None:
    """Write a date to the statistics CSV file for a specified index.

    Args:
        month (str): The month of the images to be stored, e.g. "aug".
        year (str): The year of the images to be stored, e.g. "22".
        date (str): A string representing the date in YYYY-MM-DD format, e.g. '2022-01-01' for January 1st, 2022.
        index (int): An integer representing the index of the row to update in the CSV file.

    Returns:
        None

    """
    stats_location = f"images/sentinel_2/stats/{year}_{month}_stats.csv"
    stats_path = Path(stats_location)

    header = "index, date, coordinate, B02, B03, B04, B05, B8A"
    data = np.loadtxt(stats_path, delimiter = ',', dtype=object)

    data[index, 1] = date

    np.savetxt(stats_path, data, header=header, delimiter=',', fmt='%s', encoding="utf-8")

def get_coordinates() -> list:
    """
    Retrieves a list of coordinates from a GeoJSON file.

    Returns:
        List: A list of lists of coordinates.
    """    
    file = COORDINATE_PATH.open()
    geojson_coordinates = json.load(file)
    
    coordinates = []
    for feature in geojson_coordinates["features"]:
        coordinate = feature["geometry"]["coordinates"]
        coordinates.append(coordinate)

    return coordinates


def check_data_valid(data: np.ndarray) -> bool:
    """
    Checks if the input data is valid by counting the percentage of zero values.

    Args:
        data (np.ndarray): A NumPy array containing the data to be checked.

    Returns:
        bool: True if the percentage of zero values is less than 5% (e.g. black pixels), False otherwise.
    """
    num_zeros = np.count_nonzero(data == 0)
    per_zeros = (num_zeros / data.size)
    return per_zeros < 0.05

def read_band_data(asset_href, area_of_interest):
    """Read the data for the given band from the Sentinel-2 L2A asset at the given href.

    Args:
        asset_href (str): The href for the asset to read.
        area_of_interest (dict): A dictionary describing the area of interest in GeoJSON format.

    Returns:
        numpy.ndarray: The data for the given band.
    """
    with rasterio.open(asset_href) as ds:
        aoi_bounds = features.bounds(area_of_interest)
        warped_aoi_bounds = warp.transform_bounds("epsg:4326", ds.crs, *aoi_bounds)
        aoi_window = windows.from_bounds(transform=ds.transform, *warped_aoi_bounds)
        return ds.read(window=aoi_window)
    
def save_as_np_array(least_cloudy_item, bands, area_of_interest, folder_path: Path):
    """
    Saves a set of Sentinel-2 bands as a NumPy array in the specified folder.

    Args:
        least_cloudy_item (pystac.item.Item): A Sentinel-2 image with the least amount of clouds.
        bands (List[str]): A list of band names to save as a NumPy array.
        area_of_interest (dict): The area of interest to crop the data.
        folder_path (Path): The path to the folder where the NumPy array will be saved.
    """
    band_data = []
    for band in bands:
        if band not in least_cloudy_item.assets:
            raise ValueError(f"Band {band} is not available in the Sentinel-2 image")
        asset_href = least_cloudy_item.assets[band].href
        band_data.append(read_band_data(asset_href, area_of_interest))
    
    np.savez(
        folder_path.joinpath("data"),
        **{band: data for band, data in zip(bands, band_data)}
    )

def save_as_visual_plot(img_data, folder_path: Path):
    """Save image data as a PNG file in the specified folder path.

    Args:
        img_data (numpy.ndarray): An array containing image data.
        folder_path (pathlib.Path): A Path object representing the folder path where the PNG file will be saved.

    Returns:
        None

    """
    img = Image.fromarray(np.transpose(img_data, axes=[1, 2, 0]))
    img.save(folder_path.joinpath("plot.png"))

def get_data_from_coordinate(catalog, coordinate, month, year, index) -> int:
    """Get image data from a specified coordinate and save it as a PNG and NumPy array.

    Args:
        catalog (pystac_client.Client): A PySTAC client object.
        coordinate (list): A list of coordinate pairs specifying a polygonal area of interest.
        month (str): The month of the images to be stored, e.g. "aug".
        year (str): The year of the images to be stored, e.g. "22".
        index (int): An integer representing the index of the row to update in the CSV file.

    Returns:
        int: 1 if the image data is successfully retrieved and saved, 0 otherwise.

    """
    area_of_interest = {"type": "Polygon",
                        "coordinates": coordinate}
    time_of_interest = f"20{year}-{MONTHS[month]}-01/20{year}-{MONTHS[month]}-28"
    try:
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            intersects=area_of_interest,
            datetime=time_of_interest,
            query={"eo:cloud_cover": {"lt": 25}},
        )
        items = search.item_collection()
        least_cloudy_item = min(items, key=lambda item: eo.ext(item).cloud_cover)
        visual_href = least_cloudy_item.assets["visual"].href
        img_data = read_band_data(visual_href, area_of_interest)
        
        if check_data_valid(img_data):
            date = least_cloudy_item.datetime.strftime("%Y-%m-%d")
            folder_path = create_folder(month, year, index)
            save_as_np_array(least_cloudy_item, SENTINEL_BANDS, area_of_interest, folder_path)
            save_as_visual_plot(img_data, folder_path)
            write_date_to_stats(month, year, date, index)
            return 1
        else:
            return 0
    except:
        return 0

def download_s2_data(month: str, year: str) -> None:
    create_folder(month, year)
    create_stats(month, year)

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    coordinates = get_coordinates()

    successful_downloads = 0
    for index, coordinate in enumerate(coordinates):
        successful_downloads += get_data_from_coordinate(catalog, coordinate, month, year, index)

    print(f"Successfully downloaded {successful_downloads} images for {month} {year}")
