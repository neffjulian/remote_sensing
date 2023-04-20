import json
import os

import numpy as np
import planetary_computer
import pystac_client
import rasterio
from PIL import Image
from pystac.extensions.eo import EOExtension as eo
from rasterio import features, warp, windows

# Define the bands to retrieve
bands = ["B02", "B03", "B04", "B05", "B8A"]

# Define a dictionary mapping month names to date ranges
MONTHS = {
    'january': ("01-01", "01-31"),
    'february': ("02-01", "02-28"),
    'march': ("03-01", "03-31"),
    'april': ("04-01", "04-30"),
    'may': ("05-01", "05-31"),
    'june': ("06-01", "06-30"),
    'july': ("07-01", "07-31"),
    'august': ("08-01", "08-31"),
    'september': ("09-01", "09-30"),
    'october': ("10-01", "10-31"),
    'november': ("11-01", "11-30"),
    'december': ("12-01", "12-31"),
}

def query_catalog(catalog, area_of_interest, time_of_interest):
    """Query the catalog for Sentinel-2 L2A collections that intersect the given area of interest
    and have a cloud cover of less than 10% at the given time.

    Args:
        catalog (pystac_client.Catalog): The STAC catalog to query.
        area_of_interest (dict): A dictionary describing the area of interest in GeoJSON format.
        time_of_interest (str): A datetime string in ISO 8601 format.

    Returns:
        pystac_client.ItemCollection: A collection of STAC items matching the query parameters.
    """
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=area_of_interest,
        datetime=time_of_interest,
        query={"eo:cloud_cover": {"lt": 25}},
    )
    return search.item_collection()

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
    
def save_band_data(bands, band_data, name, foldername):
    """Save the data for multiple bands to a compressed .npz file.

    Args:
        bands (list): A list of band names.
        band_data (list): A list of band data arrays.
        name (str): The name of the .npz file to create.
        foldername (str): The folder to save the .npz file in.
    """
    np.savez(
        f"images/sentinel_2/{foldername}/{name}.npz",
        **{band: data for band, data in zip(bands, band_data)}
    )

def process_data(least_cloudy_item, bands, area_of_interest, name, foldername):
    """Process the data for a Sentinel-2 L2A item by saving it to a compressed .npz file and
    converting it to a PNG image file.

    Args:
        least_cloudy_item (pystac_client.Item): The least cloudy item for the given area and time.
        bands (list): A list of band names to process.
        area_of_interest (dict): A dictionary describing the area of interest in GeoJSON format.
        name (str): The name to use for the processed data file and image.
        foldername (str): The folder to save the files in.
    """
    band_data = []
    for band in bands:
        asset_href = least_cloudy_item.assets[band].href
        band_data.append(read_band_data(asset_href, area_of_interest))
    save_band_data(bands, band_data, name, foldername)

def convert_to_image_and_save(img_data, name, foldername):
    """
    Convert numpy array image data to a PIL Image and save it as a PNG file.

    Args:
        img_data (numpy.ndarray): Numpy array containing image data.
        name (str): Name of the file to be saved.
        foldername (str): Name of the folder where the file is to be saved.
    """
    img = Image.fromarray(np.transpose(img_data, axes=[1, 2, 0]))
    img.save(f'images/sentinel_2/{foldername}/{name}.png')

def get_coordinates():
    """
    Load geojson file containing coordinates and extract the coordinates of each feature.

    Returns:
        List of tuples, where each tuple represents the coordinates of a feature.
    """
    with open("images/coordinates/squares.geojson", "r") as f:
        file = json.load(f)
        f.close()
    coordinates = []
    for feature in file["features"]:
        coordinates.append(feature["geometry"]["coordinates"])
    return coordinates

def open_catalog():
    """
    Open a connection to the Planetary Computer STAC API.

    Returns:
        pystac_client.Client: Instance of the Planetary Computer STAC API client.
    """
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

def is_valid(file):
    """
    Check if the data contains any large black patches.

    Args:
        file (numpy.ndarray): Numpy array containing image data.

    Returns:
        bool: True if percentage of zero values is less than 0.01, False otherwise.
    """
    num_zeros = np.count_nonzero(file == 0)
    per_zeros = (num_zeros / file.size)
    return per_zeros < 0.05

def create_folder(name):
    """
    Create a folder with the given name if it does not exist.

    Args:
        name (str): Name of the folder to be created.
    """
    if not os.path.exists(f"images/sentinel_2/{name}"):
        os.mkdir(f"images/sentinel_2/{name}")

def get_data(catalog, coordinates, time_of_interest, index, foldername) -> int:
    """
    Get Sentinel-2 L2A image data for the given coordinates and time of interest, and save it as a .npz and .png file.

    Args:
        catalog (pystac_client.Client): Instance of the Planetary Computer STAC API client.
        coordinates (list of tuple): List of tuples, where each tuple represents the coordinates of a feature.
        time_of_interest (str): Date and time of interest in ISO 8601 format.
        index (int): Index of the coordinate in the list.
        foldername (str): Name of the folder where the files are to be saved.
    """
    name = f"{index:04d}"
    area_of_interest = {"type": "Polygon", "coordinates": coordinates}
    try:
        items = query_catalog(catalog, area_of_interest, time_of_interest)
        least_cloudy_item = min(items, key=lambda item: eo.ext(item).cloud_cover)
        visual_href = least_cloudy_item.assets["visual"].href
        img_data = read_band_data(visual_href, area_of_interest)

        if is_valid(img_data):
            process_data(least_cloudy_item, bands, area_of_interest, name, foldername)
            convert_to_image_and_save(img_data, name, foldername)
    except:
        return 0
    return 1

def download_s2_data(month: str, year: str):
    """Downloads Sentinel-2 data for a given time of interest and saves it to disk.

    Args:
        month (str): Name of the month the data should be collected from
        year (str): The year during which the data should be collected from

    Returns:
        None
    """

    time_of_interest = f"20{year[-2:]}-{MONTHS[month][0]}/20{year[-2:]}-{MONTHS[month][1]}"
    name = f"{year[-2:]}_{month}"

    create_folder(name)
    catalog = open_catalog()
    coordinates = get_coordinates()

    successful_downloads = 0
    for index, coordinate in enumerate(coordinates):
        successful_downloads += get_data(catalog, coordinate, time_of_interest, index, name)
    
    print(f"Successfully downloaded {successful_downloads} images for {month}-{year}")