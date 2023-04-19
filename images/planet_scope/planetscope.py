# Based on "scripts/planet_download.py" from eodal repo
import geopandas as gpd
import os
import osgeo.gdal as gd
import matplotlib.pyplot as plt
import numpy as np
import shutil

from eodal.downloader.planet_scope import PlanetAPIClient
from eodal.config import get_settings
from pathlib import Path
from datetime import date

def download_data(start_date: date, end_date: date) -> None:
    """
    Downloads satellite images from the Planet API for a given date range and bounding box.

    Args:
        start_date: A date object representing the start date of the date range.
        end_date: A date object representing the end date of the date range.
    """
    settings = get_settings()
    settings.USE_STAC = True

    order_name = f'{date.today()}_Julian_Thesis'
    cloud_cover = 50.

    client = PlanetAPIClient.query_planet_api(
        start_date=start_date,
        end_date=end_date,
        bounding_box=gpd.read_file('coordinates/coordinates.geojson'),
        cloud_cover_threshold=cloud_cover
    )

    order_url = client.place_order(order_name=order_name)

    client.check_order_status(order_url, loop=True)

    download_dir = Path('downloaded_images/')

    client.download_order(
        download_dir=download_dir,
        order_name=order_name,
        order_url=order_url
    )

def is_analytic_image(file_name: str) -> bool:
    """
    Determines if a given file name is an analytic image.

    Args:
        file_name: A string representing the file name to be checked.

    Returns:
        True if the file name contains 'Analytic' and ends with '.tif', False otherwise.
    """
    return "Analytic" in file_name and file_name.endswith(".tif")

def is_metadata(file_name: str) -> bool:
    """
    Determines if a given file name is metadata.

    Args:
        file_name: A string representing the file name to be checked.

    Returns:
        True if the file name contains 'metadata', False otherwise.
    """
    return "metadata" in file_name

def save_image(img, location: Path) -> None:
    """
    Saves an image to a given location.

    Args:
        img: A NumPy array representing the image.
        location: A Path object representing the location where the image will be saved.
    """
    img = plt.imshow(get_band_image(img, [2, 4, 6]), interpolation='lanczos')
    plt.axis('off')
    plt.savefig(location, dpi=200, bbox_inches='tight', pad_inches = 0)

def normalize(array: np.ndarray) -> np.ndarray:
    """
    Normalizes a given NumPy array.

    Args:
        array: A NumPy array to be normalized.

    Returns:
        A normalized NumPy array.
    """
    min_value, max_value = array.min(), array.max()
    return (array - min_value) / (max_value - min_value)

def brighten(band: np.ndarray, alpha: float = 0.13, beta: float = 0) -> np.ndarray:
    """
    Brightens a given NumPy array.

    Args:
        band: A NumPy array to be brightened.
        alpha: A float representing the alpha value for the brightening (default 0.13).
        beta: A float representing the beta value for the brightening (default 0).

    Returns:
        A brightened NumPy array.
    """
    return np.clip(alpha*band+beta, 0,255)

def gammacorr(band: np.ndarray, gamma=2) -> np.ndarray:
    """
    Applies a gamma correction to an image band.

    Args:
        band: A numpy array representing a band of an image.
        gamma: A float indicating the gamma value for the correction (default 2).

    Returns:
        A numpy array representing the band after gamma correction.
    """
    return np.power(band, 1/gamma)

def get_band_array(img, band):
    """
    Reads a band of a raster image into a numpy array, applies normalization,
    brightness correction, and gamma correction to the array.

    Args:
        img: An instance of the GDAL image class representing the raster image.
        band: An integer indicating the band to be read.

    Returns:
        A numpy array representing the band after preprocessing.
    """
    rasterband = img.GetRasterBand(band)
    array = rasterband.ReadAsArray()
    print(np.shape(array), band)
    return normalize(brighten(gammacorr(array)))

def get_band_image(img, band_indices):
    """
    Reads multiple bands of a raster image into numpy arrays and stacks them
    into a single 3D array.

    Args:
        img: An instance of the GDAL image class representing the raster image.
        band_indices: A list of integers indicating the bands to be read.

    Returns:
        A numpy array representing the image after preprocessing.
    """
    band_arrays = [get_band_array(img, band_index) for band_index in band_indices]
    return np.dstack(band_arrays)

def preprocess_data():
    """
    Preprocesses a directory of raster images by reading specific bands into
    numpy arrays, applying preprocessing, and saving the resulting arrays and
    a visual representation of the image to disk.
    """
    source_dir = Path('22_may')
    target_dir = Path('22_may')
    for root, _, files in os.walk(source_dir):
        for file in files:
            curr_file = os.path.join(root, file)
            new_dir = target_dir / Path(root).name
            new_dir.mkdir(parents=True, exist_ok=True)

            if is_analytic_image(file):
                print(file)
                image_data = gd.Open(Path(curr_file).as_posix(), gd.GA_ReadOnly)
                image = get_band_image(image_data, [2, 4, 6])
                plt.imshow(image)
                save_image(image_data, new_dir / "plot.png")
                data = get_band_image(image_data, [2, 4, 6, 7, 8])
                np.save(new_dir / "data.npy", data)
            elif is_metadata(file):
                new_file = new_dir / "metadata.json"
                shutil.copyfile(curr_file, new_file)

if __name__ == "__main__":
    # download_data(
    #     start_date = date(2022,3,1),
    #     end_date = date(2022,3,31)
    # )

    preprocess_data()
