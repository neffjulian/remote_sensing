# Download files. store as .tif

import json
import requests
from pathlib import Path

from PIL import Image
import numpy as np
import rasterio
from rasterio import features, warp, windows
from osgeo import gdal
import pystac_client
from pystac.extensions.eo import EOExtension as eo
import planetary_computer

SOURCE_DIR = Path('images/sentinel_2')
COORDINATE_PATH = Path('images/coordinates/squares_1.geojson')

SENTINEL_BANDS = ["B02", "B03", "B04", "B05", "B8A"]

MONTHS = {"jan": "01", "feb": "02", "mar": "03", "apr": "04", 
          "may": "05", "jun": "06", "jul": "07", "aug": "08", 
          "sep": "09", "oct": "10", "nov": "11", "dec": "12"}


def create_stats(year: str, month: str) -> None:
    """Create a CSV file with empty statistics data for Sentinel-2 images.

    Args:
        month (str): The month of the images to be stored, e.g. "aug".
        year (str): The year of the images to be stored, e.g. "22".

    Returns:
        None

    """
    
    stats_name = f'stats/{year}/'
    stats_path = SOURCE_DIR.joinpath(stats_name)
    stats_path.mkdir(parents=True)

    file_name = f'{MONTHS[month]}_{month}_.csv'
    file_location = stats_path.joinpath(file_name)
    
    header = "index, date, coordinate"
    data = np.empty((296, 8), dtype=object)
    data[:, 0] = np.arange(0, 296)
    np.savetxt(file_location, data, header=header, delimiter=',', fmt='%s', encoding="utf-8")

def write_date_to_stats(year: str, month: str, date: str, index: int) -> None:
    """Saves a file with the dates, when the pictures have been captured and where.

    Args:
        month (str): The month of the images to be stored, e.g. "aug".
        year (str): The year of the images to be stored, e.g. "2022".
        date (str): A string representing the date in YYYY-MM-DD format, e.g. '2022-01-01' for January 1st, 2022.
        index (int): An integer representing the index of the row to update in the CSV file.

    Returns:
        None

    """
    stats_filename = f'stats/{year}/{MONTHS[month]}_{month}_.csv'
    stats_path = SOURCE_DIR.joinpath(stats_filename)

    header = "index, date, coordinate"
    data = np.loadtxt(stats_path, delimiter = ',', dtype=object)

    data[index, 1] = date

    np.savetxt(stats_path, data, header=header, delimiter=',', fmt='%s', encoding="utf-8")

def get_coordinates() -> list:
    """
    Retrieves a list of coordinates from a GeoJSON file.

    Returns:
        List: A list of lists of coordinates.
    """    
    coordinate_file = COORDINATE_PATH.open()
    coordinate_geojson = json.load(coordinate_file)
    
    coordinates = []
    for feature in coordinate_geojson["features"]:
        coordinate = feature["geometry"]["coordinates"]
        coordinates.append(coordinate)

    return coordinates

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

def read_band_data(asset_href, area_of_interest) -> np.ndarray:
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
    
    
def store_metadata(metadata_href, folder):
    pass

    # granule_metadata = requests.get(metadata_href).content
    # data_xml = BeautifulSoup(granule_metadata, "xml")
    # TODO: Look at it with Lukas what is important

def save_data_as_tif(least_cloudy_item, area_of_interest, folder_path: Path, date: str):
    """
    Saves a set of Sentinel-2 bands as a tif file in the specified folder.

    Args:
        least_cloudy_item (pystac.item.Item): A Sentinel-2 image with the least amount of clouds.
        bands (List[str]): A list of band names to save as a NumPy array.
        area_of_interest (dict): The area of interest to crop the data.
        folder_path (Path): The path to the folder where the tif file will be saved.
    """
    band_names = ["Blue", "Green", "Red", "RedEdge", "NIR"]
    band_data = []
    for band in SENTINEL_BANDS:
        asset_href = least_cloudy_item.assets[band].href
        band_as_np = read_band_data(asset_href, area_of_interest)
        band_data.append(np.squeeze(band_as_np))
    
    max_len = max([arr.shape[0] for arr in band_data])
    max_wid = max([arr.shape[1] for arr in band_data])

    # Returns array of size 5 with dtype unit16

    driver = gdal.GetDriverByName('GTiff')
    filename = folder_path.joinpath('data.tif').as_posix()
    out_file = driver.Create(filename, max_wid, max_len, 5, gdal.GDT_UInt16)

    for i, band in enumerate(band_data):
        len, wid = band.shape
        len_offset = max_len - len
        wid_offset = max_wid - wid
        out_band = out_file.GetRasterBand(i+1)
        out_band.WriteArray(np.squeeze(band), wid_offset, len_offset)
        out_band.SetDescription(band_names[i])

    out_file.SetMetadataItem("CREATION_DATE", date)
    out_file.SetProjection("EPSG:4326")

def get_data_from_coordinate(catalog: pystac_client.Client, coordinate, year: str, month: str, index) -> bool:
    area_of_interest = {"type": "Polygon",
                        "coordinates": coordinate}
    time_of_interest = f"{year}-{MONTHS[month]}-01/{year}-{MONTHS[month]}-28"
    
    # try:
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
        folder_path = SOURCE_DIR.joinpath(f'{year}', f'{MONTHS[month]}_{month}', f'{index}')
        folder_path.mkdir(exist_ok=True, parents=True)
        date = least_cloudy_item.datetime.strftime("%Y-%m-%d")
        save_data_as_tif(least_cloudy_item, area_of_interest, folder_path, date)
        save_as_visual_plot(img_data, folder_path)

        # 
        # write_date_to_stats(year, month, date, index)

        # asset_metadata = least_cloudy_item.assets["granule-metadata"].href
        # store_metadata(asset_metadata, folder)

        return True
        
    # except:

        # pass

    return False

def download_s2_data(year: str, month: str) -> None:
    data_dir = SOURCE_DIR.joinpath(f'{year}', f'{MONTHS[month]}_{month}')

    try:
        data_dir.mkdir(exist_ok=True, parents=True)
    except:
        raise Exception(f"Folder '{data_dir}' exists already. Please delete it first.")

    # create_stats(year, month)
    coordinates = get_coordinates()
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )


    successful_downloads = 0
    for index, coordinate in enumerate(coordinates):
        successful_downloads += get_data_from_coordinate(catalog, coordinate, year, month, index)

    print(f"Successfully downloaded {successful_downloads} images for {month} {year}")
