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
bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12",]

def query_catalog(catalog, area_of_interest, time_of_interest):
    # Search for Sentinel-2 L2A collections that intersect the area of interest,
    # at the given time and have a cloud cover of less than 10%
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=area_of_interest,
        datetime=time_of_interest,
        query={"eo:cloud_cover": {"lt": 10}},
    )
    return search.item_collection()

def read_band_data(asset_href, area_of_interest):
    with rasterio.open(asset_href) as ds:
        # Get the bounds of the area of interest
        aoi_bounds = features.bounds(area_of_interest)
        # Transform the bounds to match the coordinate reference system (CRS) of the dataset
        warped_aoi_bounds = warp.transform_bounds("epsg:4326", ds.crs, *aoi_bounds)
        # Get the window corresponding to the area of interest
        aoi_window = windows.from_bounds(transform=ds.transform, *warped_aoi_bounds)
        # Read the data from the dataset for the window corresponding to the area of interest
        return ds.read(window=aoi_window)
    
def save_band_data(bands, band_data, name, foldername):
    np.savez(
        f"{foldername}/{name}.npz",
        **{band: data for band, data in zip(bands, band_data)}
    )

def process_data(least_cloudy_item, bands, area_of_interest, name, foldername):
    band_data = []
    # Loop through each band and get its data for the least cloudy item
    for band in bands:
        asset_href = least_cloudy_item.assets[band].href
        band_data.append(read_band_data(asset_href, area_of_interest))
    save_band_data(bands, band_data, name, foldername)

def convert_to_image_and_save(img_data, name, foldername):
    img = Image.fromarray(np.transpose(img_data, axes=[1, 2, 0]))
    img.save(f'{foldername}/{name}.png')

def get_coordinates():
    with open("../coordinates/squares.geojson", "r") as f:
        file = json.load(f)
        f.close()
    coordinates = []
    for feature in file["features"]:
        coordinates.append(feature["geometry"]["coordinates"])
    return coordinates

def open_catalog():
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

def is_valid(file):
    num_zeros = np.count_nonzero(file == 0)
    per_zeros = (num_zeros / file.size)
    return per_zeros < 0.01

def create_folder(name):
    if not os.path.exists(name):
        os.mkdir(name)

def get_data(catalog, coordinates, time_of_interest, index, foldername):
    # Generate a name for the image based on the index
    name = f"{index:04d}"
    # Create a polygon for the area of interest using the given coordinates
    area_of_interest = {"type": "Polygon", "coordinates": coordinates}

    # Query the catalog for Sentinel-2 L2A collections that intersect the area of interest
    # at the given time and have a cloud cover of less than 10%
    try:
        items = query_catalog(catalog, area_of_interest, time_of_interest)
        # Get the least cloudy item from the query result
        least_cloudy_item = min(items, key=lambda item: eo.ext(item).cloud_cover)

        # Get the visual asset href for the least cloudy item
        visual_href = least_cloudy_item.assets["visual"].href
        # Read the band data for the visual asset and the area of interest
        img_data = read_band_data(visual_href, area_of_interest)

        # Check if the image data is valid
        if is_valid(img_data):
            # Process the image data by saving it to a .npz file and converting it to an image file
            process_data(least_cloudy_item, bands, area_of_interest, name, foldername)
            convert_to_image_and_save(img_data, name, foldername)
    except:
        return

def download_data(name, time_of_interest):
    create_folder(name)
    catalog = open_catalog()
    coordinates = get_coordinates()

    for index, coordinate in enumerate(coordinates):
        get_data(catalog, coordinate, time_of_interest, index, name)