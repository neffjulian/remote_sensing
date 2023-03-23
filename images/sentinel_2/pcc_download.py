import numpy as np
from PIL import Image
import pystac_client
import rasterio
from rasterio import features, warp, windows
from pystac.extensions.eo import EOExtension as eo

import planetary_computer
import json


def get_data(catalog, coordinates, time_of_interest, name):
    area_of_interest = {"type": "Polygon", "coordinates": coordinates}

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=area_of_interest,
        datetime=time_of_interest,
        query={"eo:cloud_cover": {"lt": 10}},
    )

    items = search.item_collection()
    least_cloudy_item = min(items, key=lambda item: eo.ext(item).cloud_cover)
    bands = [
        "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"
    ]
    band_data = []

    for band in bands:
        asset_href = least_cloudy_item.assets[band].href

        with rasterio.open(asset_href) as ds:
            aoi_bounds = features.bounds(area_of_interest)
            warped_aoi_bounds = warp.transform_bounds("epsg:4326", ds.crs, *aoi_bounds)
            aoi_window = windows.from_bounds(transform=ds.transform, *warped_aoi_bounds)
            band_data.append(ds.read(window=aoi_window))

    np.savez(f'../../images/test_image/s2_{name}.npz', 
        B01=band_data[0], B02=band_data[1], B03=band_data[2], B04=band_data[3], 
        B05=band_data[4], B06=band_data[5], B07=band_data[6], B08=band_data[7], 
        B8A=band_data[8], B09=band_data[9], B11=band_data[10], B12=band_data[11])

    asset_href = least_cloudy_item.assets['visual'].href

    with rasterio.open(asset_href) as ds:
        aoi_bounds = features.bounds(area_of_interest)
        warped_aoi_bounds = warp.transform_bounds("epsg:4326", ds.crs, *aoi_bounds)
        aoi_window = windows.from_bounds(transform=ds.transform, *warped_aoi_bounds)
        band_data = ds.read(window=aoi_window)

    img = Image.fromarray(np.transpose(band_data, axes=[1, 2, 0]))
    img.save(f'../../images/test_image/s2_{name}.png')

def load_data():
    # TODO when evaluating baseline

    loaded = np.load('../../images/test_image/s2_data.npz')
    # b01 = loaded['B01']

def get_coordinates():
    with open('../coordinates.geojson', 'r') as f:
        file = json.load(f)

    coordinates = []

    for feature in file['features']:
        print(feature['geometry']['coordinates'])
        coordinates.append(feature['geometry']['coordinates'])

    return coordinates


def main():
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    time_of_interest = "2022-06-01/2022-08-01"
    coordinates = get_coordinates()
    index = 0

    for coordinate in coordinates:
        index += 1
        name = str(index).zfill(4)
        get_data(catalog, coordinate, time_of_interest, name)

if __name__ == "__main__":
    main()