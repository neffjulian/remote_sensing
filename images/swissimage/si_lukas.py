'''
Script to download Swiss Image Tiles from SwissTopo for a user-defined
Area of Interest.
Created on Mar 14, 2023
@author: graflu
'''

import geopandas as gpd
from pathlib import Path
from pystac_client import Client
from shapely.geometry import box
from eodal.core.raster import RasterCollection

stac_api_url = 'https://data.geo.admin.ch/api/stac/v0.9'
collection = 'ch.swisstopo.swissimage-dop10'

def get_tiles(fpath_bbox: Path, out_dir: Path) -> None:
"""
Query and download tiles SwissImage tiles from SwissTopo (10cm and 2m spatial resolution)
:param fpath_bbox: geographic extent for which to download tiles
:param out_dir: directory where to store results
"""

# read bounding box file and project it to WGS84
gdf = gpd.read_file(fpath_bbox)
gdf.to_crs(epsg=4326, inplace=True)

# get STAC API clien
cat = Client.open(url=stac_api_url)

# search datasets on catalog

search = cat.search(
collections=collection,
intersects=box(*gdf.total_bounds)
)

# fetch items by their spatial resolution
items = search.get_all_items()
item_json = items.to_dict()
scenes = {'0.1': [], '2': []}

for item in item_json['features']:
    fname = list(item['assets'].keys())

    for fname in fname:
        spatial_resolution = fname.split('_')[3]

# append URL of resource (image tile)
scenes[spatial_resolution].append(item['assets'][fname]['href'])

# loop over items and save them as GeoTiff files to disk
for resolution in scenes.keys():
    out_dir_res = out_dir.joinpath(resolution)
    out_dir_res.mkdir(exist_ok=True)

    url_list = scenes[resolution]

for url in url_list:
# check year of the tile
    fname = url.split('/')[-1]
    year = fname.split('_')[1]
    out_dir_res_year = out_dir_res.joinpath(year)
    out_dir_res_year.mkdir(exist_ok=True)
    ds = RasterCollection.from_multi_band_raster(fpath_raster=url , nodata=0)
    ds.to_rasterio(out_dir_res_year.joinpath(fname))

    # save dataset bounds as geojson
    bound_df = gpd.GeoDataFrame(geometry=[ds['B1'].bounds],
    crs=ds['B1'].geo_info.epsg)
    fname_bbox = fname.replace('.tif', '_bbox.geojson')
    bound_df.to_file(out_dir_res_year.joinpath(fname_bbox))

if __name__ == '__main__':




data_dir = Path('/run/media/graflu/ETH-KP-SSD6/MA_JULIAN/Data/Eschikon')
fpath_bbox = data_dir.joinpath('Bounding_Box/bounding_box_strickhof.geojson')
out_dir = data_dir.joinpath('SwissImage')
get_tiles(fpath_bbox=fpath_bbox, out_dir=out_dir)