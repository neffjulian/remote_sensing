import json
from pathlib import Path
import geopandas as gpd
from pystac_client import Client
from eodal.core.raster import RasterCollection
from shapely.geometry import box

COORDINATE_PATH = Path("images/coordinates/squares_1.geojson")
STAC_API_URL = "https://data.geo.admin.ch/api/stac/v0.9"
COLLECTION = "ch.swisstopo.swissimage-dop10"


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
    folder_location = f"images/swissimage/{year}_{month}"
    folder_path = Path(folder_location)

    if index is not None:
        folder_path = folder_path.joinpath(f"{index:04d}")

    folder_path.mkdir(exist_ok=True)

    return folder_path

def get_coordinates() -> list:
    """
    Retrieves a list of coordinates from a GeoJSON file.

    Returns:
        List: A list of lists of coordinates.
    """    
    with open(COORDINATE_PATH, "r") as f:
        geojson_coordinates = json.load(f)
    
    coordinates = []
    for feature in geojson_coordinates["features"]:
        gdf = gpd.GeoDataFrame.from_features([feature])
        gdf = gdf.set_crs(epsg=4326)
        coordinates.append(gdf)
        print(box(*gdf.total_bounds))

    return coordinates

def download_si_data(month:str, year:int) -> int:
    create_folder(month, year)
    orders = get_coordinates()
    catalog = Client.open(STAC_API_URL)
    capture = []

    for index, order in enumerate(orders):
        download_dir = create_folder(month, year, index)
        search = catalog.search(
            collections=COLLECTION,
            intersects=box(*order.total_bounds)
        )
        items = search.get_all_items()
        item_json = items.to_dict()
        scenes = {'0.1': [], '2': []}
        dates = []

        for item in item_json['features']:
            filenames = list(item['assets'].keys())
            date = item['properties']['created']
            dates.append(date[:10])

        dates = list(set(dates))
        capture.append(dates)

        for filename in filenames:
            spatial_resolution = filename.split('_')[3]
            scenes[spatial_resolution].append(item['assets'][filename]['href'])
        
        for resolution in scenes.keys():
            out_dir_res = download_dir.joinpath(resolution)
            out_dir_res.mkdir(exist_ok=True)

            
        url_list = scenes[resolution]

        for url in url_list:
        # check year of the tile
            fname = url.split('/')[-1]
            out_dir_res.mkdir(exist_ok=True)
            ds = RasterCollection.from_multi_band_raster(fpath_raster=url , nodata=0)
            ds.to_rasterio(out_dir_res.joinpath(fname))

            # save dataset bounds as geojson
            bound_df = gpd.GeoDataFrame(geometry=[ds['B1'].bounds],
            crs=ds['B1'].geo_info.epsg)
            fname_bbox = fname.replace('.tif', '_bbox.geojson')
            bound_df.to_file(out_dir_res.joinpath(fname_bbox))

        print(url_list)