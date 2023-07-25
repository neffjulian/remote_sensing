import numpy as np
import pandas as pd
from pathlib import Path
from eodal.core.raster import RasterCollection
from eodal.core.band import Band, GeoInfo
import matplotlib.pyplot as plt

s2_dir = Path(__file__).parent.parent.joinpath('data', 'processed', '20m_in_situ')
ps_dir = Path(__file__).parent.parent.joinpath('data', 'processed', '4b_in_situ')

filter_2s_dir = Path(__file__).parent.parent.joinpath('data', 'filtered', 'sentinel')
filter_ps_dir = Path(__file__).parent.parent.joinpath('data', 'filtered', 'planetscope')

results_dir = Path(__file__).parent.parent.joinpath('data', 'results')

def main():
    months = {"jan": "01", "feb": "02", "mar": "03", "apr": "04", 
              "may": "05", "jun": "06", "jul": "07", "aug": "08", 
              "sep": "09", "oct": "10", "nov": "11", "dec": "12"}
    
    s2_tif_file = results_dir.joinpath('0001_scene_20m_lai.tif')
    sr_npy_file = results_dir.joinpath('0001.npy')
    ps_tif_file = results_dir.joinpath('0001_lai_4bands.tif')

    s2_data = RasterCollection.from_multi_band_raster(s2_tif_file)
    ps_data = RasterCollection.from_multi_band_raster(ps_tif_file)
    sr_data = np.load(sr_npy_file)

    s2_geo_info = s2_data["lai"].geo_info
    ps_geo_info = ps_data["lai"].geo_info

    max_diff = 0

                # if(s2_diff - ps_diff > 100):
                #     print(i, year.name, month.name, s2_geo_info.uly, s2_geo_info.ulx, ps_geo_info.uly, ps_geo_info.ulx)


                
    print(max_diff)
    # for year in filter_ps_dir.iterdir():
    #     for month in year.iterdir():
    #         for filename in month.iterdir():
    #             print(filename)
    # print(s2_geo_info.uly - s2_geo_info.ulx)
    # print(ps_geo_info.uly - ps_geo_info.ulx)

    # band_name = "lai"
    # geo_info = s2_data["lai"].geo_info
    # epsg = geo_info.epsg
    # ulx = geo_info.ulx
    # uly = geo_info.uly
    # pixres_x = 3.33
    # pixres_y = -3.33

    # new_geo_info = GeoInfo(
    #     epsg=epsg,
    #     ulx=ulx,
    #     uly=uly,
    #     pixres_x=pixres_x,
    #     pixres_y=pixres_y
    # )

    # epsg = 32633

    # raster = RasterCollection(
    #     band_constructor=Band,
    #     band_name=band_name,
    #     values = sr_data,
    #     geo_info=new_geo_info
    # )

    # # raster.to_rasterio()
    # # TEST IN QGIS!

    # f = raster.plot_band("lai", colormap="viridis", vmin=0, vmax=8)
    # f.show()
    # plt.show()


if __name__ == '__main__':
    main()