import numpy as np
import pandas as pd
from pathlib import Path
from eodal.core.raster import RasterCollection
from eodal.core.band import Band, GeoInfo
import matplotlib.pyplot as plt

s2_dir = Path(__file__).parent.parent.joinpath('data', 'processed', '20m_in_situ')
ps_dir = Path(__file__).parent.parent.joinpath('data', 'processed', '4b_in_situ')

results_dir = Path(__file__).parent.parent.joinpath('data', 'results')

def main():
    s2_tif_file = results_dir.joinpath('0001_scene_20m_lai.tif')
    sr_npy_file = results_dir.joinpath('0001.npy')
    ps_tif_file = results_dir.joinpath('0001_lai_4bands.tif')

    s2_data = RasterCollection.from_multi_band_raster(s2_tif_file)
    ps_data = RasterCollection.from_multi_band_raster(ps_tif_file)["lai"].values
    sr_data = np.load(sr_npy_file)

    band_name = "lai"
    geo_info = s2_data["lai"].geo_info
    epsg = geo_info.epsg
    ulx = geo_info.ulx
    uly = geo_info.uly

    pixres_x, pixres_y = 3.33, -3.33

    new_geo_info = GeoInfo(
        epsg=epsg,
        ulx=ulx,
        uly=uly,
        pixres_x=pixres_x,
        pixres_y=pixres_y
    )

    epsg = 32633

    raster = RasterCollection(
        band_constructor=Band,
        band_name=band_name,
        values = sr_data,
        geo_info=new_geo_info
    )

    f = raster.plot_band("lai", colormap="viridis", vmin=0, vmax=8)
    f.show()
    plt.show()


if __name__ == '__main__':
    main()