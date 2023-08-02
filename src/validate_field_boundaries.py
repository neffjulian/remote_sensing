from pathlib import Path

import cv2
import numpy as np
from eodal.core.raster import RasterCollection
from eodal.core.band import Band, GeoInfo
from scipy.stats import spearmanr

VALIDATE_DIR = Path(__file__).parent.parent.joinpath("data", "validate")

def compare_field_boundaries(raster_path: Path, field_boundary_path: Path) -> None:
    raster = RasterCollection.from_multi_band_raster(raster_path)["lai"]
    field_boundary = RasterCollection.from_multi_band_raster(field_boundary_path)["lai"]

    

    if raster.geo_info.epsg != field_boundary.geo_info.epsg:
        print("EPSG codes do not match.", raster_path.name, field_boundary_path.name)
        return
    
    print(raster.values.shape, field_boundary.values.shape)
    print(raster.geo_info, field_boundary.geo_info)

    # Get difference in coordinates
    ulx_diff = field_boundary.geo_info.ulx - raster.geo_info.ulx
    uly_diff = raster.geo_info.uly - field_boundary.geo_info.uly

    # Scale SR data to have equal pixel resolution
    scale = raster.geo_info.pixres_x / field_boundary.geo_info.pixres_x
    y, x = raster.values.shape
    raster_shape = (int(x * scale), int(y * scale))
    raster_values = cv2.resize(raster.values, raster_shape, interpolation=cv2.INTER_CUBIC)

    # Get values at exact coordinates of field boundary
    x, y = int(uly_diff / 3), int(ulx_diff / 3)
    size_x, size_y = field_boundary.values.shape
    values = raster_values[x:x+size_x, y:y+size_y]

    correlation_coefficient, p_value = spearmanr(field_boundary.values.flatten(), values.flatten())
    print(correlation_coefficient, p_value)
    # In this example the correlation coefficient is 0.689 and the p-value is 1.15e-26 (nearly zero)

    # new_raster = RasterCollection(
    #     band_constructor=Band,
    #     band_name = "lai",
    #     values = values,
    #     geo_info = field_boundary.geo_info
    # )

    # new_raster.to_rasterio("test.tif")


def main() -> None:
    bicubic = VALIDATE_DIR.joinpath("20m_bicubic", "03_mar", "0000.tif")
    field_boundary = VALIDATE_DIR.joinpath("4b", "03_mar", "0000_field_boundaries", "17_reference_neighbor.tif")

    compare_field_boundaries(bicubic, field_boundary)

if __name__ == "__main__":
    main()
