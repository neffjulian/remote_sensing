from pathlib import Path
from math import log10, sqrt

import cv2
from eodal.core.raster import RasterCollection
import numpy as np
from scipy.stats import spearmanr


VALIDATE_DIR = Path(__file__).parent.parent.joinpath("data", "validate")

def psnr(x, y):
    return 20 * log10(8. / sqrt(np.mean((x - y) ** 2)))

def compare_field_boundaries(raster_path: Path, field_boundary_path: Path) -> None:
    raster = RasterCollection.from_multi_band_raster(raster_path)["lai"]
    field_boundary = RasterCollection.from_multi_band_raster(field_boundary_path)["lai"]

    shape = field_boundary.values.shape
    if raster.geo_info.epsg != field_boundary.geo_info.epsg:
        print("EPSG codes do not match.", raster_path.name, field_boundary_path.name)
        return None
    elif shape[0] * shape[1] < 25 or min(raster.values.shape) < 100:
        return None
    
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

    # psnr_index = psnr(field_boundary.values, values)
    # print(psnr_index)
    if type(correlation_coefficient) != np.float64:
        return None
    return correlation_coefficient, p_value

def main() -> None:
    # bicubic = VALIDATE_DIR.joinpath("20m_bicubic", "03_mar", "0000.tif")
    # field_boundary = VALIDATE_DIR.joinpath("4b", "03_mar", "0000_field_boundaries", "17_reference_neighbor.tif")

    # compare_field_boundaries(bicubic, field_boundary)

    ps_dir = VALIDATE_DIR.joinpath("4b")
    sr_dir = VALIDATE_DIR.joinpath("20m_srcnn")
    correlations = []
    for month in sr_dir.iterdir():
        print(month.name)
        month_correlations = []
        for index in month.iterdir():
            if index.name[:4] == "0030":
                continue
            field_boundaries = ps_dir.joinpath(month.name, f"{index.name[:4]}_field_boundaries")
            for boundary in field_boundaries.glob("*.tif"):
                out = compare_field_boundaries(index, boundary)
                if out:
                    month_correlations.append(out)

        correlations.append(month_correlations)
    for correlation in correlations:
        print(len(correlation))
        print(sum([x[0] for x in correlation]) / len(correlation))
        print(sum([x[1] for x in correlation]) / len(correlation))        


if __name__ == "__main__":
    main()
