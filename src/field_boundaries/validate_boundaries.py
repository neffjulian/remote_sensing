import argparse
from pathlib import Path
from math import log10, sqrt

import cv2
from eodal.core.raster import RasterCollection
import numpy as np
import pandas as pd
from scipy.stats.mstats import spearmanr
from skimage.metrics import structural_similarity as ssim


VALIDATE_DIR = Path(__file__).parent.parent.parent.joinpath("data", "validate")

def psnr(x, y):
    return 20 * log10(8. / sqrt(np.mean((x - y) ** 2)))

def compare_field_boundaries(raster_path: Path, field_boundary_path: Path) -> None:
    """
    Uses a raster and neighborhood of a sharp gradient. First collects same neighborhood 
    from raster and compares it with the gradient neighborhood.

    """
    raster = RasterCollection.from_multi_band_raster(raster_path)["lai"]
    field_boundary = RasterCollection.from_multi_band_raster(field_boundary_path)["lai"]

    shape = field_boundary.values.shape
    # Never occured during testing but leaving for concistency
    if raster.geo_info.epsg != field_boundary.geo_info.epsg:
        print("EPSG codes do not match.", raster_path.name, field_boundary_path.name)
        return None, None, None, None
    # Do not use field boundaries that are too small
    elif shape[0] * shape[1] < 25:
        return None, None, None, None
    
    # Get difference in coordinates
    ulx_diff = field_boundary.geo_info.ulx - raster.geo_info.ulx
    uly_diff = raster.geo_info.uly - field_boundary.geo_info.uly
    
    # Scale SR data to have equal pixel resolution
    scale = raster.geo_info.pixres_x / field_boundary.geo_info.pixres_x
    y, x = raster.values.shape
    raster_shape = (int(x * scale), int(y * scale))
    raster_values = cv2.resize(raster.values, raster_shape, interpolation=cv2.INTER_CUBIC)
    raster_values = np.nan_to_num(raster_values)
    field_boundary_values = np.nan_to_num(field_boundary.values)

    # Get values at exact coordinates of field boundary
    x, y = int(uly_diff / 3), int(ulx_diff / 3)
    size_x, size_y = field_boundary.values.shape
    values = raster_values[x:x+size_x, y:y+size_y]

    # Calculate metrics
    correlation_coefficient, p_value = spearmanr(field_boundary_values.flatten(), values.flatten())
    psnr_value = psnr(field_boundary_values, values)
    if min(shape) < 7:
        ssim_value = None
    else:
        ssim_value, _ = ssim((field_boundary_values * (255. / 8.)).astype(np.uint8), (values * (255. / 8.)).astype(np.uint8), full=True)

    if not p_value >= 0:
        raise Exception("p_value is not positive", p_value)
    
    return correlation_coefficient, p_value, psnr_value, ssim_value
def validate_boundaries(ps_name: str, sr_name: str) -> None:
    ps_dir = VALIDATE_DIR.joinpath(ps_name)
    sr_dir = VALIDATE_DIR.joinpath(sr_name)

    for month in sr_dir.iterdir():
        print("Validating: ", month.name)
        results = []
        for index in month.iterdir():
            field_boundaries = ps_dir.joinpath(month.name, f"{index.name[:4]}_field_boundaries")
            for boundary in field_boundaries.glob("*.tif"):
                corr_coeff, p_value, psnr_value, ssim_value = compare_field_boundaries(index, boundary)
                results.append((index.name, boundary.name, corr_coeff, p_value, psnr_value, ssim_value))
        df = pd.DataFrame(results, columns=["index", "boundary", "corr_coeff", "p_value", "psnr", "ssim"])
        df.to_csv(month.joinpath(f"{ps_name}_results.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ps_name", type=str, required=True, help="Either '4b' or '8b'")
    parser.add_argument("--sr_name", type=str, required=True, help="Something like '10m_edsr' for 10m validation using edsr")
    args = parser.parse_args()
    validate_boundaries(args.ps_name, args.sr_name)