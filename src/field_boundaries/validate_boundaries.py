import argparse
from pathlib import Path
from math import log10, sqrt

import cv2
from eodal.core.raster import RasterCollection
import numpy as np
import pandas as pd
from scipy.stats.mstats import spearmanr
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot as plt

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

def create_dirs():
    VALIDATE_DIR.joinpath("10m_best").mkdir(parents=True, exist_ok=True)
    VALIDATE_DIR.joinpath("10m_worst").mkdir(parents=True, exist_ok=True)
    VALIDATE_DIR.joinpath("20m_best").mkdir(parents=True, exist_ok=True)
    VALIDATE_DIR.joinpath("20m_worst").mkdir(parents=True, exist_ok=True)

def compare(s2: Path, ps: Path, best_s2: bool, s2_res: str):
    best = "best" if best_s2 else "worst"
    out_dir = VALIDATE_DIR.joinpath(f"{s2_res}_{best}")
    out_name = out_dir.joinpath(s2.parent.name)
    
    sr_raster = RasterCollection.from_multi_band_raster(s2)["lai"]
    ps_raster = RasterCollection.from_multi_band_raster(ps)["lai"]

    ulx_diff = ps_raster.geo_info.ulx - sr_raster.geo_info.ulx
    uly_diff = sr_raster.geo_info.uly - ps_raster.geo_info.uly

    scale = sr_raster.geo_info.pixres_x / ps_raster.geo_info.pixres_x
    y, x = sr_raster.values.shape
    sr_shape = (int(x * scale), int(y * scale))
    sr_values = cv2.resize(sr_raster.values, sr_shape, interpolation=cv2.INTER_CUBIC)
    sr_values = np.nan_to_num(sr_values)
    ps_values = np.nan_to_num(ps_raster.values)

    x, y = int(uly_diff / 3), int(ulx_diff / 3)
    size_x, size_y = ps_raster.values.shape
    values = sr_values[x:x+size_x, y:y+size_y]

    if not values.shape == ps_values.shape:
        raise Exception("Shapes do not match", values.shape, ps_values.shape)
    
    correlation_coefficient, p_value = spearmanr(ps_values.flatten(), values.flatten())
    psnr_value = psnr(ps_values, values)
    ssim_value, _ = ssim((ps_values * (255. / 8.)).astype(np.uint8), (values * (255. / 8.)).astype(np.uint8), full=True)
    print(out_name.parent.name, correlation_coefficient, psnr_value, ssim_value)
    # Visualize sr and ps 

    out = np.hstack((values, ps_values))
    fig, ax = plt.subplots()
    ax.imshow(out, cmap="viridis")
    ax.axis("off")
    ax.set_facecolor('none')
    fig.set_facecolor('none')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(out_name)
    plt.close()

def print_selected_parcels():
    create_dirs()

    best_20m_name = "0002_101" # [:4], [5:]
    worst_20m_name = "0033_66" # 4 bands
    best_10m_name = "0012_104"
    worst_10m_name = "0001_206" # 8 bands

    months = ["03_mar", "04_apr", "05_may", "06_jun", "07_jul", "08_aug", "09_sep"]
    best_s2_20m = [VALIDATE_DIR.joinpath("20m_esrgan", month, f"{best_20m_name[:4]}.tif") for month in months]
    worst_s2_20m = [VALIDATE_DIR.joinpath("20m_esrgan", month, f"{worst_20m_name[:4]}.tif") for month in months]
    best_s2_10m = [VALIDATE_DIR.joinpath("10m_esrgan", month, f"{best_10m_name[:4]}.tif") for month in months]
    worst_s2_10m = [VALIDATE_DIR.joinpath("10m_esrgan", month, f"{worst_10m_name[:4]}.tif") for month in months]

    best_ps_4b = [VALIDATE_DIR.joinpath("4b", month, f"{best_20m_name[:4]}_field_boundaries", f"{best_20m_name[5:]}_reference_neighbor.tif") for month in months]
    worst_ps_4b = [VALIDATE_DIR.joinpath("4b", month, f"{worst_20m_name[:4]}_field_boundaries", f"{worst_20m_name[5:]}_reference_neighbor.tif") for month in months]
    best_ps_8b = [VALIDATE_DIR.joinpath("8b", month, f"{best_10m_name[:4]}_field_boundaries", f"{best_10m_name[5:]}_reference_neighbor.tif") for month in months]
    worst_ps_8b = [VALIDATE_DIR.joinpath("8b", month, f"{worst_10m_name[:4]}_field_boundaries", f"{worst_10m_name[5:]}_reference_neighbor.tif") for month in months]

    for i in range(7):
        print(i)
        compare(best_s2_20m[i], best_ps_4b[i], True, "20m")
        compare(worst_s2_20m[i], worst_ps_4b[i], False, "20m")
        compare(best_s2_10m[i], best_ps_8b[i], True, "10m")
        compare(worst_s2_10m[i], worst_ps_8b[i], False, "10m")

if __name__ == "__main__":
    print_selected_parcels()

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--ps_name", type=str, required=True, help="Either '4b' or '8b'")
    # parser.add_argument("--sr_name", type=str, required=True, help="Something like '10m_edsr' for 10m validation using edsr")
    # args = parser.parse_args()
    # validate_boundaries(args.ps_name, args.sr_name)