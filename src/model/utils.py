"""
Collection of utility functions used for visualization and evaluation.

Paper: https://arxiv.org/abs/2104.14951

@date: 2023-08-30
@author: Julian Neff, ETH Zurich

Copyright (C) 2023 Julian Neff

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import os
import gc
from pathlib import Path
from math import sqrt, log10
import cv2

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.pyplot as plt
from eodal.core.raster import RasterCollection
from eodal.core.band import Band, GeoInfo
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import r2_score
import geopandas as gpd
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

RESULT_DIR = Path(__file__).parent.parent.parent.joinpath('data', 'results')
FILTER_DIR = Path(__file__).parent.parent.parent.joinpath('data', 'filtered')
FIELD_DATA = Path(__file__).parent.parent.parent.joinpath('data', 'coordinates', 'field_data.csv')
IN_SITU = Path(__file__).parent.parent.parent.joinpath('data', 'coordinates', 'in-situ_glai.gpkg')

# Peak Signal to Noise Ratio
def psnr(x, y):
    return 20 * log10(8. / sqrt(np.mean((x - y) ** 2)))

# Transform model output into 4x4 grid
def transform_model_output(model_output: list, s2: bool) -> list[np.ndarray]:
    reconstructed_images = []
    name = model_output[0][3][:-2]
    if s2:
        name += "s2"
    else:
        name += "ps"

    image_s2 = np.zeros((480, 480))
    image_sr = np.zeros((480, 480))
    image_ps = np.zeros((480, 480))
    for j in range(4):
        for k in range(4):
            if j == 0:
                x_start = 0
                x_end = 150
            elif j == 1:
                x_start = 90
                x_end = 240
            elif j == 2:
                x_start = 240
                x_end = 390
            elif j == 3:
                x_start = 330
                x_end = 480

            if k == 0:
                y_start = 0
                y_end = 150
            elif k == 1:
                y_start = 90
                y_end = 240
            elif k == 2:
                y_start = 240
                y_end = 390
            elif k == 3:
                y_start = 330
                y_end = 480

            image_s2[x_start:x_end, y_start:y_end] = model_output[j*4 + k][0]
            image_sr[x_start:x_end, y_start:y_end] = model_output[j*4 + k][1]
            image_ps[x_start:x_end, y_start:y_end] = model_output[j*4 + k][2]
    
    reconstructed_images.append((image_s2, np.abs(image_ps - image_sr), image_ps, name + "_error"))
    reconstructed_images.append((image_s2, image_sr, image_ps, name))
    return reconstructed_images

# Save output visualization as png
def save_output_visualization(sentinel_2: np.ndarray, super_resolved: np.ndarray, planet_scope: np.ndarray, dir: Path, ps_downsampled: bool = False, error: bool = False):
    # Save Sentinel-2 Image
    TITLE_SIZE = 20
    AXIS_LABEL_SIZE = 18
    COLORBAR_LABEL_SIZE = 16
    TICK_SIZE = 15
    plt.rc('xtick', labelsize=TICK_SIZE)
    plt.rc('ytick', labelsize=TICK_SIZE)

    index = dir.name[:4]

    plt.figure(figsize=(10, 10))
    if ps_downsampled is True:
        plt.title("Downsampled + Upsampled PlanetScope Image", fontsize=TITLE_SIZE)
    else:
        plt.title('Upsampled Sentinel-2 Image', fontsize=TITLE_SIZE)
    plt.imshow(sentinel_2, cmap='viridis', vmin=0., vmax=8.)
    cbar = plt.colorbar(label=r'LAI [m$^2$ m$^{-2}$]')
    cbar.ax.set_ylabel(r'LAI [m$^2$ m$^{-2}$]', fontsize=COLORBAR_LABEL_SIZE)
    plt.xlabel('X', fontsize=AXIS_LABEL_SIZE)
    plt.ylabel('Y', fontsize=AXIS_LABEL_SIZE)
    plt.tight_layout(pad=1)
    plt.savefig(dir.parent.joinpath(index + "_s2.png"), dpi=300)
    plt.close()

    # Save Super-Resolved Image
    plt.figure(figsize=(10, 10))
    plt.imshow(super_resolved, cmap='viridis', vmin=0., vmax=8.)
    plt.title('Super-Resolved Image', fontsize=TITLE_SIZE)
    cbar = plt.colorbar(label=r'LAI [m$^2$ m$^{-2}$]')
    cbar.ax.set_ylabel(r'LAI [m$^2$ m$^{-2}$]', fontsize=COLORBAR_LABEL_SIZE)
    plt.xlabel('X', fontsize=AXIS_LABEL_SIZE)
    plt.ylabel('Y', fontsize=AXIS_LABEL_SIZE)
    plt.tight_layout(pad=1)
    plt.savefig(dir.parent.joinpath(index + "_sr.png"), dpi=300)
    plt.close()

    # Save PlanetScope Image
    plt.figure(figsize=(10, 10))
    plt.imshow(planet_scope, cmap='viridis', vmin=0., vmax=8.)
    plt.title('Original PlanetScope Image', fontsize=TITLE_SIZE)
    cbar = plt.colorbar(label=r'LAI [m$^2$ m$^{-2}$]')
    cbar.ax.set_ylabel(r'LAI [m$^2$ m$^{-2}$]', fontsize=COLORBAR_LABEL_SIZE)
    plt.xlabel('X', fontsize=AXIS_LABEL_SIZE)
    plt.ylabel('Y', fontsize=AXIS_LABEL_SIZE)
    plt.tight_layout(pad=1)
    plt.savefig(dir.parent.joinpath(index + "_ps.png"), dpi=300)
    plt.close()

    # Save Error Image
    plt.figure(figsize=(10, 10))
    error_image = np.abs(planet_scope - super_resolved)
    plt.imshow(error_image, cmap='Reds', vmin=0., vmax=4.)
    plt.title('Error: Super-Resolved to original PlanetScope Image', fontsize=TITLE_SIZE)
    cbar = plt.colorbar(label=r'L1 Error')
    cbar.ax.set_ylabel(r'L1 Error', fontsize=COLORBAR_LABEL_SIZE)
    plt.xlabel('X', fontsize=AXIS_LABEL_SIZE)
    plt.ylabel('Y', fontsize=AXIS_LABEL_SIZE)
    plt.tight_layout(pad=1)
    plt.savefig(dir.parent.joinpath(index + "_er.png"), dpi=300)
    plt.close()

# Visualize model output
def visualize_output(name: str, output: list) -> None:
    lr, sr, hr, names = [], [], [], []
    for out in output:
        lr += out[0]
        sr += out[1]
        hr += out[2]
        names += out[3]
    outputs = []
    for i in range(len(names)):
        outputs.append(
            (torch.squeeze(lr[i]).numpy(), 
             torch.squeeze(sr[i]).numpy(), 
             torch.squeeze(hr[i]).numpy(), 
             names[i][:-4]))
    in_situ = sorted(outputs[:-32], key=lambda x: x[3])
    ps_sr = sorted(outputs[-32:-16], key=lambda x: x[3])
    s2_sr = sorted(outputs[-16:], key=lambda x: x[3])

    transformer_ps_sr = transform_model_output(ps_sr, False)
    transformer_s2_sr = transform_model_output(s2_sr, True)

    transformed_output = in_situ + transformer_ps_sr + transformer_s2_sr

    results = RESULT_DIR.joinpath(name)
    results.mkdir(parents=True, exist_ok=True)
    lr_hr_psnrs = []
    sr_hr_psnrs = []
    lr_hr_ssims = []
    sr_hr_ssims = []
    j = 0
    for i, out in enumerate(transformed_output):
        out_file = results.joinpath(out[3] + '.png')
        lr_hr_psnr = psnr(out[0], out[2])
        sr_hr_psnr = psnr(out[1], out[2])
        lr_hr_ssim, _ = ssim((out[0] * (255.0 / 8.0)).astype(np.uint8), (out[2] * (255.0 / 8.0)).astype(np.uint8), full=True)
        sr_hr_ssim, _ = ssim((out[1] * (255.0 / 8.0)).astype(np.uint8), (out[2] * (255.0 / 8.0)).astype(np.uint8), full=True)
        print(out_file.name, "LR-HR PSNR:", lr_hr_psnr, "  SR-HR PSNR:", sr_hr_psnr," |  LR-HR SSIM:", lr_hr_ssim, " SR-HR SSIM:", sr_hr_ssim)
        j += 1
        if len(out[3]) == 4:
            if (sr_hr_ssim > 0.5 or sr_hr_psnr > 25) and j < 5:
                save_output_visualization(out[0], out[1], out[2], out_file)
            np.save(results.joinpath(out[3] + '.npy'), out[1])
            lr_hr_psnrs.append(lr_hr_psnr)
            sr_hr_psnrs.append(sr_hr_psnr)
            lr_hr_ssims.append(lr_hr_ssim)
            sr_hr_ssims.append(sr_hr_ssim)
        else:
            save_output_visualization(out[0], out[1], out[2], out_file)
            np.save(results.joinpath(out[3] + '.npy'), out[1])

    print("--------------------- MEAN ------------------------")
    lh_psnr = np.mean(lr_hr_psnrs)
    sh_psnr = np.mean(sr_hr_psnrs)
    lh_ssim = np.mean(lr_hr_ssims)
    sh_ssim = np.mean(sr_hr_ssims)
    change_psnr = round(sh_psnr / lh_psnr * 100 - 100, 2)
    change_ssim = round(sh_ssim / lh_ssim * 100 - 100, 2)
    print("LR-HR PSNR:", lh_psnr, " SR-HR PSNR:", sh_psnr, " (", change_psnr,"%) |", \
          "LR-HR SSIM:", lh_ssim, " SR-HR SSIM:", sh_ssim, " (", change_ssim,"%)")
    print("-------------------- MEDIAN -----------------------")
    lh_psnr = np.median(lr_hr_psnrs)
    sh_psnr = np.median(sr_hr_psnrs)
    lh_ssim = np.median(lr_hr_ssims)
    sh_ssim = np.median(sr_hr_ssims)
    change_psnr = round(sh_psnr / lh_psnr * 100 - 100, 2)
    change_ssim = round(sh_ssim / lh_ssim * 100 - 100, 2)
    print("LR-HR PSNR:", lh_psnr, " SR-HR PSNR:", sh_psnr, " (", change_psnr,"%) |", \
            "LR-HR SSIM:", lh_ssim, " SR-HR SSIM:", sh_ssim, " (", change_ssim,"%)")

# Function used before training
def report_gpu():
   print(torch.cuda.list_gpu_processes())
   gc.collect()
   torch.cuda.empty_cache()

# Collect in-situ data
def get_in_situ(ps_bands: str, s2_bands: str):
    ps_in_situ = FILTER_DIR.joinpath("in_situ", "planetscope_in_situ", "2022", "03_mar", "lai")
    s2_in_situ = FILTER_DIR.joinpath("in_situ", "sentinel_in_situ", "2022", "03_mar", "lai")
    data = []

    for file in ps_in_situ.iterdir():
        if file.name.endswith(".tif") and f"{ps_bands}ands" in file.name:
            s2_file_name = f"{file.name[:4]}_scene_{s2_bands}_lai.tif"
            if s2_in_situ.joinpath(s2_file_name).exists():
                ps_raster = RasterCollection.from_multi_band_raster(file)
                s2_raster = RasterCollection.from_multi_band_raster(s2_in_situ.joinpath(s2_file_name))

                ps_file = np.nan_to_num(ps_raster["lai"].values)
                s2_file = np.nan_to_num(s2_raster["lai"].values)

                ps_file_interp = cv2.resize(ps_file, (150, 150), interpolation=cv2.INTER_AREA)
                s2_file_interp = cv2.resize(s2_file, (25, 25), interpolation=cv2.INTER_AREA)

                data.append((s2_file_interp, ps_file_interp, s2_raster, file.name[:4]))
    return data

# Given image returns the pixel value at the center pixel
def get_lai_pred(mat: np.ndarray):
    if mat.shape == (75, 75):
        return mat[37,37]
    elif mat.shape == (25, 25):
        return mat[12,12]
    elif mat.shape == (150, 150):
        return (mat[74,74] + mat[74,75] + mat[75,75] + mat[75,74]) / 4
    else:
        raise Exception("Invalid shape: ", mat.shape)

# Visualize in-situ data
def visualize_in_situ(results: tuple, experiment_name: str) -> None:
    field_data = pd.read_csv(FIELD_DATA)
    lai = field_data["lai"].values
    dates = field_data["date"].values
    lai_preds = []

    path = RESULT_DIR.joinpath(experiment_name)
    path.mkdir(parents=True, exist_ok=True)
    psnr_lr, psnr_sr = [], []
    ssim_lr, ssim_sr = [], []
    for lr, sr, hr, s2_raster, name in results:
        lr_interp = cv2.resize(lr, (150, 150), interpolation=cv2.INTER_CUBIC)
        save_output_visualization(lr_interp, sr, hr, path.joinpath(name + ".png"))

        index = int(name)
        lai_preds.append((index, get_lai_pred(lr), get_lai_pred(sr), get_lai_pred(hr), lai[index], dates[index]))

        geo_info = GeoInfo(
            epsg=s2_raster["lai"].geo_info.epsg,
            ulx=s2_raster["lai"].geo_info.ulx,
            uly=s2_raster["lai"].geo_info.uly,
            pixres_x=10/3,
            pixres_y=-10/3
        )

        x, y = s2_raster["lai"].values.shape
        raster = RasterCollection(
            band_constructor=Band,
            band_name="lai",
            values = cv2.resize(sr,(y * 6, x * 6), interpolation=cv2.INTER_CUBIC),
            geo_info = geo_info
        )

        raster.to_rasterio(path.joinpath(name + ".tif"))

        psnr_lr.append(psnr(lr_interp, hr))
        psnr_sr.append(psnr(sr, hr))
        ssim_lr.append(ssim((lr_interp * (255.0 / 8.0)).astype(np.uint8), (hr * (255.0 / 8.0)).astype(np.uint8), full=True)[0])
        ssim_sr.append(ssim((sr * (255.0 / 8.0)).astype(np.uint8), (hr * (255.0 / 8.0)).astype(np.uint8), full=True)[0])

    print("--------------------- MEAN ------------------------")
    print("LR-HR PSNR:", np.mean(psnr_lr), " SR-HR PSNR:", np.mean(psnr_sr), " |  LR-HR SSIM:", np.mean(ssim_lr), " SR-HR SSIM:", np.mean(ssim_sr))
    print("-------------------- MEDIAN -----------------------")
    print("LR-HR PSNR:", np.median(psnr_lr), " SR-HR PSNR:", np.median(psnr_sr), " |  LR-HR SSIM:", np.median(ssim_lr), " SR-HR SSIM:", np.median(ssim_sr))

    df = pd.DataFrame(lai_preds, columns=["index", "s2_lai", "sr_lai", "hr_lai", "in_situ_lai", "date"])
    csv_path = path.joinpath("lai_preds.csv")
    df.to_csv(csv_path, index=False)
    # calculate_lai_error(csv_path)

# Visualizes a sample
def visualize_sample(lr_tiles: list, sr_tiles: list, hr_tiles: list, experiment_name: str, ps_downsampled: bool, raster: RasterCollection) -> None:
    path = RESULT_DIR.joinpath(experiment_name)
    path.mkdir(parents=True, exist_ok=True)
    file_name = "s2_0023" if ps_downsampled is False else "ps_0023"

    lr = np.zeros((600, 600))
    sr = np.zeros((600, 600))
    hr = np.zeros((600, 600))
    x, y = 150, 150
    for i in range(len(lr_tiles)):
        j = i // 4
        k = i % 4
        lr[j * x: (j + 1) * x, k * y: (k + 1) * y] = cv2.resize(lr_tiles[i], (150, 150), interpolation=cv2.INTER_CUBIC)
        sr[j * x: (j + 1) * x, k * y: (k + 1) * y] = sr_tiles[i]
        hr[j * x: (j + 1) * x, k * y: (k + 1) * y] = hr_tiles[i]
    save_output_visualization(lr, sr, hr, path.joinpath(file_name), ps_downsampled)
    save_output_visualization(lr, sr, hr, path.joinpath("error_" + file_name), ps_downsampled, True)
    
    geo_info = GeoInfo(
        epsg=raster["lai"].geo_info.epsg,
        ulx=raster["lai"].geo_info.ulx,
        uly=raster["lai"].geo_info.uly,
        pixres_x=10/3 if ps_downsampled is False else 3,
        pixres_y=-10/3 if ps_downsampled is False else -3
    )
    x, y = raster["lai"].values.shape
    if ps_downsampled is False:
        x *= 6
        y *= 6

    raster = RasterCollection(
        band_constructor=Band,
        band_name="lai",
        values = cv2.resize(sr,(y, x), interpolation=cv2.INTER_CUBIC),
        geo_info = geo_info
    )
    raster.to_rasterio(path.joinpath(file_name + ".tif"))

# Calculates the LAI error given a csv file with predictions
def calculate_lai_error(csv_path: Path):
    df = pd.read_csv(csv_path, index_col=0)
    data = df.to_numpy()
    lai_lr = data[:, 0]
    lai_sr = data[:, 1]
    lai_hr = data[:, 2]
    lai_in_situ = data[:, 3]

    diff_lr = np.abs(lai_lr - lai_in_situ)
    diff_sr = np.abs(lai_sr - lai_in_situ)
    diff_hr = np.abs(lai_hr - lai_in_situ)

    l1_lr = np.mean(diff_lr)
    l1_sr = np.mean(diff_sr)
    l1_hr = np.mean(diff_hr)

    sq_lr = np.square(diff_lr)
    sq_sr = np.square(diff_sr)
    sq_hr = np.square(diff_hr)

    mean_lr = np.mean(sq_lr)
    mean_sr = np.mean(sq_sr)
    mean_hr = np.mean(sq_hr)

    root_lr = np.sqrt(mean_lr)
    root_sr = np.sqrt(mean_sr)
    root_hr = np.sqrt(mean_hr)

    r2_lr = r2_score(lai_in_situ, lai_lr)
    r2_sr = r2_score(lai_in_situ, lai_sr)
    r2_hr = r2_score(lai_in_situ, lai_hr)

    print("L1:", "LR:", l1_lr, "SR:", l1_sr, "HR:", l1_hr)
    print("RMSE:", "LR:", root_lr, "SR:", root_sr, "HR:", root_hr)
    print("R2:", "LR:", r2_lr, "SR:", r2_sr, "HR:", r2_hr)

# Creates 4x4 grid from image
def create_tiles(data: np.ndarray) -> list[np.ndarray]:
    x, y = data.shape
    x, y = x // 4, y // 4

    tiles = []
    for i in range(4):
        for j in range(4):
            tiles.append(data[i * x: (i + 1) * x, j * y: (j + 1) * y])
    return tiles

# Gets a sample from the dataset
def get_sample(ps_bands: str, s2_bands: str):
    ps_fileloc = FILTER_DIR.joinpath("planetscope", "2022", "06_jun", "lai", f"0023_lai_{ps_bands}ands.tif")
    s2_fileloc = FILTER_DIR.joinpath("sentinel", "2022", "06_jun", "lai", f"0023_scene_{s2_bands}_lai.tif")

    ps_file = np.nan_to_num(RasterCollection.from_multi_band_raster(ps_fileloc)["lai"].values)
    s2_file = np.nan_to_num(RasterCollection.from_multi_band_raster(s2_fileloc)["lai"].values)

    ps_file_interp = cv2.resize(ps_file, (600, 600), interpolation=cv2.INTER_AREA)
    s2_file_interp = cv2.resize(s2_file, (100, 100), interpolation=cv2.INTER_AREA)

    ps_tiles = create_tiles(ps_file_interp)
    s2_tiles = create_tiles(s2_file_interp)

    return s2_tiles, ps_tiles, RasterCollection.from_multi_band_raster(s2_fileloc)

# Gets a PlanetScope sample from the dataset
def get_ps_sample(ps_bands: str):
    ps_fileloc = FILTER_DIR.joinpath("planetscope", "2022", "06_jun", "lai", f"0023_lai_{ps_bands}ands.tif")
    ps_file = np.nan_to_num(RasterCollection.from_multi_band_raster(ps_fileloc)["lai"].values)
    ps_file_interp = cv2.resize(ps_file, (600, 600), interpolation=cv2.INTER_AREA)
    lr_file_interp = cv2.resize(ps_file, (100, 100), interpolation=cv2.INTER_AREA)
    ps_tiles = create_tiles(ps_file_interp)
    lr_tiles = create_tiles(lr_file_interp)
    return lr_tiles, ps_tiles, RasterCollection.from_multi_band_raster(ps_fileloc)

# Makes a plot from given CSV
def make_plot_from_csv(lai_preds: np.ndarray):
    preds = lai_preds[:-1]
    dates = [datetime.strptime(date, "%Y-%m-%d %H:%M:%S") for date in preds['date']]
    preds = sorted(list(zip(dates, preds['s2_lai'], preds['sr_lai'], preds['hr_lai'], preds['in_situ_lai'])), key=lambda x: x[0])

    dates = [pred[0] for pred in preds]
    s2_lai = [pred[1] for pred in preds]
    sr_lai = [pred[2] for pred in preds]
    hr_lai = [pred[3] for pred in preds]
    in_situ_lai = [pred[4] for pred in preds]

    fig, ax = plt.subplots(figsize=(10, 6))
    # Scatter
    ax.plot(dates, s2_lai, label="Sentinel-2", marker='o', linestyle='-')
    ax.plot(dates, sr_lai, label="Super-Resolved", marker='o', linestyle='-')
    ax.plot(dates, hr_lai, label="PlanetScope")
    ax.plot(dates, in_situ_lai, label="In-Situ")

    ax.set_xlabel("Date")
    ax.set_ylabel(r"LAI [m$^2$ m$^{-2}$]")
    ax.set_title("LAI Prediction")
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # get_in_situ()
    # get_sample()
    # report_gpu()

    csv = RESULT_DIR.joinpath("RRDB_2023_07_27_08_56", "lai_preds.csv")
    df = pd.read_csv(csv)
    npy = df.to_numpy()
    print(npy.shape)
    make_plot_from_csv(df[:-1])
