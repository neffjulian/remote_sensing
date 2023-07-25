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

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
RESULT_DIR = Path(__file__).parent.parent.parent.joinpath('data', 'results')
FILTER_DIR = Path(__file__).parent.parent.parent.joinpath('data', 'filtered')
FIELD_DATA = Path(__file__).parent.parent.parent.joinpath('data', 'coordinates', 'field_data.csv')

def psnr(x, y):
    return 20 * log10(8. / sqrt(np.mean((x - y) ** 2)))

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

def save_output_visualization(sentinel_2: np.ndarray, super_resolved: np.ndarray, planet_scope: np.ndarray, dir: Path, ps_downsampled: bool = False, error: bool = False):
    f, axes = plt.subplots(1, 3, figsize=(30, 10))

    # Plot for S2 image
    ax1 = axes[0]
    im1 = ax1.imshow(sentinel_2, cmap='viridis', vmin=0., vmax=8.)
    if ps_downsampled is True:
        ax1.set_title("Downsampled + Upsampled PlanetScope Image")
    else:
        ax1.set_title('Upsampled Sentinel-2 Image')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1)

    # Plot for SR image
    ax2 = axes[1]
    if error is True: 
        error = np.abs(planet_scope - super_resolved)
        im2 = ax2.imshow(error, cmap='Reds', vmin=0., vmax=1.)
        ax2.set_title('Error: Super-Resolved Image to original PlanetScope Image')
    else:
        im2 = ax2.imshow(super_resolved, cmap='viridis', vmin=0., vmax=8.)
        ax2.set_title('Super-Resolved Image')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2)

    # Plot for PlanetScope image
    ax3 = axes[2]
    im3 = ax3.imshow(planet_scope, cmap='viridis', vmin=0., vmax=8.)
    ax3.set_title('Original PlanetScope Image')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_aspect('equal')
    plt.colorbar(im3, ax=ax3, label=r'LAI [m$^2$ m$^{-2}$]')

    plt.tight_layout()
    if error is True:
        plt.savefig(dir.parent / ("error_ " + dir.name), dpi=300)
    else:
        plt.savefig(dir, dpi=300)
    plt.close(f)

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

def report_gpu():
   print(torch.cuda.list_gpu_processes())
   gc.collect()
   torch.cuda.empty_cache()

def get_in_situ():
    ps_in_situ = FILTER_DIR.joinpath("in_situ", "planetscope_in_situ", "2022", "03_mar", "lai")
    s2_in_situ = FILTER_DIR.joinpath("in_situ", "sentinel_in_situ", "2022", "03_mar", "lai")
    data = []

    for file in ps_in_situ.iterdir():
        if file.name.endswith(".tif") and "4bands" in file.name:
            s2_file_name = f"{file.name[:4]}_scene_20m_lai.tif"
            if s2_in_situ.joinpath(s2_file_name).exists():
                ps_raster = RasterCollection.from_multi_band_raster(file)
                s2_raster = RasterCollection.from_multi_band_raster(s2_in_situ.joinpath(s2_file_name))

                ps_file = np.nan_to_num(ps_raster["lai"].values)
                s2_file = np.nan_to_num(s2_raster["lai"].values)

                ps_file_interp = cv2.resize(ps_file, (150, 150), interpolation=cv2.INTER_AREA)
                s2_file_interp = cv2.resize(s2_file, (25, 25), interpolation=cv2.INTER_AREA)

                data.append((s2_file_interp, ps_file_interp, s2_raster, file.name[:4]))
    return data

def get_lai_pred(mat: np.ndarray) -> float:
    return (mat[74:74] + mat[74:75] + mat[75:75] + mat[75:74]) / 4

def visualize_in_situ(results: tuple, experiment_name: str) -> None:
    field_data = pd.read_csv(FIELD_DATA)
    lai = field_data["lai"].values
    lai_preds = []

    path = RESULT_DIR.joinpath(experiment_name)
    path.mkdir(parents=True, exist_ok=True)
    psnr_lr, psnr_sr = [], []
    ssim_lr, ssim_sr = [], []
    for lr, sr, hr, s2_raster, name in results:
        lr_interp = cv2.resize(lr, (150, 150), interpolation=cv2.INTER_CUBIC)
        save_output_visualization(lr_interp, sr, hr, path.joinpath(name + ".png"))

        geo_info = GeoInfo(
            epsg=s2_raster["lai"].geo_info.epsg,
            ulx=s2_raster["lai"].geo_info.ulx,
            uly=s2_raster["lai"].geo_info.uly,
            pixres_x=10/3,
            pixres_y=-10/3
        )

        index = int(name) - 1
        lai_lr = get_lai_pred(lr_interp)
        lai_sr = get_lai_pred(sr)
        lai_hr = get_lai_pred(hr)
        lai_preds.append((index, lai_lr, lai_sr, lai_hr, lai[index]))

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

    lr_error, sr_error, hr_error = [], [], []
    for index, lai_lr, lai_sr, lai_hr, lai in lai_preds:
        lr_error.append(abs(lai_lr - lai))
        sr_error.append(abs(lai_sr - lai))
        hr_error.append(abs(lai_hr - lai))
    lai.preds.append((1000, np.mean(lr_error), np.mean(sr_error), np.mean(hr_error), 1000))

    df = pd.DataFrame(lai_preds, columns=["index", "lr_error", "sr_error", "hr_error", "lai"])
    df.to_csv(path.joinpath("lai_preds.csv"), index=False)

def visualize_sample(lr_tiles: list, sr_tiles: list, hr_tiles: list, experiment_name: str, ps_downsampled: bool, raster: RasterCollection) -> None:
    path = RESULT_DIR.joinpath(experiment_name)
    path.mkdir(parents=True, exist_ok=True)
    file_name = "s2_0052" if ps_downsampled is False else "ps_0052"

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
    save_output_visualization(lr, sr, hr, path.joinpath(file_name), ps_downsampled, True)
    
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


def create_tiles(data: np.ndarray) -> list[np.ndarray]:
    x, y = data.shape
    x, y = x // 4, y // 4

    tiles = []
    for i in range(4):
        for j in range(4):
            tiles.append(data[i * x: (i + 1) * x, j * y: (j + 1) * y])
    return tiles

def get_sample():
    ps_fileloc = FILTER_DIR.joinpath("planetscope", "2022", "06_jun", "lai", "0052_lai_4bands.tif")
    s2_fileloc = FILTER_DIR.joinpath("sentinel", "2022", "06_jun", "lai", "0052_scene_20m_lai.tif")

    ps_file = np.nan_to_num(RasterCollection.from_multi_band_raster(ps_fileloc)["lai"].values)
    s2_file = np.nan_to_num(RasterCollection.from_multi_band_raster(s2_fileloc)["lai"].values)

    ps_file_interp = cv2.resize(ps_file, (600, 600), interpolation=cv2.INTER_AREA)
    s2_file_interp = cv2.resize(s2_file, (100, 100), interpolation=cv2.INTER_AREA)

    ps_tiles = create_tiles(ps_file_interp)
    s2_tiles = create_tiles(s2_file_interp)

    return s2_tiles, ps_tiles, RasterCollection.from_multi_band_raster(s2_fileloc)

def get_ps_sample():
    ps_fileloc = FILTER_DIR.joinpath("planetscope", "2022", "06_jun", "lai", "0052_lai_4bands.tif")
    ps_file = np.nan_to_num(RasterCollection.from_multi_band_raster(ps_fileloc)["lai"].values)
    ps_file_interp = cv2.resize(ps_file, (600, 600), interpolation=cv2.INTER_AREA)
    lr_file_interp = cv2.resize(ps_file, (100, 100), interpolation=cv2.INTER_AREA)
    ps_tiles = create_tiles(ps_file_interp)
    lr_tiles = create_tiles(lr_file_interp)
    return lr_tiles, ps_tiles, RasterCollection.from_multi_band_raster(ps_fileloc)

if __name__ == "__main__":
    # get_in_situ()
    # get_sample()
    report_gpu()