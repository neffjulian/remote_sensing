import os
import argparse
from pathlib import Path
from math import log10, sqrt

import numpy as np
import cv2
from eodal.core.raster import RasterCollection
from skimage.exposure import match_histograms

from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim

DATA_DIR = Path(__file__).parent.parent.joinpath("data")
FOLDERS = {"4b": 3, "8b": 3, "10m": 10, "20m": 20}

def get_filenames(foldername: str, in_situ: bool) -> list[str]:
    """Get a list of filenames in the specified folder."""
    if in_situ is True:
        dir = DATA_DIR.joinpath("processed", f"{foldername}_in_situ")
    else:
        dir = DATA_DIR.joinpath("processed", foldername)

    return [file.name for file in dir.iterdir()]

def remove_files(foldername: str, files_to_keep: list[str], in_situ: bool) -> None:
    """Remove files from the folder that are not in the files_to_keep list."""
    if in_situ is True:
        dir = DATA_DIR.joinpath("processed", f"{foldername}_in_situ")
    else:
        dir = DATA_DIR.joinpath("processed", foldername)

    files_to_remove = [file for file in dir.iterdir() if not file.name in files_to_keep]

    for file in files_to_remove:
        file.unlink()

def remove_unused_images(planetscope_bands: str, sentinel_bands: str, in_situ: bool) -> None:
    """Remove unused images from the processed folders."""
    folders = [planetscope_bands, sentinel_bands]
    files = [get_filenames(folder, in_situ) for folder in folders]
    files_to_keep = list(set.intersection(*map(set, files)))
    print(f"Equal number of files in both folders: {len(files_to_keep)}")

    for folder in folders:
        remove_files(folder, files_to_keep, in_situ)

    if in_situ is True:
        files = [[file.name for file in DATA_DIR.joinpath("processed", f"{folder}_in_situ").iterdir()] for folder in folders]
    else:
        files = [[file.name for file in DATA_DIR.joinpath("processed", folder).iterdir()] for folder in folders]
    assert all(files[0] == files[i] for i in range(1, len(files))), "Error in original Sentinel-2 Files"

def create_tiles(data: np.ndarray, satellite: str, tiles: int = 4):
    """Create tiles from the given data."""
    sub_data = []
    arr_x, arr_y = data.shape
    x, y = arr_x // tiles, arr_y // tiles
    
    if satellite == "sentinel":
        margin = 10
    else: 
        margin = 60

    for i in range(tiles):
        for j in range(tiles):
            x_start, x_end = i * x, (i + 1) * x
            y_start, y_end = j * y, (j + 1) * y

            if i == 0:
                x_start += margin
                x_end += margin
            if i == tiles - 1:
                x_start -= margin
                x_end -= margin
            if j == 0:
                y_start += margin
                y_end += margin
            if j == tiles - 1:
                y_start -= margin
                y_end -= margin

            tile = data[x_start:x_end, y_start:y_end]
            if np.isnan(tile).sum() / tile.size > 0.01:
                sub_data.append(None)
            else:
                sub_data.append(np.clip(np.nan_to_num(tile), 0., 8.))
    return sub_data

def rename_in_situ_data(band: str):
    index_to_field = {'0000': 'broatefaeld', '0001': 'bramenwies', '0002': 'fluegenrain', '0003': 'hohrueti', 
                      '0004': 'altkloster', '0005': 'ruetteli', '0006': 'witzwil', '0007': 'strickhof', '0008': 'eschikon'}
    in_situ_dir = DATA_DIR.joinpath("processed", f"{band}_in_situ")
    for root, _, files in os.walk(in_situ_dir):
        for file in files:
            path = Path(root).joinpath(file)
            if file[3:7].isdigit():
                new_name = file.replace(file[3:7], index_to_field[file[3:7]])
                path.rename(Path(root).joinpath(new_name))

def psnr(x, y):
    return 20 * log10(8. / sqrt(np.mean((x - y) ** 2)))

def process_file(satellite: str, target_dir: Path, target_name: str, data: np.ndarray, in_situ: bool, augment: bool = False) -> None:
    if in_situ is True: # In situ data covers an area of 500x500m 
        out_dim = (25, 25) if satellite == "sentinel" else (150, 150)
    else:
        out_dim = (100, 100) if satellite == "sentinel" else (600, 600)

    resized_data = cv2.resize(data, out_dim, interpolation=cv2.INTER_AREA)

    if in_situ is False:
        tiles = create_tiles(resized_data, satellite)
        for i, tile in enumerate(tiles):
            if tile is None:
                continue

            if augment is True:
                for j in range(4):
                    tile_rot = np.rot90(tile, j)
                    np.save(target_dir.joinpath(f"{target_name}_{i:02d}_{j:02}_00.npy"), tile_rot)
                    np.save(target_dir.joinpath(f"{target_name}_{i:02d}_{j:02}_01.npy"), np.flip(tile_rot, axis=1))
            else:
                np.save(target_dir.joinpath(f"{target_name}_{i:02d}.npy"), tile)
    else:
        np.save(target_dir.joinpath(f"{target_name[3:]}.npy"), np.nan_to_num(resized_data))

def process_satellite_data(satellite: str, band: str, in_situ: bool) -> None:
    months = {"jan": "01", "feb": "02", "mar": "03", "apr": "04", 
              "may": "05", "jun": "06", "jul": "07", "aug": "08", 
              "sep": "09", "oct": "10", "nov": "11", "dec": "12"}

    if in_situ is True:
        folder = DATA_DIR.joinpath("filtered", "in_situ", f"{satellite}_in_situ")
        min_shape = 25 if satellite == "sentinel" else 150
    else:
        folder = DATA_DIR.joinpath("filtered", satellite)
        min_shape = 100 if satellite == "sentinel" else 660

    
    for year in folder.iterdir():
        for month in year.iterdir():
            if not month.name[-3:] in months:
                continue
            
            source_dir = month.joinpath("lai")
            if in_situ is True:
                print(f"Preprocess in situ data for {satellite}")
                target_dir = DATA_DIR.joinpath("processed", f"{band}_in_situ")
            else:
                print(f"Preprocess {satellite} data for {year.name} {month.name}")

                target_dir = DATA_DIR.joinpath("processed", band)
            target_dir.mkdir(parents=True, exist_ok=True)

            file_ending = band
            file_ending += "ands.tif" if satellite == "planetscope" else "_lai.tif"
            for file in source_dir.iterdir():
                if file.name.endswith(file_ending):
                    data = RasterCollection.from_multi_band_raster(file)["lai"].values
                    if min(data.shape) < min_shape:
                        continue
                    target_name = month.name[0:2] + "_" + file.name[0:4]
                    process_file(satellite, target_dir, target_name, data, in_situ)

def remove_outliers(ps_bands: str, s2_bands: str) -> None:
    ps_folder = DATA_DIR.joinpath("processed", ps_bands)
    s2_folder = DATA_DIR.joinpath("processed", s2_bands)

    files_to_keep = []

    print(f"Number of files in {ps_folder.name}: {len(list(ps_folder.iterdir()))}")
    print(f"Number of files in {s2_folder.name}: {len(list(s2_folder.iterdir()))}")

    ps_psnr_ = []
    ps_ssim_ = []

    s2_psnr_ = []
    s2_ssim_ = []

    for ps_filename in ps_folder.iterdir():
        ps_file = np.load(ps_filename)
        if s2_folder.joinpath(ps_filename.name).exists():
            s2_file = np.load(s2_folder.joinpath(ps_filename.name))
        else:
            continue

        downsampled_file = cv2.resize(ps_file, (25, 25), interpolation=cv2.INTER_AREA)
        upsampled_file = cv2.resize(downsampled_file, (150, 150), interpolation=cv2.INTER_CUBIC)

        ps_psnr = psnr(upsampled_file, ps_file)
        ps_ssim, _ = ssim((upsampled_file * (255. / 8.)).astype(np.uint8), (ps_file * (255. / 8.)).astype(np.uint8), full=True)

        s2_psnr = psnr(downsampled_file, s2_file)
        s2_ssim, _ = ssim((downsampled_file * (255. / 8.)).astype(np.uint8), (s2_file * (255. / 8.)).astype(np.uint8), full=True)

        if ps_psnr <= 0.25 or s2_psnr <= 0.1:
            continue
        elif ps_ssim <= 0.75 or s2_ssim <= 0.1:
            continue

        files_to_keep.append(ps_filename.name)

        ps_psnr_.append(ps_psnr)
        ps_ssim_.append(ps_ssim)
        s2_psnr_.append(s2_psnr)
        s2_ssim_.append(s2_ssim)

    print(f"Number of files to keep: {len(files_to_keep)}")

    remove_files(ps_folder, files_to_keep, False)
    remove_files(s2_folder, files_to_keep, False)

    assert [file.name for file in ps_folder.iterdir()] == [file.name for file in s2_folder.iterdir()]

    print(f"Number of files in {ps_folder.name}: {len(list(ps_folder.iterdir()))}")
    print(f"Number of files in {s2_folder.name}: {len(list(s2_folder.iterdir()))}")

    # overlap = np.intersect1d(ps_psnr_, s2_psnr_)
    # plt.hist(s2_psnr_, bins=100, range=(0,100), label='S2', color='red', alpha=0.5)
    # plt.hist(ps_psnr_, bins=100, range=(0,100), label='PS', color='blue', alpha=0.5)
    # plt.hist(overlap, bins=100, range=(0,100), label='Overlap', color='purple', )
    # plt.legend()
    # plt.xlabel('PSNR')
    # plt.ylabel('Frequency')
    # plt.title(f'Error: Downsampled PS vs S2 and Reupsampled PS vs PS')
    # plt.show()

    # overlap = np.intersect1d(ps_ssim_, s2_ssim_)
    # plt.hist(s2_ssim_, bins=100, range=(0,1), label='S2', color='red', alpha=0.5)
    # plt.hist(ps_ssim_, bins=100, range=(0,1), label='PS', color='blue', alpha=0.5)
    # plt.hist(overlap, bins=100, range=(0,1), label='Overlap', color='purple', alpha=0.7)    
    # plt.legend()
    # plt.xlabel('SSIM')
    # plt.ylabel('Frequency')
    # plt.title(f'Error: Downsampled PS vs S2 and Reupsampled PS vs PS')
    # plt.show()

    # plt.hist(s2_psnr, bins=100, range=(0,100), label='Overlap')
    # plt.legend()
    # plt.xlabel('PSNR')
    # plt.ylabel('Frequency')
    # plt.title(f'Comparison: Downsampled PS vs. Sentinel-2')
    # plt.show()

    # plt.hist(s2_ssim, bins=100, range=(0,1), label='Overlap')
    # plt.legend()
    # plt.xlabel('SSIM')
    # plt.ylabel('Frequency')
    # plt.title(f'Comparison: Downsampled PS vs. Sentinel-2')
    # plt.show()

def create_lr_dataset(ps_band: str):
    source_dir = DATA_DIR.joinpath("processed", ps_band)
    target_dir = DATA_DIR.joinpath("processed", ps_band + "_lr")
    target_dir.mkdir(parents=True, exist_ok=True)

    psnr_scores = []
    ssim_scores = []

    for file in source_dir.iterdir():
        data = np.load(file)
        downsampled_data = cv2.resize(data, (25, 25), interpolation=cv2.INTER_AREA)
        np.save(target_dir.joinpath(file.name), downsampled_data)

        upsampled_data = cv2.resize(downsampled_data, (150, 150), interpolation=cv2.INTER_CUBIC)

        psnr_scores.append(psnr(upsampled_data, data))
        ssim_score, _ = ssim((upsampled_data * (255. / 8.)).astype(np.uint8), (data * (255. / 8.)).astype(np.uint8), full=True)
        ssim_scores.append(ssim_score)

    print("Scores when downsampling PS to 25x25 and upsampling back to 150x150:")
    print(f"PSNR: {np.mean(psnr_scores)}")
    print(f"SSIM: {np.mean(ssim_scores)}")

def main(planetscope_bands: str, sentinel_bands: str):
    process_satellite_data("sentinel", sentinel_bands, False)
    process_satellite_data("planetscope", planetscope_bands, False)
    remove_unused_images(planetscope_bands, sentinel_bands, in_situ=False)
    remove_outliers(planetscope_bands, sentinel_bands)
    create_lr_dataset(planetscope_bands)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentinel_bands", type=str, required=True, help="Either '10m' or '20m'")
    parser.add_argument("--planetscope_bands", type=str, required=True, help="Either '4b' or '8b'")
    args = parser.parse_args()

    main(args.planetscope_bands, args.sentinel_bands)