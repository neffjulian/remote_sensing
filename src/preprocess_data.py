import os
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
        dir = DATA_DIR.joinpath("processed", "in_situ", foldername)
    else:
        dir = DATA_DIR.joinpath("processed", foldername)

    return [file.name for file in dir.iterdir()]

def remove_files(foldername: str, files_to_keep: list[str], in_situ: bool) -> None:
    """Remove files from the folder that are not in the files_to_keep list."""
    if in_situ is True:
        dir = DATA_DIR.joinpath("processed", "in_situ", foldername)
    else:
        dir = DATA_DIR.joinpath("processed", foldername)

    for file in dir.iterdir():
        if not file.name in files_to_keep:
            file.unlink()

def remove_unused_images(in_situ: bool = False) -> None:
    """Remove unused images from the processed folders."""
    files = [get_filenames(folder, in_situ) for folder in FOLDERS.keys()]
    files_to_keep = list(set.intersection(*map(set, files)))

    for folder in FOLDERS.keys():
        remove_files(folder, files_to_keep, in_situ)

    if in_situ is True:
        files = [[file.name for file in DATA_DIR.joinpath("processed", "in_situ", folder).iterdir()] for folder in FOLDERS.keys()]
    else:
        files = [[file.name for file in DATA_DIR.joinpath("processed", folder).iterdir()] for folder in FOLDERS.keys()]
    
    assert all(files[0] == files[i] for i in range(1, len(files))), "Error in original Sentinel-2 Files"

def preprocess_file(src_data: Path, tar_dir: Path, in_situ: bool, rotate: bool = False) -> None:
    """Preprocess a single file and save the tiles."""
    file_index = src_data.name[:4]
    month = src_data.parent.parent.name[:2]
    year = src_data.parent.parent.parent.name
    satellite = src_data.parent.parent.parent.parent.name

    raster = RasterCollection.from_multi_band_raster(src_data)
    if satellite.startswith("planetscope"):
        OUT_DIM = (640, 640)
    elif satellite.startswith("sentinel"):
        OUT_DIM = (160, 160)
    else:
        raise Exception(f"Invalid Satellite encountered: ", satellite)
    lai = cv2.resize(raster["lai"].values, OUT_DIM, interpolation=cv2.INTER_CUBIC)

    if np.isnan(lai).sum() / lai.size > 0.1:
        return
    
    tiles = create_tiles(lai, in_situ)
    for tile_index, tile in enumerate(tiles):
        if tile is None:
            continue # The tile is None if the original data contained more than 1% nan's

        if rotate:
            for i in range(4):
                number = tile_index * 4 + i
                tile_name = f"{year}_{month}_{file_index}_{number:02d}"
                np.save(tar_dir.joinpath(tile_name), np.rot90(tile, i))
        else:
            tile_name = f"{year}_{month}_{file_index}_{tile_index:02d}"
            np.save(tar_dir.joinpath(tile_name), tile)

def is_outlier(src: Path, band: int) -> bool:
    """Check if the file is an outlier based on specified criteria."""
    band_size = {3: 660, 10: 200, 20: 100, 60: 33}
    raster = RasterCollection.from_multi_band_raster(src)
    return min(raster['lai'].values.shape) < band_size[band]

def create_tiles(data: np.ndarray, in_situ: bool, tiles: int = 4):
    """Create tiles from the given data."""
    sub_data = []
    arr_x, arr_y = data.shape
    x, y = arr_x // tiles, arr_y // tiles
    for i in range(tiles):
        for j in range(tiles):
            x_start, x_end = i * x, (i + 1) * x
            y_start, y_end = j * y, (j + 1) * y

            if not in_situ:
                margin = 40
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

            if np.isnan(tile).sum() / tile.size > 0.01 and not in_situ:
                sub_data.append(None)
            else:
                sub_data.append(np.clip(np.nan_to_num(tile), 0., 8.))
    return sub_data

def preprocess_folder(folder: Path, band: str, in_situ: bool) -> None:
    """Preprocess all files in the specified folder and save the tiles in the target directory."""
    if in_situ is True:
        target_dir = DATA_DIR.joinpath("processed", "in_situ", band)
    else:
        target_dir = DATA_DIR.joinpath("processed", band)

    target_dir.mkdir(exist_ok=True, parents=True)

    if band == "4b" or band == "8b":
        file_ending = band + "ands.tif"
    else:
        file_ending = band + "_lai.tif"

    for file in folder.iterdir():
        if file.name.endswith(file_ending):
            if not is_outlier(file, FOLDERS[band]):
                preprocess_file(file, target_dir, in_situ)

def preprocess(satellite: str, in_situ: bool):
    months = {"jan": "01", "feb": "02", "mar": "03", "apr": "04", 
          "may": "05", "jun": "06", "jul": "07", "aug": "08", 
          "sep": "09", "oct": "10", "nov": "11", "dec": "12"}
    
    if in_situ:
        folder = DATA_DIR.joinpath("filtered", "in_situ", satellite + "_in_situ")
    else:
        folder = DATA_DIR.joinpath("filtered", satellite)
        
    bands = ("10m", "20m") if satellite == "sentinel" else ("4b", "8b")
    
    for year in folder.iterdir():
        for month in year.iterdir():
            if not month.name[-3:] in months:
                continue
            print(f"Preprocess {satellite} data with in_situ ({in_situ}) for {year.name} {month.name}")
            data = month.joinpath("lai")
            preprocess_folder(data, bands[0], in_situ)
            preprocess_folder(data, bands[1], in_situ)

def rename_in_situ_data():
    index_to_field = {'0000': 'broatefaeld', '0001': 'bramenwies', '0002': 'fluegenrain', '0003': 'hohrueti', 
                      '0004': 'altkloster', '0005': 'ruetteli', '0006': 'witzwil', '0007': 'strickhof', '0008': 'eschikon'}
    in_situ_dir = DATA_DIR.joinpath("processed", "in_situ")
    for root, _, files in os.walk(in_situ_dir):
        for file in files:
            path = Path(root).joinpath(file)
            if file[8:12].isdigit():
                new_name = file.replace(file[8:12], index_to_field[file[8:12]])
                path.rename(Path(root).joinpath(new_name))

def check_and_remove_outliers(threshold:float = .75):
    print(f"Checking for outliers which have more than {int((1-threshold)*100)}% of zero entries")
    folder_names = ["4b", "8b", "10m", "20m"]
    folders = [DATA_DIR.joinpath("processed", folder_name) for folder_name in folder_names]

    to_remove = list(set([count_zeros(folder, threshold) for folder in folders]))

    number_of_files = len([file for file in folders[0].iterdir()])
    print(f"Percentage of outliers {len(to_remove) / number_of_files * 100}%. Removing them now.")

    for folder in folders:
        remove_outliers(folder, to_remove)

def count_zeros(folder: Path, threshold: float):
    percentage_zeros = []
    for file in folder.iterdir():
        file_np = np.load(file)
        percentage_zeros.append((np.count_nonzero(file_np) / np.size(file_np), file.name))
    return [x[1] for x in percentage_zeros if x[0] < threshold]

def remove_outliers(folder: Path, files_to_remove: list[str]):
    for file in files_to_remove:
        file_location = folder.joinpath(file)
        assert file_location.exists(), file_location
        file_location.unlink()

def do_histogram_matching(s2_bands: str, ps_bands: str):
    s2_folder = DATA_DIR.joinpath("processed", s2_bands)
    ps_folder = DATA_DIR.joinpath("processed", ps_bands)
    histogram_folder = DATA_DIR.joinpath("processed", f"hist_{s2_bands}_{ps_bands}")
    histogram_folder.mkdir()

    filenames = [file.name for file in s2_folder.iterdir()]
    for filename in filenames:
        s2_file = s2_folder.joinpath(filename)
        ps_file = ps_folder.joinpath(filename)
        np.save(histogram_folder.joinpath(filename), match_histograms(np.load(ps_file), np.load(s2_file)))

def psnr(x, y):
    return 20 * log10(8. / sqrt(np.mean((x - y) ** 2)))

def plot_histogram(s2_bands: str, ps_bands: str):
    s2_folder = DATA_DIR.joinpath("processed", s2_bands)
    ps_folder = DATA_DIR.joinpath("processed", ps_bands)
    hist_folder = DATA_DIR.joinpath("processed", f"hist_{s2_bands}_{ps_bands}")

    filenames = [file.name for file in s2_folder.iterdir()]
    # assert filenames == [file.name for file in ps_folder.iterdir()]
    # assert filenames == [file.name for file in hist_folder.iterdir()]

    scores_normal_ssim = []
    scores_normal_psnr = []
    scores_hist_ssim = []
    scores_hist_psnr = []

    for filename in filenames:
        s2_file = np.load(s2_folder.joinpath(filename)) 
        ps_file = np.load(ps_folder.joinpath(filename))
        hist_file = np.load(hist_folder.joinpath(filename))

        ssim_normal, _ = ssim((s2_file * (255 / 8)).astype(np.uint8), (ps_file * (255 / 8)).astype(np.uint8), full=True)
        ssim_hist, _ = ssim((s2_file * (255 / 8)).astype(np.uint8), (hist_file * (255 / 8)).astype(np.uint8), full=True)

        psnr_normal = psnr(s2_file, ps_file)
        psnr_hist = psnr(s2_file, hist_file)

        scores_normal_ssim.append(ssim_normal)
        scores_hist_ssim.append(ssim_hist)
        scores_normal_psnr.append(psnr_normal)
        scores_hist_psnr.append(psnr_hist)

    plt.hist(scores_normal_psnr, bins=33, range=(0, 32), color='red', alpha=0.5, label='Normal PSNR')
    plt.hist(scores_hist_psnr, bins=33, range=(0, 32), color='blue', alpha=0.5, label='HIST PSNR')
    overlap = np.intersect1d(scores_normal_psnr, scores_hist_psnr)
    plt.hist(overlap, bins=33, range=(0,32), color='purple', alpha=0.7, label='Overlap')
    plt.legend()
    plt.xlabel('PSNR')
    plt.ylabel('Frequency')
    plt.title(f'Histograms of PSNR values {s2_bands}-{ps_bands}')
    plt.show()

    plt.hist(scores_normal_ssim, bins=33, range=(0, 1), color='red', alpha=0.5, label='Normal SSIM')
    plt.hist(scores_hist_ssim, bins=33, range=(0, 1), color='blue', alpha=0.5, label='HIST SSIM')
    overlap = np.intersect1d(scores_normal_ssim, scores_hist_psnr)
    plt.hist(overlap, bins=33, range=(0,1), color='purple', alpha=0.7, label='Overlap')
    plt.legend()
    plt.xlabel('SSIM')
    plt.ylabel('Frequency')
    plt.title(f'Histograms of SSIM values {s2_bands}-{ps_bands}')
    plt.show()

    print("Mean SSIM Normal: ", np.sum(scores_normal_ssim) / len(scores_normal_ssim))
    print("Mean PSNR Normal: ", np.sum(scores_normal_psnr) / len(scores_normal_psnr))
    print("Mean SSIM Hist: ", np.sum(scores_hist_ssim) / len(scores_hist_ssim))
    print("Mean PSNR Hist: ", np.sum(scores_hist_psnr) / len(scores_hist_psnr))


def show_max_min(s2_bands: str, ps_bands: str, hist: bool):
    s2_folder = DATA_DIR.joinpath("processed", s2_bands)

    if hist:
        ps_folder = DATA_DIR.joinpath("processed", f"hist_{s2_bands}_{ps_bands}")
        titel_appendix = "With histogram matching."
    else:    
        ps_folder = DATA_DIR.joinpath("processed", ps_bands)
        titel_appendix = "No histogram matching."

    filenames = [file.name for file in s2_folder.iterdir()]
    scores = []
    for filename in filenames:
        s2_file = np.load(s2_folder.joinpath(filename)) 
        ps_file = np.load(ps_folder.joinpath(filename))

        ssim_score, _ = ssim((s2_file * (255 / 8)).astype(np.uint8), (ps_file * (255 / 8)).astype(np.uint8), full=True)
        psnr_score = psnr(s2_file, ps_file)

        scores.append((psnr_score, ssim_score, filename))

    psnr_scores = [entry[0] for entry in scores]
    print("PSNR Mean: ", np.sum(psnr_scores) / len(psnr_scores))

    psnr_scores = sorted(scores, key=lambda x: x[0])
    print(psnr_scores[-1])
    biggest_psnr_s2 = np.load(s2_folder.joinpath(psnr_scores[-1][2]))
    biggest_psnr_ps = np.load(ps_folder.joinpath(psnr_scores[-1][2]))
    plt.title(f"Biggest PSNR score {psnr_scores[-1][0]}. {titel_appendix}")
    plt.imshow(np.concatenate((biggest_psnr_s2, biggest_psnr_ps), axis=1))
    plt.show()

    print(psnr_scores[0])
    lowest_psnr_s2 = np.load(s2_folder.joinpath(psnr_scores[0][2]))
    lowest_psnr_ps = np.load(ps_folder.joinpath(psnr_scores[0][2]))
    plt.title(f"Lowest PSNR score {psnr_scores[0][0]}. {titel_appendix}")
    plt.imshow(np.concatenate((lowest_psnr_s2, lowest_psnr_ps), axis=1))
    plt.show()

    ssim_scores = sorted(scores, key=lambda x: x[1])
    print(ssim_scores[-1])
    biggest_similar_s2 = np.load(s2_folder.joinpath(ssim_scores[-1][2]))
    biggest_similar_ps = np.load(ps_folder.joinpath(ssim_scores[-1][2]))
    plt.title(f"Biggest similarity score {ssim_scores[-1][1]}. {titel_appendix}")
    plt.imshow(np.concatenate((biggest_similar_s2, biggest_similar_ps), axis=1))
    plt.show()

    print(ssim_scores[0])
    lowest_similar_s2 = np.load(s2_folder.joinpath(ssim_scores[0][2]))
    lowest_similar_ps = np.load(ps_folder.joinpath(ssim_scores[0][2]))
    plt.title(f"Lowest similarity score {ssim_scores[0][1]}. {titel_appendix}")
    plt.imshow(np.concatenate((lowest_similar_s2, lowest_similar_ps), axis=1))
    plt.show()

def show_error():
    filename = "2022_07_0263_03.npy"
    ps_file = DATA_DIR.joinpath("processed", "4b", filename)
    s2_file = DATA_DIR.joinpath("processed", "10m", filename)

    ps_np = np.load(ps_file)
    s2_np = np.load(s2_file)

    diff = np.abs(ps_np - s2_np)

    x = np.concatenate((s2_np, diff, ps_np), axis=1)

    plt.title(f"Biggest difference")
    plt.imshow(x)
    plt.show()

def remove_lowest_10_percentile(s2_bands: str, ps_bands: str, hist: bool) -> None:
    s2_folder = DATA_DIR.joinpath("processed", s2_bands)
    
    if hist is True:
        ps_folder = DATA_DIR.joinpath("processed", f"hist_{s2_bands}_{ps_bands}") 
    else:
        ps_folder = DATA_DIR.joinpath("processed", ps_bands)

    filenames = [file.name for file in s2_folder.iterdir()]

    scores_ssim = []
    scores_psnr = []

    for filename in filenames:
        s2_file = np.load(s2_folder.joinpath(filename)) 
        ps_file = np.load(ps_folder.joinpath(filename))

        ssim_normal, _ = ssim((s2_file * (255 / 8)).astype(np.uint8), (ps_file * (255 / 8)).astype(np.uint8), full=True)
        psnr_normal = psnr(s2_file, ps_file)

        scores_ssim.append((ssim_normal, filename))
        scores_psnr.append((psnr_normal, filename))

    sorted_ssim = [x[1] for x in sorted(scores_ssim, key = lambda x: x[0])]
    sorted_psnr = [x[1] for x in sorted(scores_psnr, key = lambda x: x[0])]

    first_10_percent_ssim = sorted_ssim[0:int(len(sorted_ssim)*0.1)]
    first_10_percent_psnr = sorted_psnr[0:int(len(sorted_psnr)*0.1)]

    intersection = list(set(first_10_percent_ssim + first_10_percent_psnr))
    remove_outliers(s2_folder, intersection)
    remove_outliers(ps_folder, intersection)

def main():
    preprocess(satellite="sentinel", in_situ=False)
    preprocess(satellite="planetscope", in_situ=False)
    remove_unused_images(in_situ=False)

    preprocess(satellite="sentinel", in_situ=True)
    preprocess(satellite="planetscope", in_situ=True)
    remove_unused_images(in_situ=True)
    rename_in_situ_data()

    # do_histogram_matching("10m", "4b")
    # do_histogram_matching("10m", "8b")
    do_histogram_matching("20m", "4b")
    # do_histogram_matching("20m", "8b")

    # plot_histogram("20m", "4b")
    # plot_histogram("20m", "8b")

    # show_max_min("10m", "4b", True)
    # show_max_min("10m", "8b", True)
    # show_max_min("10m", "4b", False)
    # show_max_min("10m", "8b", False)
    # show_max_min("20m", "4b", True)
    # show_max_min("20m", "8b", True)
    # show_max_min("20m", "4b", False)
    # show_max_min("20m", "8b", False)

if __name__ == "__main__":
    main()