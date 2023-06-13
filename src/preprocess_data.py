import os
from pathlib import Path

import numpy as np
import cv2
from eodal.core.raster import RasterCollection

MONTHS = {"jan": "01", "feb": "02", "mar": "03", "apr": "04", 
          "may": "05", "jun": "06", "jul": "07", "aug": "08", 
          "sep": "09", "oct": "10", "nov": "11", "dec": "12"}

DATA_DIR = Path(__file__).parent.parent.joinpath("data")
FOLDERS = {"4b": 3, "8b": 3, "10m": 10, "20m": 20}
OUT_DIM = (640, 640)

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

    [file.unlink() for file in dir.iterdir() if file.name not in files_to_keep]

def remove_unused_images(in_situ: bool = False) -> None:
    """Remove unused images from the processed folders."""
    files = [get_filenames(folder, in_situ) for folder in FOLDERS.keys()]
    files_to_keep = list(set.intersection(*map(set, files)))

    for folder in FOLDERS.keys():
        remove_files(folder, files_to_keep, in_situ)

    # Verify that all 4 folders contain the same elements
    if in_situ is True:
        files = [[file.name for file in DATA_DIR.joinpath("processed", "in_situ", folder).iterdir()] for folder in FOLDERS.keys()]
    else:
        files = [[file.name for file in DATA_DIR.joinpath("processed", folder).iterdir()] for folder in FOLDERS.keys()]
    
    assert all(files[0] == files[i] for i in range(1, len(files)))

def preprocess_file(src_data: Path, tar_dir: Path, rotate: bool = False) -> None:
    """Preprocess a single file and save the tiles."""
    file_index = src_data.name[:4]
    month = src_data.parent.parent.name[:2]
    year = src_data.parent.parent.parent.name

    raster = RasterCollection.from_multi_band_raster(src_data)
    lai = cv2.resize(raster["lai"].values, OUT_DIM, interpolation=cv2.INTER_CUBIC)
    lai_processed = np.clip(np.nan_to_num(lai), 0, 8)
    tiles = create_tiles(lai_processed)

    for tile_index, tile in enumerate(tiles):
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
    data = raster['lai'].values
    x, y = data.shape

    if min(x, y) < band_size[band]:
        return True

    percentage_ones = (np.sum(data == 1) / data.size) * 100
    percentage_zeros = (np.sum(data == 0) / data.size) * 100

    if max(percentage_ones, percentage_zeros) > 15:
        return True

    return False

def create_tiles(data: np.ndarray, tiles: int = 2):
    """Create tiles from the given data."""
    sub_data = []
    arr_x, arr_y = data.shape
    x, y = int(arr_x / tiles), int(arr_y / tiles)

    for i in range(tiles):
        for j in range(tiles):
            sub_data.append(data[(i * x):((i + 1) * x), (j * y):((j + 1) * y)])

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
                preprocess_file(file, target_dir)

def preprocess(satellite: str, in_situ: bool):
    if in_situ:
        folder = DATA_DIR.joinpath("filtered", "in_situ", satellite + "_in_situ")
    else:
        folder = DATA_DIR.joinpath("filtered", satellite)
        
    bands = ("10m", "20m") if satellite == "sentinel" else ("4b", "8b")
    
    for year in folder.iterdir():
        for month in year.iterdir():
            if not month.name[-3:] in MONTHS:
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

def main():
    preprocess(satellite="sentinel", in_situ=False)
    preprocess(satellite="planetscope", in_situ=False)
    remove_unused_images(in_situ=False)

    preprocess(satellite="sentinel", in_situ=True)
    preprocess(satellite="planetscope", in_situ=True)
    remove_unused_images(in_situ=True)

    rename_in_situ_data()
    
if __name__ == "__main__":
    main()