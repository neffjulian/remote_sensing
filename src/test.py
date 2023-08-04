from math import log10, sqrt
import shutil
import cv2
import numpy as np
from pathlib import Path
from eodal.core.raster import RasterCollection
from eodal.core.band import Band, GeoInfo
import matplotlib.pyplot as plt
import torch
import yaml

from model.rrdb import RRDB
from model.srcnn import SRCNN
from model.edsr import EDSR

filter_dir = Path(__file__).parent.parent.joinpath('data', 'filtered')
results_dir = Path(__file__).parent.parent.joinpath('data', 'results')

WEIGHT_DIR = Path(__file__).parent.parent.joinpath("weights")
CONFIG_DIR = Path(__file__).parent.parent.joinpath("configs")
VALIDATE_DIR = Path(__file__).parent.parent.joinpath("data", "validate")

with open(CONFIG_DIR.joinpath("rrdb.yaml"), "r") as f:
    hparams = yaml.load(f, Loader=yaml.FullLoader)

rrdb = RRDB.load_from_checkpoint(
                checkpoint_path=WEIGHT_DIR.joinpath("rrdb.ckpt"), 
                map_location=torch.device('cpu'),
                hparams=hparams)
rrdb.eval()

with open(CONFIG_DIR.joinpath("srcnn.yaml"), "r") as f:
    hparams = yaml.load(f, Loader=yaml.FullLoader)

srcnn = SRCNN.load_from_checkpoint(
                checkpoint_path=WEIGHT_DIR.joinpath("srcnn.ckpt"),
                map_location=torch.device('cpu'),
                hparams=hparams)
srcnn.eval()

with open(CONFIG_DIR.joinpath("edsr.yaml"), "r") as f:
    hparams = yaml.load(f, Loader=yaml.FullLoader)

edsr = EDSR.load_from_checkpoint(
                checkpoint_path=WEIGHT_DIR.joinpath("edsr.ckpt"),
                map_location=torch.device('cpu'),
                hparams=hparams)
edsr.eval()

# First 25 entries which the "get_common_indices" function returns. Results in around 10% of all data which is also used for validation
INDICES = ['0000', '0001', '0002', '0003', '0004', '0006', '0008', '0011', '0012', '0023', '0025', '0026', '0028', '0029', '0030', '0031', '0032', '0033', '0034', '0035', '0036', '0037', '0038', '0040', '0046']


def get_common_indices(s2_bands, ps_bands):
    # Iterates over the filtered data and returns the indices for which we have a pair of images for each month

    s2_dir = filter_dir.joinpath("sentinel")
    ps_dir = filter_dir.joinpath("planetscope")

    indices_per_month = []
    for year in s2_dir.iterdir():
        for month in year.iterdir():
            curr_ps_dir = ps_dir.joinpath(year.name, month.name, "lai")
            curr_s2_dir = s2_dir.joinpath(year.name, month.name, "lai")

            if not curr_ps_dir.exists() or not curr_s2_dir.exists():
                continue

            s2_files = list(curr_s2_dir.glob(f"*_scene_{s2_bands}_lai.tif"))
            ps_files = list(curr_ps_dir.glob(f"*_lai_{ps_bands}ands.tif"))

            common_indices = list(set([f.name[0:4] for f in s2_files]).intersection([f.name[0:4] for f in ps_files]))
            indices_per_month.append(common_indices)
    return sorted(list(set.intersection(*map(set, indices_per_month))), key=lambda x: int(x))

def process_files(lr: np.ndarray, algorithm: str = "edsr") -> np.ndarray:
    data = torch.tensor(lr).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        if algorithm == "bicubic":
            out = torch.nn.functional.interpolate(data, scale_factor=6, mode="bicubic")
        elif algorithm == "rrdb":
            out = rrdb(data)
        elif algorithm == "srcnn":
            out = srcnn(data)
        elif algorithm == "edsr":
            out = edsr(data)
        else:
            raise Exception
    return out.squeeze(0).squeeze(0).numpy()


def reconstruct_tiles(tiles: list[np.ndarray]) -> np.ndarray:
    x, y = tiles[0].shape
    new_x, new_y = x * 4, y * 4
    reconstructed = np.empty((new_x, new_y))
    for i in range(0, new_x, x):
        for j in range(0, new_y, y):
            reconstructed[i:i+x, j:j+y] = tiles.pop(0)
    return reconstructed

def create_raster(data: np.ndarray, collection: RasterCollection):
    geo_info = GeoInfo(
            epsg=collection["lai"].geo_info.epsg,
            ulx=collection["lai"].geo_info.ulx,
            uly=collection["lai"].geo_info.uly,
            pixres_x=10/3, # New pixres is now 3.33 m as we have 6 times more pixels
            pixres_y=-10/3
        )

    raster = RasterCollection(
        band_constructor=Band,
        band_name="lai",
        values = data,
        geo_info = geo_info
    )

    return raster

def process_s2_files(dir: Path, s2_files: list[Path]):
    for path in s2_files:
        if dir.joinpath(path.name[0:4] + ".tif").exists():
            continue
        print(dir.name, path.name[0:4])

        file = RasterCollection.from_multi_band_raster(path)
        lai = np.nan_to_num(file["lai"].values)
        shape = lai.shape

        ps_file_interp = cv2.resize(lai, (100, 100), interpolation=cv2.INTER_AREA)
        tiles = [ps_file_interp[i:i+25, j:j+25] for i in range(0, 100, 25) for j in range(0, 100, 25)]

        tiles = [process_files(tile) for tile in tiles]
        sr_file = cv2.resize(reconstruct_tiles(tiles), (shape[1] * 6, shape[0] * 6), interpolation=cv2.INTER_CUBIC)
        sr_raster = create_raster(sr_file, file)
        sr_raster.to_rasterio(dir.joinpath(path.name[0:4] + ".tif"))

def main(s2_bands: str, ps_bands: str):
    # Get the indices for which we have a pair of images for each month
    # common_indices = get_common_indices(s2_bands, ps_bands)
    common_indices = INDICES

    s2_dir = filter_dir.joinpath("sentinel")
    ps_dir = filter_dir.joinpath("planetscope")
    out_dir = VALIDATE_DIR.joinpath(f"{s2_bands}_edsr")

    for year in s2_dir.iterdir():
        for month in year.iterdir():
            curr_ps_dir = ps_dir.joinpath(year.name, month.name, "lai")
            curr_s2_dir = s2_dir.joinpath(year.name, month.name, "lai")

            if not curr_ps_dir.exists() or not curr_s2_dir.exists():
                continue

            s2_files = [curr_s2_dir.joinpath(f"{index}_scene_{s2_bands}_lai.tif") for index in common_indices]
            # ps_files = [curr_ps_dir.joinpath(f"{index}_lai_{ps_bands}ands.tif") for index in common_indices]

            validate_dir = out_dir.joinpath(month.name)
            validate_dir.mkdir(parents=True, exist_ok=True)
            process_s2_files(validate_dir, s2_files)

if __name__ == '__main__':
    s2_bands = "20m"
    ps_bands = "8b"

    main(s2_bands, ps_bands)

    # sentinel_dir = filter_dir.joinpath("sentinel", "2022")
    # planetscope_dir = filter_dir.joinpath("planetscope", "2022")
    # months = [month.name for month in sentinel_dir.iterdir() if month.name[:2].isdigit()]

    # for index in INDICES:
    #     for month in months:
    #         print(f"Processing {index} for {month}")
    #         sentinel_file = sentinel_dir.joinpath(month, "lai", f"{index}_scene_{s2_bands}_lai.tif")
    #         planetscope_file = planetscope_dir.joinpath(month, "lai", f"{index}_lai_{ps_bands}ands.tif")

    #         if not sentinel_file.exists() or not planetscope_file.exists():
    #             print(f"Missing {index} for {month}")
    #             break

    #         VALIDATE_DIR.joinpath(s2_bands, month).mkdir(parents=True, exist_ok=True)
    #         VALIDATE_DIR.joinpath(ps_bands, month).mkdir(parents=True, exist_ok=True)

    #         s2_new_loc = VALIDATE_DIR.joinpath(s2_bands, month, f"{index}.tif")
    #         ps_new_loc = VALIDATE_DIR.joinpath(ps_bands, month, f"{index}.tif")
    #         shutil.copy(sentinel_file, s2_new_loc)
    #         shutil.copy(planetscope_file, ps_new_loc)