import cv2
import numpy as np
from pathlib import Path
from eodal.core.raster import RasterCollection
from eodal.core.band import Band, GeoInfo
import matplotlib.pyplot as plt
import torch
import yaml

from model.rrdb import RRDB

filter_dir = Path(__file__).parent.parent.joinpath('data', 'filtered')
results_dir = Path(__file__).parent.parent.joinpath('data', 'results')

WEIGHT_DIR = Path(__file__).parent.parent.joinpath("weights")
CONFIG_DIR = Path(__file__).parent.parent.joinpath("configs")
VALIDATE_DIR = Path(__file__).parent.parent.joinpath("data", "validate", "bicubic")

with open(CONFIG_DIR.joinpath("rrdb.yaml"), "r") as f:
    hparams = yaml.load(f, Loader=yaml.FullLoader)

model = RRDB.load_from_checkpoint(
                checkpoint_path=WEIGHT_DIR.joinpath("rrdb.ckpt"), 
                map_location=torch.device('cpu'),
                hparams=hparams)
model.eval()


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

def process_files(lr: np.ndarray, algorithm: str = "rrdb") -> np.ndarray:
    data = torch.tensor(lr).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        if algorithm == "bicubic":
            out = torch.nn.functional.interpolate(data, scale_factor=6, mode="bicubic")
        elif algorithm == "rrdb":
            out = model(data)
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
    common_indices = get_common_indices(s2_bands, ps_bands)

    print(common_indices[0:25])

    # s2_dir = filter_dir.joinpath("sentinel")
    # ps_dir = filter_dir.joinpath("planetscope")
    # out_dir = VALIDATE_DIR.joinpath(f"{s2_bands}_{ps_bands}")

    # for year in s2_dir.iterdir():
    #     for month in year.iterdir():
    #         print(month.name)
    #         curr_ps_dir = ps_dir.joinpath(year.name, month.name, "lai")
    #         curr_s2_dir = s2_dir.joinpath(year.name, month.name, "lai")

    #         if not curr_ps_dir.exists() or not curr_s2_dir.exists():
    #             continue

    #         s2_files = [curr_s2_dir.joinpath(f"{index}_scene_{s2_bands}_lai.tif") for index in common_indices]
    #         # ps_files = [curr_ps_dir.joinpath(f"{index}_lai_{ps_bands}ands.tif") for index in common_indices]

    #         validate_dir = out_dir.joinpath(year.name, month.name)
    #         validate_dir.mkdir(parents=True, exist_ok=True)
    #         process_s2_files(validate_dir, s2_files)


if __name__ == '__main__':
    s2_bands = "20m"
    ps_bands = "4b"

    main(s2_bands, ps_bands)