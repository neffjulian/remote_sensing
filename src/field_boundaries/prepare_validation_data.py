import argparse
import sys
import cv2
import numpy as np
from pathlib import Path
from eodal.core.raster import RasterCollection
from eodal.core.band import Band, GeoInfo
import torch
import yaml

sys.path.append(str(Path(__file__).parent.parent.parent.joinpath("src")))
from model.rrdb import RRDB
from model.srcnn import SRCNN
from model.edsr import EDSR

filter_dir = Path(__file__).parent.parent.parent.joinpath('data', 'filtered')
results_dir = Path(__file__).parent.parent.parent.joinpath('data', 'results')

WEIGHT_DIR = Path(__file__).parent.parent.parent.joinpath("weights")
CONFIG_DIR = Path(__file__).parent.parent.parent.joinpath("configs")
VALIDATE_DIR = Path(__file__).parent.parent.parent.joinpath("data", "validate")

MODEL = {
    "rrdb": RRDB,
    "srcnn": SRCNN,
    "edsr": EDSR
}

INDICES = ['0000', '0001', '0002', '0003', '0004', '0006', '0008', '0011', '0012', '0023', '0025', '0026', '0028', '0029', '0031', '0032', '0033', '0034', '0035', '0036', '0037', '0038', '0040', '0046']

def get_common_indices(s2_bands, ps_bands):
    """
    Iterates over the filtered data and returns the indices for which we have a pair of images for each month
    """
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

def load_model_from_checkpoint(model_name: str):
    with open(CONFIG_DIR.joinpath(f"{model_name}.yaml"), "r") as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)
    model = MODEL[model_name].load_from_checkpoint(
        checkpoint_path=WEIGHT_DIR.joinpath(f"{model_name}.ckpt"),
        map_location=torch.device('cpu'),
        hparams=hparams
    )
    model.eval()
    return model

def process_files(lr: np.ndarray, model: None | SRCNN | EDSR | RRDB) -> np.ndarray:
    data = torch.tensor(lr).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        if model is None:
            out = torch.nn.functional.interpolate(data, scale_factor=6, mode="bicubic")
        else:
            out = model(data)
    return out.squeeze(0).squeeze(0).numpy()

def reconstruct_tiles(tiles: list[np.ndarray]) -> np.ndarray:
    x, y = tiles[0].shape
    new_x, new_y = x * 4, y * 4
    reconstructed = np.empty((new_x, new_y))
    for i in range(0, new_x, x):
        for j in range(0, new_y, y):
            reconstructed[i:i+x, j:j+y] = tiles.pop(0)
    return reconstructed

def create_raster(data: np.ndarray, collection: RasterCollection, out_dir: Path):
    raster = RasterCollection(
        band_constructor=Band,
        band_name="lai",
        values = data,
        geo_info = GeoInfo(
            epsg=collection["lai"].geo_info.epsg,
            ulx=collection["lai"].geo_info.ulx,
            uly=collection["lai"].geo_info.uly,
            pixres_x=10/3, # New pixres is now 3.33 m as we have 6 times more pixels
            pixres_y=-10/3
        )
    )
    raster.to_rasterio(out_dir)

def process_s2_files(dir: Path, s2_files: list[Path], model: None | SRCNN | EDSR | RRDB, factor: int):
    for path in s2_files:
        if dir.joinpath(path.name[0:4] + ".tif").exists():
            continue

        file = RasterCollection.from_multi_band_raster(path)
        lai = np.nan_to_num(file["lai"].values)
        shape = lai.shape

        ps_file_interp = cv2.resize(lai, (100, 100), interpolation=cv2.INTER_AREA)
        tiles = [ps_file_interp[i:i+25, j:j+25] for i in range(0, 100, 25) for j in range(0, 100, 25)]

        sr_image = reconstruct_tiles([process_files(tile, model) for tile in tiles])
        sr_file = cv2.resize(sr_image, (shape[1] * factor, shape[0] * factor), interpolation=cv2.INTER_CUBIC)
        create_raster(sr_file, file, dir.joinpath(path.name[0:4] + ".tif"))

def prepare_validation_data(s2_bands: str, model_name: str):
    # Get the indices for which we have a pair of images for each month
    # hardcoded so we do not have any overlap with the training data
    common_indices = INDICES
    # common_indices = get_common_indices(s2_bands, ps_bands)

    s2_dir = filter_dir.joinpath("sentinel")
    ps_dir = filter_dir.joinpath("planetscope")
    out_dir = VALIDATE_DIR.joinpath(f"{s2_bands}_{model_name}")
    model = load_model_from_checkpoint(model_name) if model_name != "bicubic" else None
    upscaling_factor = 3 if s2_bands == "10m" else 6

    for year in s2_dir.iterdir():
        for month in year.iterdir():
            print("Processing:", month.name)
            curr_ps_dir = ps_dir.joinpath(year.name, month.name, "lai")
            curr_s2_dir = s2_dir.joinpath(year.name, month.name, "lai")

            if not curr_ps_dir.exists() or not curr_s2_dir.exists():
                continue

            s2_files = [curr_s2_dir.joinpath(f"{index}_scene_{s2_bands}_lai.tif") for index in common_indices]

            validate_dir = out_dir.joinpath(month.name)
            validate_dir.mkdir(parents=True, exist_ok=True)
            process_s2_files(validate_dir, s2_files, model, upscaling_factor)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentinel_bands", type=str, required=True, help="Either '10m' or '20m'")
    parser.add_argument("--model", type=str, required=True, help="Either 'bicubic', 'srcnn', 'rrdb' or 'edsr'")
    args = parser.parse_args()

    prepare_validation_data(args.sentinel_bands, args.model)