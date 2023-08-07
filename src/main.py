import os
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities.memory import garbage_collection_cuda

from model.dataset import SRDataModule
from model.utils import visualize_output, get_in_situ, get_sample, visualize_in_situ, visualize_sample, get_ps_sample
from model.edsr import EDSR
from model.srcnn import SRCNN
from model.srresnet import SRResNet
from model.srgan import SRGAN
from model.esrgan import ESRGAN
from model.rrdb import RRDB
from model.srdiff_no_pos_encoding import SRDIFF_simple

DOTENV_PATH = Path(__file__).parent.parent.joinpath(".env")
LOG_DIR = Path(__file__).parent.parent.joinpath("logs")
WEIGHT_DIR = Path(__file__).parent.parent.joinpath("weights")
load_dotenv(DOTENV_PATH)
WANDB_API_KEY = os.getenv('WANDB_API_KEY')
os.environ["WANDB_SILENT"] = "true"

MODELS = {
    "edsr": EDSR,
    "srcnn": SRCNN,
    "srresnet": SRResNet,
    "srgan": SRGAN,
    "esrgan": ESRGAN,
    "rrdb": RRDB,
    "srdiff_simple": SRDIFF_simple 
}

def main(hparams: dict) -> None:
    garbage_collection_cuda()

    date_string = datetime.now().strftime("%Y_%m_%d_%H_%M")
    experiment_name = hparams["sentinel_resolution"] + "_" + hparams["experiment_name"] + "_" + date_string

    if hparams["train"] is True:
        seed_everything(hparams["random_seed"])
        model = MODELS[hparams["model"]["name"]](hparams)
        
        datamodule = SRDataModule(hparams)
        datamodule.setup()

        wandb_logger = WandbLogger(
            name=experiment_name, 
            project="remote_sensing",
            save_dir=LOG_DIR)

        trainer = pl.Trainer(
            max_epochs = hparams["trainer"]["max_epochs"],
            profiler = hparams["trainer"]["profiler"],
            log_every_n_steps = hparams["trainer"]["log_every_n_steps"],
            callbacks = [DeviceStatsMonitor()],
            logger = wandb_logger,
            default_root_dir=LOG_DIR,
            accelerator="auto",
            accumulate_grad_batches=2
        )

        trainer.fit(model=model, datamodule=datamodule)

    if hparams["predict"] is True:
        if hparams["train"] is False:
            model = MODELS[hparams["model"]["name"]].load_from_checkpoint(
                checkpoint_path=WEIGHT_DIR.joinpath(f"{hparams['model']['name']}.ckpt"), 
                map_location=torch.device('cpu'), 
                hparams=hparams
            )
        model.eval()
        
        in_situ = get_in_situ(hparams["planetscope_bands"], hparams["sentinel_resolution"])
        results_in_situ = []
        for lr, hr, s2, name in in_situ:
            sr = model(torch.tensor(lr).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).detach().numpy()
            results_in_situ.append((lr, sr, hr, s2, name))
        visualize_in_situ(results_in_situ, experiment_name)
        
        s2_tiles, ps_tiles, raster = get_sample(hparams["planetscope_bands"], hparams["sentinel_resolution"])
        sr_tiles = []
        for tile in s2_tiles:
            sr = model(torch.tensor(tile).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).detach().numpy()
            sr_tiles.append(sr)
        visualize_sample(s2_tiles, sr_tiles, ps_tiles, experiment_name, False, raster)
        print("Sample visualization done")
        ps_lr_tiles, ps_tiles, raster = get_ps_sample(hparams["planetscope_bands"])
        sr_tiles = []
        for tile in ps_lr_tiles:
            sr = model(torch.tensor(tile).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).detach().numpy()
            sr_tiles.append(sr)
        visualize_sample(ps_lr_tiles, sr_tiles, ps_tiles, experiment_name, True, raster)
        print("PS Sample done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, help="Choose a config file from the 'configs/' folder", required=True)

    args = parser.parse_args()
    with open(args.config, "r") as file:
        hparams = yaml.load(file, Loader=yaml.FullLoader)

    main(hparams)