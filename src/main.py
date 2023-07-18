import os
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

import pytorch_lightning as pl
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities.memory import garbage_collection_cuda

from model.dataset import SRDataModule
from model.utils import visualize_output

from model.edsr import EDSR
from model.srcnn import SRCNN
from model.srresnet import SRResNet
from model.srgan import SRGAN
from model.esrgan import ESRGAN
from model.rrdb import RRDB


DOTENV_PATH = Path(__file__).parent.parent.joinpath(".env")
LOG_DIR = Path(__file__).parent.parent.joinpath("logs")
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
}

def main(hparams: dict) -> None:
    garbage_collection_cuda()

    date_string = datetime.now().strftime("%Y_%m_%d_%H_%M")
    experiment_name = hparams["experiment_name"] + "_" + date_string

    seed_everything(hparams["random_seed"])
    model = MODELS[hparams["model"]["name"]](hparams)

    datamodule = SRDataModule(hparams)
    datamodule.setup()

    if hparams["train"] is True:
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
            accelerator="gpu",
            devices=4,
            accumulate_grad_batches=2
        )

        trainer.fit(model=model, datamodule=datamodule)

    if hparams["predict"] is True:
        trainer = pl.Trainer(devices=1, accelerator="gpu", default_root_dir=LOG_DIR)
        model.eval()
        output = trainer.predict(model=model, datamodule=datamodule)
        visualize_output(experiment_name, output)

        print("--------------------")
        print(hparams)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, help="Choose a config file from the 'configs/' folder", required=True)

    args = parser.parse_args()
    with open(args.config, "r") as file:
        hparams = yaml.load(file, Loader=yaml.FullLoader)

    main(hparams)