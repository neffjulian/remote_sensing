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
# from pytorch_lightning.tuner import Tuner

from model.dataset import SRDataModule
from model.edsr import EDSR
from model.srcnn import SRCNN
from model.utils import visualize_output

DOTENV_PATH = Path(__file__).parent.parent.joinpath(".env")
load_dotenv(DOTENV_PATH)
WANDB_API_KEY = os.getenv('WANDB_API_KEY')

MODELS = {
    "edsr": EDSR,
    "srcnn": SRCNN
}


def main(hparams: dict) -> None:
    date_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    experiment_name = hparams["experiment_name"] + "_" + date_string

    seed_everything(hparams["random_seed"])
    model = MODELS[hparams["model"]["name"]](hparams)

    if "resume_from_checkpoint" in hparams:
        model.load_from_checkpoint(hparams["resume_from_checkpoint"])

    data_module = SRDataModule(hparams)
    data_module.setup()

    wandb_logger = WandbLogger(experiment_name, project="remote_sensing")

    trainer = pl.Trainer(
        max_epochs = hparams["trainer"]["max_epochs"],
        profiler = hparams["trainer"]["profiler"],
        log_every_n_steps = hparams["trainer"]["log_every_n_steps"],
        callbacks = [DeviceStatsMonitor()],
        logger = wandb_logger
    )

    # # tuner = Tuner(trainer)
    
    # # tuner.scale_batch_size(model, datamodule=data_module)
    # # tuner.lr_find(model)
    if hparams["train"] is True:
        trainer.fit(model, data_module)
        trainer.test(model, data_module)
    if hparams["predict"] is True:
        model.eval()
        output = trainer.predict(model, data_module)
        visualize_output(experiment_name, output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, help="Choose a config file from the 'configs/' folder", required=True)

    args = parser.parse_args()
    with open(args.config, "r") as file:
        hparams = yaml.load(file, Loader=yaml.FullLoader)

    main(hparams)