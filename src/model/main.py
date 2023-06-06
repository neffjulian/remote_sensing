import argparse
import pytorch_lightning as pl

from dataset import SRDataModule

from edsr import EDSR
from srcnn import SRCNN
from utils import visualize_output

MODELS = {
    'edsr': EDSR,
    'srcnn': SRCNN
}

def main(name: str, sentinel_bands: str, planetscope_bands: str):
    data_module = SRDataModule(sentinel_bands, planetscope_bands)
    data_module.setup()

    model = MODELS[name]()
    trainer = pl.Trainer()

    # trainer.fit(model, data_module)
    # trainer.test(model, data_module)

    output = trainer.predict(model, data_module)
    visualize_output(name, output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--sentinel_bands", default="10m", type=str)
    parser.add_argument("--planetscope_bands", default="4b", type=str)
    args = parser.parse_args()
    main(**vars(args))