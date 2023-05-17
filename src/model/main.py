import argparse

from torch.utils.data import DataLoader
import pytorch_lightning as pl

from edsr import EDSR
from srcnn import SRCNN
from dataset import get_datasets, show_random_result

def main(model: str):
    train_dataset, val_dataset = get_datasets()
    
    train_loader = DataLoader(train_dataset, batch_size=32)
    val_loader = DataLoader(val_dataset, batch_size=32)

    if model == "srcc":
        selected_model = SRCNN()
    elif model == "edsr":
        selected_model = EDSR()
    else:
        raise Exception("Invalid model selected!")

    show_random_result(selected_model, train_dataset)

    trainer = pl.Trainer(
        max_epochs = 10, 
        min_epochs = 5
        )
    trainer.fit(selected_model, train_loader, val_loader)

    for i in range(10):
        show_random_result(selected_model, train_dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    args = parser.parse_args()
    main(**vars(args))
    main()