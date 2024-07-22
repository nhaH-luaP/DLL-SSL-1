from model import build_model
from dataset import build_dataset
from utils import seed_everything

from birdset.datamodule.base_datamodule import DatasetConfig
from birdset.datamodule.birdset_datamodule import BirdSetDataModule

from torchvision.datasets import CIFAR10

import os
import hydra
import json
import logging
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as T


from lightning import Trainer, LightningDataModule
from omegaconf import OmegaConf

@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(args):
    # Print args
    logging.info('>>> Using config: \n%s', OmegaConf.to_yaml(args))

    # Create directory for results
    logging.info(">>> Creating output directory at: "+str(args.path.output_dir))
    os.makedirs(args.path.output_dir, exist_ok=True)

    # Enable reproducability
    logging.info(">>> Seed experiment with random seed {args.random_seed}.")
    seed_everything(args.random_seed + 42)
    
#    # Initialize Dataset
#    logging.info(f">>> Initialize Dataset {args.dataset}.")
#    dm = BirdSetDataModule(
#        dataset= DatasetConfig(
#            data_dir='/home/phahn/datasets/birdset/HSN', # specify your data directory!
#            dataset_name='HSN',
#            hf_path='DBD-research-group/BirdSet',
#            hf_name='HSN',
#            n_workers=3,
#            val_split=0.2,
#            task="multilabel",
#            classlimit=500,
#            eventlimit=5,
#            sampling_rate=32000,
#        ),
#    )
#
#    # Prepare the data (download dataset, ...)
#    dm.prepare_data()
#    dm.setup(stage="fit")

    # Using CIFAR-10 as dev example for ViT training
    dm = LitCIFAR10DataModule()

    # Initialize Model
    logging.info(f">>> Initialize Model {args.model.name}.")
    model = build_model(args)

    # Train Model
    trainer = Trainer(max_epochs=args.model.n_epochs)
    trainer.fit(model=model, datamodule=dm)

    # Export results
    history = {}
    with open(os.path.join(args.path.output_dir, 'results.json'), 'w') as f:
        json.dump(history, f)

    logging.info(">>> Finished.")


class LitCIFAR10DataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.247, 0.243, 0.262)

    def train_dataloader(self):
        transform = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(self.mean, self.std)])
        dataset = dataset = CIFAR10(
            root='/home/phahn/datasets/',
            train=True,
            transform=transform,
            download=True,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=64,
            num_workers=1,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        dataset = dataset = dataset = CIFAR10(
            root='/home/phahn/datasets/',
            train=False,
            transform=transform,
            download=True,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=64,
            num_workers=1,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()


if __name__ == '__main__':
    main()