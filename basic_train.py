from model import build_model
from dataset import build_dataset
from utils import seed_everything
from birdset.datamodule.base_datamodule import DatasetConfig
from birdset.datamodule.birdset_datamodule import BirdSetDataModule

import os
import hydra
import json
import logging

from lightning import Trainer, LightningModule
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
    
    # Initialize Dataset
    logging.info(f">>> Initialize Dataset {args.dataset}.")
    dm = BirdSetDataModule(
        dataset= DatasetConfig(
            data_dir='/home/phahn/datasets/birdset/HSN', # specify your data directory!
            dataset_name='HSN',
            hf_path='DBD-research-group/BirdSet',
            hf_name='HSN',#n_classes=21,
            n_workers=3,
            val_split=0.2,
            task="multilabel",
            classlimit=500,
            eventlimit=5,
            sampling_rate=32000,
        ),
    )

    # prepare the data (download dataset, ...)
    dm.prepare_data()
    dm.setup(stage="fit")

    # Initialize Model
    logging.info(f">>> Initialize Model {args.model.name}.")
    model = build_model(args)

    # Train Model
    # TODO(Paul): Current Error is that the model wants a squared image while the image presented on top has shape 128x1024
    trainer = Trainer(max_epochs=5, accelerator="gpu", devices=1)
    trainer.fit(model=model, datamodule=dm)

    # Export results
    history = {}
    with open(os.path.join(args.path.output_dir, 'results.json'), 'w') as f:
        json.dump(history, f)

    logging.info(">>> Finished.")


if __name__ == '__main__':
    main()