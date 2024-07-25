from model import build_model
from dataset import build_dataset
from utils import seed_everything

import os
import hydra
import json
import logging


from lightning import Trainer
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
    dm = build_dataset(args)

    # Prepare the data (download dataset and set mode according to subsequent procedure, 
    # i.e., 'fit', 'validate', 'test', or 'predict')
    dm.prepare_data()
    dm.setup(stage="fit")

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


if __name__ == '__main__':
    main()