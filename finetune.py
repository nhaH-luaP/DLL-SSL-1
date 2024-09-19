from utils import seed_everything
from utils import MetricsCallback, load_yaml_as_omegaconf, build_dataset
from models import build_pretrained_model

import os
import json
import logging

from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf



def main():
    # Load args manually from file to avoid clash between hydra and fairseq
    args = load_yaml_as_omegaconf(yaml_file_path="./config.yaml")
    logging.info('>>> Using config: \n%s', OmegaConf.to_yaml(args))

    # Create directory for results
    logging.info(">>> Creating output directory at: "+str(args.path.output_dir))
    os.makedirs(args.path.output_dir, exist_ok=True)

    # Enable reproducability
    logging.info(f">>> Seed experiment with random seed {args.random_seed}.")
    seed_everything(args.random_seed + 42)

    # Init wandb logger
    wandb_logger = WandbLogger(
        project=args.wandb.project,
        name=args.wandb.run_name
    )
    
    # Initialize Dataset and set to fit
    logging.info(f">>> Initialize Dataset {args.dataset}.")
    dm = build_dataset(args)
    dm.prepare_data()
    dm.setup(stage="fit")

    # Initialize Model
    logging.info(f">>> Initialize Model.")
    model = build_pretrained_model(args)

    # Initialize callback for keeping track of metrics
    metrics_callback = MetricsCallback()

    # Finetune Model
    trainer = Trainer(max_epochs=args.model.num_epochs, callbacks=[metrics_callback], logger=wandb_logger)
    trainer.fit(model=model, datamodule=dm)

    # Evaluate Model
    dm.setup(stage='test')
    trainer.test(model=model, datamodule=dm)

    # Extract metrics and export into json file
    metrics_dict = {'train_metrics': metrics_callback.train_metrics, 'test_metrics': metrics_callback.test_metrics}
    with open(os.path.join(args.path.output_dir, 'results.json'), 'w') as f:
        json.dump(metrics_dict, f)

    logging.info(">>> Finished.")


if __name__ == '__main__':
    main()