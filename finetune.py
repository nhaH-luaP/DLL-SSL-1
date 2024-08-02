from model import build_model
from dataset import build_dataset
from utils import seed_everything

import os
import hydra
import json
import logging
import torch

from lightning import Trainer
from lightning import Callback
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
    model = build_pretrained_model(args)

    # Initialize callback for keeping track of metrics
    metrics_callback = MetricsCallback()

    # Finetune Model
    trainer = Trainer(max_epochs=args.model.num_epochs, callbacks=[metrics_callback])
    trainer.fit(model=model, datamodule=dm)

    # Evaluate Model
    dm.setup(stage='test')
    trainer.test(model=model, datamodule=dm)

    # Extract metrics and export into json file
    metrics_dict = {'train_metrics':metrics_callback.train_metrics,  'val_metrics':metrics_callback.val_metrics, 'test_metrics':metrics_callback.test_metrics}
    with open(os.path.join(args.path.output_dir, 'results.json'), 'w') as f:
        json.dump(metrics_dict, f)

    logging.info(">>> Finished.")


def build_pretrained_model(args):
    model = build_model(args)
    model.load_state_dict(torch.load(args.path.pretrained_weights_dir))
    model.prepare_finetuning()
    return model


class MetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.train_metrics = []
        self.val_metrics = []
        self.test_metrics = []

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_metrics.append({k:i.item() for k,i in trainer.callback_metrics.items()})

    def on_val_epoch_end(self, trainer, pl_module):
        self.val_metrics.append({k:i.item() for k,i in trainer.callback_metrics.items()})

    def on_test_epoch_end(self, trainer, pl_module):
        self.test_metrics.append({k:i.item() for k,i in trainer.callback_metrics.items()})


if __name__ == '__main__':
    main()