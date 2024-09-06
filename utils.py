import numpy as np
import random
import torch
import os
import torch

from lightning import Callback
from omegaconf import OmegaConf

from birdset.datamodule.base_datamodule import DatasetConfig
from birdset.datamodule.birdset_datamodule import BirdSetDataModule


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


        
def load_yaml_as_omegaconf(yaml_file_path):
    """Load a YAML file and return it as an OmegaConf object."""
    config = OmegaConf.load(yaml_file_path)
    return config



def build_dataset(args):
    dm = BirdSetDataModule(
        dataset=DatasetConfig(
            data_dir=args.path.data_dir,
            dataset_name=args.dataset.name,
            hf_path=args.path.hf_path,
            hf_name=args.dataset.name,
            n_workers=args.dataset.num_workers,
            val_split=args.dataset.val_split,
            task=args.task,
            classlimit=500,
            eventlimit=5,
            sampling_rate=32000,
        ),
    )
    return dm