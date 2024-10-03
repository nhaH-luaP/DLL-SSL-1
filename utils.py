import numpy as np
import random
import os
import torch
import torchmetrics

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


# From: https://github.com/DBD-research-group/BirdSet/blob/main/birdset/modules/metrics/multilabel.py
class TopKAccuracy(torchmetrics.Metric):
    def __init__(self, topk=1, include_nocalls=False, threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self.topk = topk
        self.include_nocalls = include_nocalls
        self.threshold = threshold
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        
    def update(self, preds, targets):
        # Get the top-k predictions
        _, topk_pred_indices = preds.topk(self.topk, dim=1, largest=True, sorted=True)
        targets = targets.to(preds.device)
        no_call_targets = targets.sum(dim=1) == 0

        #consider no_call instances (a threshold is needed here!)
        if self.include_nocalls:
            #check if top-k predictions for all-negative instances are less than threshold 
            no_positive_predictions = preds.topk(self.topk, dim=1, largest=True).values < self.threshold
            correct_all_negative = (no_call_targets & no_positive_predictions.all(dim=1))

        else:
            #no_calls are removed, set to 0
            correct_all_negative = torch.tensor(0).to(targets.device)

        #convert one-hot encoded targets to class indices for positive cases
        expanded_targets = targets.unsqueeze(1).expand(-1, self.topk, -1)
        correct_positive = expanded_targets.gather(2, topk_pred_indices.unsqueeze(-1)).any(dim=1)
        
        #update correct and total, excluding all-negative instances if specified
        self.correct += correct_positive.sum() + correct_all_negative.sum()
        if not self.include_nocalls:
            self.total += targets.size(0) - no_call_targets.sum()
        else:
            self.total += targets.size(0)

    def compute(self):
        return self.correct.float() / self.total

        
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
            classlimit=args.dataset.classlimit,
            eventlimit=args.dataset.eventlimit,
            sampling_rate=args.dataset.sampling_rate,
        ),
    )
    return dm