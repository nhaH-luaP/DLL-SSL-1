from models.mixup import Mixup
from utils import TopKAccuracy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MultilabelAUROC
from torchmetrics.classification.average_precision import MultilabelAveragePrecision

import lightning as L

import numpy as np

from sklearn.metrics import average_precision_score


class EATFairseqModule(L.LightningModule):
    def __init__(self, model, linear_classifier, num_classes, prediction_mode="mean_pooling", optim_params={"weight_decay":5e-4, "learning_rate":1e-1, "n_epochs":1}):
        super().__init__()
        self.model = model
        self.linear_classifier = linear_classifier
        self.prediction_mode = prediction_mode
        self.optim_params = optim_params
        self.mixup_fn = Mixup(
                mixup_alpha=0.5,
                cutmix_alpha=0.5,
                cutmix_minmax=None,
                prob=1.0,
                switch_prob=0.0,
                mode="batch",
                label_smoothing=0.0,
                num_classes=num_classes,
            )
        self.accuracy_fn = nn.ModuleList([TopKAccuracy() for _ in range(2)])
        self.auroc_fn = nn.ModuleList([MultilabelAUROC(num_labels=num_classes) for _ in range(2)])
        self.cmap_fn = nn.ModuleList([MultilabelAveragePrecision(num_labels=num_classes, threshold=None, average="macro") for _ in range(2)])

        # Set trainable params for finetuning
        self.model.requires_grad_(False)
        self.linear_classifier.requires_grad_(True)

    def training_step(self, batch, batch_idx):
        # Perform Mixup and then get the logits
        x, y = batch['input_values'], batch['labels']
        if x.shape[0] % 2 == 0: # Important due to mixup workflow to have an evenly shaped batch
            x, y = self.mixup_fn(x, y)
        logits = self.get_logits(x)

        # Calculate Loss
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)

        # Logging
        self.log_dict({'train/loss': loss.item()})

        # Return Loss for optimization
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Get logits
        x, y = batch['input_values'], batch['labels']
        logits = self.get_logits(x)

        # Calculate Loss
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)

        # Calculate metrics
        test_acc, hamming_score, mAP, cmAP, auroc  = self.calculate_metrics(logits, y, mode='val')

        # Logging
        self.log_dict({'val/loss': loss, 'val/acc': test_acc, 'val/hamming_score': hamming_score, 'val/mAP': mAP, 'val/cmAP': cmAP, 'val/AUROC': auroc})
    
    def test_step(self, batch, batch_idx):
        # Get logits
        x, y = batch['input_values'], batch['labels']
        logits = self.get_logits(x)

        # Calculate Loss
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)

        # Calculate metrics
        test_acc, hamming_score, mAP, cmAP, auroc = self.calculate_metrics(logits, y, mode='test')

        # Logging
        self.log_dict({'test/loss': loss, 'test/acc': test_acc, 'test/hamming_score': hamming_score, 'test/mAP': mAP, 'test/cmAP': cmAP, 'test/AUROC': auroc})

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.optim_params["learning_rate"], weight_decay=self.optim_params["weight_decay"], nesterov=True, momentum=0.9)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.optim_params["n_epochs"])
        return [optimizer], [lr_scheduler]
    
    def get_logits(self, x):
        with torch.no_grad():
            result = self.model(x, features_only=True, remove_extra_tokens=True, mask=False) #TODO: What about the remove extra tokens option?
        features = result['x']

        # Different prediction modes to work with the features resulting from the fairseq model. Shape: (Batch-Size, Dim1, Dim2)
        features = self.reduce_features(features)

        logits = self.linear_classifier(features)
        return logits
    
    def _calculate_mAP(self, output, target):
        classes_num = target.shape[-1]
        ap_values = {}
        for k in range(classes_num):
            avg_precision = average_precision_score(target[:, k], output[:, k], average=None)
            ap_values[k] = avg_precision
        mean_ap = np.nanmean(list(ap_values.values()))
        return mean_ap, ap_values
    
    def _calculate_hamming_score(self, y_true, y_pred):
        return (
            (y_true & y_pred).sum(axis=1) / (y_true | y_pred).sum(axis=1)
        ).mean()
    
    def calculate_metrics(self, logits, y, mode):
        if mode == 'val':
            mode = 0
        elif mode == 'test':
            mode = 1
        else:
            raise Exception(f"unknown mode {self.mode}")

        probas = torch.nn.functional.sigmoid(logits)
        preds = (probas >= 0.5).cpu().numpy().astype(int)

        test_acc = self.accuracy_fn[mode](probas, y)
        hamming_score = self._calculate_hamming_score(y_pred=preds, y_true=y.cpu().numpy().astype(int))
        mAP, _ = self._calculate_mAP(target=y.cpu(), output=probas.cpu())
        cmAP = self.cmap_fn[mode](logits, y.long())
        auroc = self.auroc_fn[mode](probas, y.long())
        return test_acc, hamming_score, mAP, cmAP, auroc

    def reduce_features(self, features):
        if self.prediction_mode == "mean_pooling":
            features = features.mean(dim=1)
        elif self.prediction_mode == "cls_token":
            features = features[:, 0]
        elif self.prediction_mode == "lin_softmax":
            dtype = features.dtype
            features = F.logsigmoid(features.float())
            features = torch.logsumexp(features + features, dim=1) - torch.logsumexp(features + 1e-6, dim=1)
            features = features.clamp(max=0)
            features = features - torch.log(-(torch.expm1(features)))
            features = torch.nan_to_num(features, nan=0, posinf=0, neginf=0)
            features = features.to(dtype=dtype)
        else:
            raise Exception(f"unknown prediction mode {self.prediction_mode}")
        return features