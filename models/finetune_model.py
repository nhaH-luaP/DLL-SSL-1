from models.mixup import Mixup
from utils import TopKAccuracy

import torch
import torchaudio
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
        self.accuracy_fn = TopKAccuracy()
        self.auroc_fn = MultilabelAUROC(num_labels=num_classes)
        self.cmap_fn = MultilabelAveragePrecision(num_labels=num_classes, threshold=None, average="macro")

        # Set trainable params for finetuning
        self.model.requires_grad_(False)
        self.linear_classifier.requires_grad_(True)

    def training_step(self, batch, batch_idx):
        # Perform Mixup and then get the logits
        x, y = batch
        if x.shape[0] % 2 == 0: # Important due to mixup workflow to have an evenly shaped batch
            x, y = self.mixup_fn(x, y)
        logits = self.get_logits(x)

        # Calculate Loss
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
        probas = torch.nn.functional.sigmoid(logits)

        # Logging
        self.log_dict({
            'train/loss': loss.item(), 
            'train/max_proba': torch.max(probas).item(), 
            'train/min_proba': torch.min(probas).item()
        })

        # Return Loss for optimization
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Get logits
        x, y = batch
        logits = self.get_logits(x)

        # Calculate Loss
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)

        # Calculate metrics
        test_acc, hamming_score, mAP, cmAP, auroc  = self.calculate_metrics(logits, y)

        # Logging
        self.log_dict({
            'val/loss': loss, 
            'val/acc': test_acc, 
            'val/hamming_score': hamming_score, 
            'val/mAP': mAP, 
            'val/cmAP': cmAP, 
            'val/AUROC': auroc
        })
    
    def test_step(self, batch, batch_idx):
        # Get logits
        x, y = batch
        logits = self.get_logits(x)

        # Calculate Loss
        loss = nn.functional.binary_cross_entropy_with_logits(logits, y)

        # Calculate metrics
        test_acc, hamming_score, mAP, cmAP, auroc = self.calculate_metrics(logits, y)

        # Logging
        self.log_dict({
            'test/loss': loss, 
            'test/acc': test_acc, 
            'test/hamming_score': hamming_score, 
            'test/mAP': mAP, 'test/cmAP': cmAP, 
            'test/AUROC': auroc
        })

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.optim_params["learning_rate"], weight_decay=self.optim_params["weight_decay"], nesterov=True, momentum=0.9)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.optim_params["n_epochs"])
        return [optimizer], [lr_scheduler]
    
    def get_logits(self, wav):
        target_length = 1024
        source = wav
        
        n_frames = source.shape[1]
        diff = target_length - n_frames
        if diff > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, diff)) 
            source = m(source)
            
        elif diff < 0:
            source = source[:,0:target_length, :]
                    
        features = self._extract_features(source)

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
    
    def calculate_metrics(self, logits, y):
        probas = torch.nn.functional.sigmoid(logits)
        preds = (probas >= 0.5).cpu().numpy().astype(int)

        test_acc = self.accuracy_fn(probas, y)
        hamming_score = self._calculate_hamming_score(y_pred=preds, y_true=y.cpu().numpy().astype(int))
        mAP, _ = self._calculate_mAP(target=y.cpu(), output=probas.cpu())
        cmAP = self.cmap_fn(logits, y.long())
        auroc = self.auroc_fn(probas, y.long())
        return test_acc, hamming_score, mAP, cmAP, auroc
    

    def _extract_features(self, source):
        granularity = 'utterance'
        with torch.no_grad():
            # source = source.unsqueeze(dim=0) #btz=1
            if granularity == 'all':
                feats = self.model.extract_features(source, padding_mask=None,mask=False, remove_extra_tokens=False)
                feats = feats['x']
            elif granularity == 'frame':
                feats = self.model.extract_features(source, padding_mask=None,mask=False, remove_extra_tokens=True)
                feats = feats['x']
            elif granularity == 'utterance':
                feats = self.model.extract_features(source, padding_mask=None,mask=False, remove_extra_tokens=False)
                feats = feats['x']
                feats = feats[:, 0]
            else:
                raise ValueError("Unknown granularity: {}".format(granularity))

        return feats