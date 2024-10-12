import torchmetrics.classification
from models.mixup import Mixup
from utils import TopKAccuracy, FocalLoss

import numpy as np
import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
from torchmetrics.classification import (
    MultilabelAccuracy,
    MultilabelAUROC,
    MultilabelAveragePrecision,
    MultilabelRecall
)


class EATFairseqModule(L.LightningModule):
    def __init__(
            self, 
            model, 
            linear_classifier, 
            num_classes, 
            optim_params={"weight_decay":5e-4, "learning_rate":1e-1, "n_epochs":1},
            granularity='utterance',
            pos_weight=None,
            loss='BCE'
        ):
        super().__init__()
        self.model = model
        self.linear_classifier = linear_classifier
        self.optim_params = optim_params
        self.granularity = granularity
        self.num_classes = num_classes
        
        # Set trainable params for finetuning
        self.model.requires_grad_(False)
        self.linear_classifier.requires_grad_(True)

        # Init loss function and metrics
        if loss == 'BCE':
            if pos_weight is not None:
                pos_weight = torch.ones(num_classes) * pos_weight
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif loss == 'focal':
            self.loss_fn = FocalLoss(alpha=2, gamma=2, reduction='mean')
        else:
            raise ValueError("Unknown loss function: {}".format(loss))
        
        self._init_metrics()

        # Init weights
        self.linear_classifier.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)


    def _init_metrics(self):
        # Torchmetric instances cannot be used over multiple stages
        # Train metrics
        self.train_acc = MultilabelAccuracy(num_labels=self.num_classes)
        self.train_auroc = MultilabelAUROC(num_labels=self.num_classes)
        self.train_map = MultilabelAveragePrecision(num_labels=self.num_classes)
        self.train_recall = MultilabelRecall(num_labels=self.num_classes)

        # Validation metrics
        self.val_acc = MultilabelAccuracy(num_labels=self.num_classes)
        self.val_auroc = MultilabelAUROC(num_labels=self.num_classes)
        self.val_map = MultilabelAveragePrecision(num_labels=self.num_classes, average="macro")
        self.val_recall = MultilabelRecall(num_labels=self.num_classes)

        # Test metrics
        self.test_acc = MultilabelAccuracy(num_labels=self.num_classes)
        self.test_auroc = MultilabelAUROC(num_labels=self.num_classes)
        self.test_map = MultilabelAveragePrecision(num_labels=self.num_classes, average="macro")
        self.test_recall = MultilabelRecall(num_labels=self.num_classes)

        
    def training_step(self, batch, batch_idx):
        # Perform Mixup and then get the logits
        x, y = batch
        logits = self.get_logits(x)

        # Calculate Loss
        loss = self.loss_fn(logits, y)
        probas = torch.nn.functional.sigmoid(logits)
        preds = (probas >= 0.5).int()

        # Calculate metrics
        y = y.int()
        self.train_acc(preds, y)
        self.train_auroc(probas, y)
        self.train_map(probas, y)
        self.train_recall(probas, y)

        # Logging
        self.log_dict({
            'train/loss': loss.item(),
            'train/acc': self.train_acc,
            'train/AUROC': self.train_auroc,
            'train/mAP': self.train_map,
            'train/recall': self.train_recall,
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
        loss = self.loss_fn(logits, y)
        probas = torch.nn.functional.sigmoid(logits)
        preds = (probas >= 0.5).int()

        # Calculate metrics
        y = y.int()
        self.val_acc(preds, y)
        self.val_auroc(probas, y)
        self.val_map(probas, y)
        self.val_recall(probas, y)

        # Logging
        self.log_dict({
            'val/loss': loss.item(),
            'val/acc': self.val_acc,
            'val/AUROC': self.val_auroc,
            'val/mAP': self.val_map,
            'val/recall': self.val_recall,
        })
    

    def test_step(self, batch, batch_idx):
        # Get logits
        x, y = batch
        logits = self.get_logits(x)

        # Calculate Loss
        loss = self.loss_fn(logits, y)
        probas = torch.nn.functional.sigmoid(logits)
        preds = (probas >= 0.5).int()

        # Calculate metrics
        y = y.int()
        self.test_acc(preds, y)
        self.test_auroc(probas, y)
        self.test_map(probas, y)
        self.test_recall(probas, y)

        # Logging
        self.log_dict({
            'test/loss': loss.item(), 
            'test/acc': self.test_acc,
            'test/AUROC': self.test_auroc,
            'test/mAP': self.test_map,
            'test/recall': self.test_recall,
        })


    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.optim_params["learning_rate"], weight_decay=self.optim_params["weight_decay"], nesterov=True, momentum=0.9)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.optim_params["n_epochs"])
        return [optimizer], [lr_scheduler]
    

    def get_logits(self, source):
        target_length = 1024
        source = source
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
        with torch.no_grad():
            # source = source.unsqueeze(dim=0) #btz=1
            if self.granularity == 'all':
                feats = self.model.extract_features(source, padding_mask=None,mask=False, remove_extra_tokens=False)
                feats = feats['x']
            elif self.granularity == 'frame':
                feats = self.model.extract_features(source, padding_mask=None,mask=False, remove_extra_tokens=True)
                feats = feats['x']
            elif self.granularity == 'utterance':
                feats = self.model.extract_features(source, padding_mask=None,mask=False, remove_extra_tokens=False)
                feats = feats['x']
                feats = feats[:, 0]
            else:
                raise ValueError("Unknown granularity: {}".format(self.granularity))

        return feats