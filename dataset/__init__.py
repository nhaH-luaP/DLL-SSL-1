from birdset.datamodule.base_datamodule import DatasetConfig
from birdset.datamodule.birdset_datamodule import BirdSetDataModule
from birdset.datamodule.components.transforms import BirdSetTransformsWrapper

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as T

from lightning import LightningDataModule

def build_dataset(args):
    dm = BirdSetDataModule(
        dataset=DatasetConfig(
            data_dir=args.path.data_dir, # specify your data directory!
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
        transforms=BirdSetTransformsWrapper(
            task=args.task,
            model_type=args.model.type,
            #TODO: Check out what needs to be specified here.
            #spectrogram_augmentations=,
            #waveform_augmentations=
        )
    )
    return dm




class LitCIFAR10DataModule(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.247, 0.243, 0.262)

    def train_dataloader(self):
        transform = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(self.mean, self.std)])
        dataset = dataset = CIFAR10(
            root='/home/phahn/datasets/CIFAR/',
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