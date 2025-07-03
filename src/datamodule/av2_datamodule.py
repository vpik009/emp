from pathlib import Path
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader as TorchDataLoader

import torch
from torch.utils.data import random_split

from .av2_dataset import Av2Dataset, collate_fn


class Av2DataModule(LightningDataModule):
    def __init__(
        self,
        data_root: str,
        data_folder: str,
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        test_batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 8,
        pin_memory: bool = True,
        test: bool = False,
        train_fraction: float = 0.5, # fraction of train data to use
        split_seed: int = 42,         # seed for splitting
    ):
        super(Av2DataModule, self).__init__()
        self.data_root = Path(data_root)
        self.data_folder = data_folder
        self.batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.test = test
        # new params
        self.train_fraction = train_fraction
        self.split_seed = split_seed

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.test:
            full_train_dataset = Av2Dataset(
                data_root=self.data_root / self.data_folder, cached_split="train"
            )
            if self.train_fraction < 1.0:
                train_len = int(len(full_train_dataset) * self.train_fraction)
                rest_len = len(full_train_dataset) - train_len
                generator = torch.Generator().manual_seed(self.split_seed)
                self.train_dataset, _ = random_split(
                    full_train_dataset, [train_len, rest_len], generator=generator
                )
            else:
                self.train_dataset = full_train_dataset

            self.val_dataset = Av2Dataset(
                data_root=self.data_root / self.data_folder, cached_split="val"
            )
        else:
            self.test_dataset = Av2Dataset(
                data_root=self.data_root / self.data_folder, cached_split="test"
            )

    def train_dataloader(self):
        return TorchDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return TorchDataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return TorchDataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
        )