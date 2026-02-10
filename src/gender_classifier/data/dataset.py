import os
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

import kagglehub

from ..config import Config
from .transforms import get_train_transforms, get_eval_transforms


@dataclass
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    class_to_idx: dict
    idx_to_class: dict
    data_root: str


def prepare_dataloaders(cfg: Config) -> DataLoaders:
    # 1) download/get path
    base_path = kagglehub.dataset_download(cfg.kaggle_dataset)
    data_root = os.path.join(base_path, cfg.use_subdir)

    # 2) full dataset without augmentation first (we will override transforms per split)
    full_eval = datasets.ImageFolder(root=data_root, transform=get_eval_transforms(cfg.img_size))

    # 3) split
    n = len(full_eval)
    n_train = int(cfg.train_ratio * n)
    n_val = int(cfg.val_ratio * n)
    n_test = n - n_train - n_val

    g = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds, test_ds = random_split(full_eval, [n_train, n_val, n_test], generator=g)

    # 4) IMPORTANT: set train transform with augmentation
    # random_split returns Subset, we can change underlying dataset transform:
    train_ds.dataset.transform = get_train_transforms(cfg.img_size)
    val_ds.dataset.transform = get_eval_transforms(cfg.img_size)
    test_ds.dataset.transform = get_eval_transforms(cfg.img_size)

    # 5) loaders
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    class_to_idx = full_eval.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    return DataLoaders(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        data_root=data_root
    )