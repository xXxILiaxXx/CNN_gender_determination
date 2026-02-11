import os
from dataclasses import dataclass
from typing import List

import torch
from torch.utils.data import DataLoader, Subset

import kagglehub

from ..config import Config
from .transforms import get_train_transforms, get_eval_transforms
from .utkface_dataset import UTKFaceGenderDataset


@dataclass
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    class_to_idx: dict
    idx_to_class: dict
    data_root: str


def _split_indices(n: int, seed: int, train_ratio: float, val_ratio: float):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()

    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    n_test = n - n_train - n_val

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    return train_idx, val_idx, test_idx


def prepare_dataloaders(cfg: Config) -> DataLoaders:
    # 1) download/get path (Kaggle)
    base_path = kagglehub.dataset_download(cfg.kaggle_dataset)
    data_root = os.path.join(base_path, cfg.use_subdir) if cfg.use_subdir else base_path

    # 2) создаём датасеты с разными transform (ВАЖНО: разные инстансы, а не dataset.transform=...)
    ds_train = UTKFaceGenderDataset(root=data_root, transform=get_train_transforms(cfg.img_size))
    ds_eval = UTKFaceGenderDataset(root=data_root, transform=get_eval_transforms(cfg.img_size))

    n = len(ds_eval)
    train_idx, val_idx, test_idx = _split_indices(n, cfg.seed, cfg.train_ratio, cfg.val_ratio)

    train_ds = Subset(ds_train, train_idx)
    val_ds = Subset(ds_eval, val_idx)
    test_ds = Subset(ds_eval, test_idx)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    class_to_idx = {"men": 0, "women": 1}
    idx_to_class = {0: "men", 1: "women"}

    return DataLoaders(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        data_root=data_root,
    )