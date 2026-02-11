from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # reproducibility
    seed: int = 42

    # dataset
    kaggle_dataset = "jangedoo/utkface-new"
    use_subdir = ""
    class_names: tuple = ("men", "women")

    # split
    train_ratio: float = 0.8
    val_ratio: float = 0.1  # test = остаток

    # images
    img_size: int = 128

    # training baseline (потом будем тюнить)
    batch_size: int = 32
    num_workers: int = 0
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # artifacts
    artifacts_dir: Path = Path("artifacts")
    models_dir: Path = Path("artifacts/models")
    logs_dir: Path = Path("artifacts/logs")
    reports_dir: Path = Path("artifacts/reports")
    