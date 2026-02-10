from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # reproducibility
    seed: int = 42

    # dataset
    kaggle_dataset: str = "playlist/men-women-classification"
    use_subdir: str = "data"          # у нас внутри versions/3 есть папка data/
    class_names: tuple = ("men", "women")

    # split
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    # test_ratio = 0.1 (остаток)

    # images
    img_size: int = 128               # стартуем с 128, потом можно 160/224
    num_channels: int = 3

    # training baseline (потом будем тюнить)
    batch_size: int = 32
    num_workers: int = 0              # для mac безопасно начать с 0
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # artifacts
    artifacts_dir: Path = Path("artifacts")
    models_dir: Path = Path("artifacts/models")
    logs_dir: Path = Path("artifacts/logs")
    reports_dir: Path = Path("artifacts/reports")