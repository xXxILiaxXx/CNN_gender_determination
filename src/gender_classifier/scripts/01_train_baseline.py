from __future__ import annotations

import torch
import torch.nn as nn
from collections import Counter

from ..config import Config
from ..data.dataset import prepare_dataloaders
from ..models.cnn import SimpleCNN
from ..training.utils import set_seed, get_device
from ..training.train import train_one_epoch
from ..training.eval import evaluate


def _collect_labels_from_dataset(ds) -> list[int]:
    """
    Универсально собирает метки из датасета.
    Работает и для torchvision ImageFolder (есть .targets),
    и для кастомных Dataset (метка возвращается из __getitem__).
    """
    # random_split -> Subset
    if hasattr(ds, "dataset") and hasattr(ds, "indices"):
        base = ds.dataset
        idxs = ds.indices

        if hasattr(base, "targets"):
            return [int(base.targets[i]) for i in idxs]

        labels = []
        for i in idxs:
            _, y = base[i]
            labels.append(int(y))
        return labels

    # обычный Dataset
    if hasattr(ds, "targets"):
        return [int(x) for x in ds.targets]

    labels = []
    for i in range(len(ds)):
        _, y = ds[i]
        labels.append(int(y))
    return labels


def main():
    cfg = Config()
    cfg.models_dir.mkdir(parents=True, exist_ok=True)

    # epochs задаём отдельно (Config frozen)
    epochs = 5

    set_seed(cfg.seed)
    device = get_device()
    print("Device:", device)

    dls = prepare_dataloaders(cfg)

    # ===== class weights по train =====
    train_ds = dls.train.dataset
    train_labels = _collect_labels_from_dataset(train_ds)

    counts = Counter(train_labels)
    total = sum(counts.values())

    # если вдруг какого-то класса нет в train
    for cls in range(2):
        counts.setdefault(cls, 1)

    class_weights = torch.tensor(
        [total / counts[i] for i in range(2)],
        dtype=torch.float32,
        device=device
    )

    print("Train class counts:", dict(counts))
    print("Class weights:", [float(x) for x in class_weights.detach().cpu().tolist()])

    # ===== model / loss / optim =====
    model = SimpleCNN(num_classes=2, dropout=0.3).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    best_val_acc = 0.0
    best_path = cfg.models_dir / "cnn_flat8x8.pt"

    # ===== train loop =====
    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(model, dls.train, optimizer, criterion, device)
        va = evaluate(model, dls.val, criterion, device)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train: loss={tr.loss:.4f} acc={tr.acc:.4f} | "
            f"val: loss={va.loss:.4f} acc={va.acc:.4f}"
        )

        if va.acc > best_val_acc:
            best_val_acc = va.acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_to_idx": dls.class_to_idx,
                    "meta": {
                        "img_size": cfg.img_size,
                        "batch_size": cfg.batch_size,
                        "lr": cfg.lr,
                        "weight_decay": cfg.weight_decay,
                        "seed": cfg.seed,
                        "epochs": epochs,
                        "class_weights": class_weights.detach().cpu().tolist(),
                        "train_class_counts": dict(counts),
                    },
                },
                best_path
            )
            print(f"  saved best -> {best_path} (val_acc={best_val_acc:.4f})")

    print("Done. Best val acc:", best_val_acc)


if __name__ == "__main__":
    main()