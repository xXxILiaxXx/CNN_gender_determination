from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from ..config import Config
from ..data.dataset import prepare_dataloaders
from ..models.cnn import SimpleCNN
from ..training.utils import get_device


@torch.no_grad()
def main():
    cfg = Config()
    device = get_device()
    print("Device:", device)

    dls = prepare_dataloaders(cfg)

    ckpt_path = cfg.models_dir / "cnn_flat8x8.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    model = SimpleCNN(num_classes=2, dropout=0.3).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    criterion = nn.CrossEntropyLoss()

    all_preds = []
    all_true = []
    total_loss = 0.0
    total = 0
    correct = 0

    for x, y in dls.test:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        preds = logits.argmax(dim=1)

        total_loss += float(loss.item()) * x.size(0)
        total += x.size(0)
        correct += int((preds == y).sum().item())

        all_preds.extend(preds.cpu().tolist())
        all_true.extend(y.cpu().tolist())

    test_loss = total_loss / total
    test_acc = correct / total

    idx_to_class = dls.idx_to_class
    target_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    print(f"\nTEST: loss={test_loss:.4f} acc={test_acc:.4f}\n")

    print("Classification report:")
    print(classification_report(all_true, all_preds, target_names=target_names, digits=4))

    cm = confusion_matrix(all_true, all_preds)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)


if __name__ == "__main__":
    main()