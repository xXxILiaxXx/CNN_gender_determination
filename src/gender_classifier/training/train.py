from dataclasses import dataclass
import torch
import torch.nn as nn
from tqdm import tqdm


@dataclass
class TrainStats:
    loss: float
    acc: float


def train_one_epoch(model, loader, optimizer, criterion, device) -> TrainStats:
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * x.size(0)
        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += x.size(0)

        pbar.set_postfix(loss=float(loss.item()))

    return TrainStats(loss=total_loss / total, acc=correct / total)