from dataclasses import dataclass
import torch


@dataclass
class EvalStats:
    loss: float
    acc: float


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> EvalStats:
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += float(loss.item()) * x.size(0)
        preds = logits.argmax(dim=1)
        correct += int((preds == y).sum().item())
        total += x.size(0)

    return EvalStats(loss=total_loss / total, acc=correct / total)