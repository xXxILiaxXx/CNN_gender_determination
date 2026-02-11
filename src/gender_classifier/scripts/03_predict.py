from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from ..config import Config
from ..models.cnn import SimpleCNN
from ..training.utils import get_device
from ..data.transforms import get_eval_transforms


def load_checkpoint(ckpt_path: Path, device: torch.device):
    # В твоей версии torch может ругаться на weights_only=True, поэтому явно False
    return torch.load(ckpt_path, map_location=device, weights_only=False)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Predict gender (men/women) from a single image")
    parser.add_argument("image", type=str, help="Path to image file (jpg/png/...)")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="artifacts/models/cnn_flat8x81.pt",
        help="Path to checkpoint .pt",
    )
    args = parser.parse_args()

    img_path = Path(args.image)
    ckpt_path = Path(args.ckpt)

    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    cfg = Config()
    device = get_device()
    print("Device:", device)

    # load ckpt
    ckpt = load_checkpoint(ckpt_path, device)
    class_to_idx = ckpt.get("class_to_idx", {"men": 0, "women": 1})
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # model
    model = SimpleCNN(num_classes=2, dropout=0.0).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # transforms (как eval)
    tfm = get_eval_transforms(cfg.img_size)

    # read image
    image = Image.open(img_path).convert("RGB")
    x = tfm(image).unsqueeze(0).to(device)  # [1, 3, H, W]

    # forward
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu()  # [2]

    pred_idx = int(torch.argmax(probs).item())
    pred_class = idx_to_class.get(pred_idx, str(pred_idx))

    # красиво печатаем
    # NB: порядок вероятностей в печати — по idx_to_class
    men_idx = class_to_idx.get("men", 0)
    women_idx = class_to_idx.get("women", 1)

    print(f"\nImage: {img_path}")
    print(f"Prediction: {pred_class}")
    print(f"Probabilities:")
    print(f"  men   : {probs[men_idx].item():.4f}")
    print(f"  women : {probs[women_idx].item():.4f}")

    ru = "Мужчина" if pred_class == "men" else "Женщина"
    print(f"\nРезультат: {ru}")


if __name__ == "__main__":
    main()
