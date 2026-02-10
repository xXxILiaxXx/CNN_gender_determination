from __future__ import annotations

import json
from pathlib import Path

import torch

from ..config import Config
from ..models.cnn import SimpleCNN


def main():
    cfg = Config()

    ckpt_path = cfg.models_dir / "cnn_flat8x8.pt"
    out_dir = Path("artifacts/exports")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_model_path = out_dir / "cnn_flat8x8_torchscript.pt"
    out_meta_path = out_dir / "cnn_flat8x8_meta.json"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Экспорт лучше делать на CPU (и стабильнее, и совместимее)
    device = torch.device("cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = SimpleCNN(num_classes=2, dropout=0.0).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # dummy input фиксирует форму входа: [B, 3, 128, 128]
    dummy = torch.randn(1, 3, cfg.img_size, cfg.img_size, device=device)

    traced = torch.jit.trace(model, dummy)
    traced.save(str(out_model_path))

    meta = {
        "img_size": cfg.img_size,
        "class_to_idx": ckpt.get("class_to_idx", {"men": 0, "women": 1}),
        "input_format": "RGB",
        "input_tensor_shape": [1, 3, cfg.img_size, cfg.img_size],
        "notes": "TorchScript traced SimpleCNN (Conv-BN-ReLU-MaxPool x4 + Flatten + MLP head)",
    }
    out_meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved TorchScript:", out_model_path)
    print("Saved meta:", out_meta_path)


if __name__ == "__main__":
    main()