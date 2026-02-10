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

    out_onnx_path = out_dir / "cnn_flat8x8.onnx"
    out_meta_path = out_dir / "cnn_flat8x8_onnx_meta.json"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = SimpleCNN(num_classes=2, dropout=0.0).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dummy = torch.randn(1, 3, cfg.img_size, cfg.img_size, device=device)

    # Экспорт ONNX
    torch.onnx.export(
        model,
        dummy,
        str(out_onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
    )

    meta = {
        "img_size": cfg.img_size,
        "class_to_idx": ckpt.get("class_to_idx", {"men": 0, "women": 1}),
        "input_format": "RGB",
        "input_tensor_shape": ["batch", 3, cfg.img_size, cfg.img_size],
        "output": "logits (before softmax), shape: [batch, 2]",
        "opset": 17,
    }
    out_meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Saved ONNX:", out_onnx_path)
    print("Saved meta:", out_meta_path)


if __name__ == "__main__":
    main()