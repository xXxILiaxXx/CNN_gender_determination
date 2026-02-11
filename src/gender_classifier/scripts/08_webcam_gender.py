from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from PIL import Image

from ..config import Config
from ..models.cnn import SimpleCNN
from ..training.utils import get_device
from ..data.transforms import get_eval_transforms


def load_checkpoint(ckpt_path: Path):
    return torch.load(ckpt_path, map_location="cpu", weights_only=False)


@torch.no_grad()
def predict_face(model, tfm, device, face_bgr, class_to_idx):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(face_rgb).convert("RGB")
    x = tfm(pil).unsqueeze(0).to(device)

    logits = model(x)
    probs = F.softmax(logits, dim=1).squeeze(0).detach().cpu()

    men_idx = class_to_idx.get("men", 0)
    women_idx = class_to_idx.get("women", 1)

    p_men = float(probs[men_idx].item())
    p_women = float(probs[women_idx].item())
    pred = "women" if p_women >= p_men else "men"
    return pred, p_men, p_women


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="artifacts/models/cnn_flat8x8.pt")
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--every", type=int, default=5, help="predict every N frames")
    args = parser.parse_args()

    cfg = Config()
    device = get_device()
    print("Device:", device)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = load_checkpoint(ckpt_path)
    class_to_idx = ckpt.get("class_to_idx", {"men": 0, "women": 1})

    model = SimpleCNN(num_classes=2, dropout=0.0)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    tfm = get_eval_transforms(cfg.img_size)

    # Haar cascade (входит в OpenCV)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError(f"Cannot load Haar cascade: {cascade_path}")

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    last_text = "no face"
    last_color = (200, 200, 200)
    frame_i = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_i += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
            )

            if len(faces) > 0:
                # самое большое лицо
                x, y, w, h = max(faces, key=lambda b: b[2] * b[3])

                # паддинг
                pad = int(0.15 * max(w, h))
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(frame.shape[1], x + w + pad)
                y2 = min(frame.shape[0], y + h + pad)

                face = frame[y1:y2, x1:x2].copy()

                if face.size > 0 and (frame_i % args.every == 0):
                    pred, p_men, p_women = predict_face(model, tfm, device, face, class_to_idx)
                    if pred == "women":
                        last_text = f"women ({p_women:.2f})"
                        last_color = (180, 105, 255)  # BGR pink-ish
                    else:
                        last_text = f"men ({p_men:.2f})"
                        last_color = (120, 220, 80)   # BGR green-ish

                cv2.rectangle(frame, (x1, y1), (x2, y2), last_color, 2)
                cv2.putText(
                    frame, last_text, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, last_color, 2, cv2.LINE_AA
                )
            else:
                last_text = "no face"
                last_color = (200, 200, 200)

            cv2.imshow("Gender Classifier (webcam)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()