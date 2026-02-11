from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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


def _clip_box(x1, y1, x2, y2, W, H):
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W, x2))
    y2 = max(0, min(H, y2))
    if x2 <= x1 + 1:
        x2 = min(W, x1 + 2)
    if y2 <= y1 + 1:
        y2 = min(H, y1 + 2)
    return x1, y1, x2, y2


def _expand_box(x, y, w, h, pad_ratio, W, H):
    pad = int(pad_ratio * max(w, h))
    x1 = x - pad
    y1 = y - pad
    x2 = x + w + pad
    y2 = y + h + pad
    return _clip_box(x1, y1, x2, y2, W, H)


def _center_of(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def _dist2(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


@dataclass
class Track:
    box: Tuple[int, int, int, int]          # (x1,y1,x2,y2)
    label: str                              # "men"/"women"/"no face"
    p_men: float
    p_women: float
    color: Tuple[int, int, int]             # BGR
    last_seen_frame: int
    last_pred_frame: int


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="artifacts/models/cnn_flat8x8.pt")
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--every", type=int, default=5, help="predict every N frames")
    parser.add_argument("--pad", type=float, default=0.15, help="bbox padding ratio (0..0.5)")
    parser.add_argument("--minsize", type=int, default=80, help="min face size for detector")
    parser.add_argument("--ttl", type=int, default=20, help="frames to keep track if face disappeared")
    parser.add_argument("--match", type=int, default=90, help="max center distance (px) to match track")
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

    # Haar cascade
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError(f"Cannot load Haar cascade: {cascade_path}")

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    tracks: Dict[int, Track] = {}
    next_id = 1
    frame_i = 0

    # match threshold
    match_dist2 = args.match * args.match

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_i += 1

            H, W = frame.shape[:2]

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(args.minsize, args.minsize),
            )

            detected_boxes: List[Tuple[int, int, int, int]] = []
            for (x, y, w, h) in faces:
                box = _expand_box(x, y, w, h, args.pad, W, H)
                detected_boxes.append(box)

            # --- match detections to existing tracks by nearest center ---
            used_track_ids = set()
            assigned: List[Tuple[int, Tuple[int, int, int, int]]] = []  # (track_id, box)

            for box in detected_boxes:
                c = _center_of(box)

                best_id = None
                best_d2 = None
                for tid, tr in tracks.items():
                    if tid in used_track_ids:
                        continue
                    # ignore too old tracks
                    if frame_i - tr.last_seen_frame > args.ttl:
                        continue
                    d2 = _dist2(c, _center_of(tr.box))
                    if d2 <= match_dist2 and (best_d2 is None or d2 < best_d2):
                        best_d2 = d2
                        best_id = tid

                if best_id is None:
                    tid = next_id
                    next_id += 1
                    # init new track with default "unknown"
                    tracks[tid] = Track(
                        box=box,
                        label="…",
                        p_men=0.0,
                        p_women=0.0,
                        color=(200, 200, 200),
                        last_seen_frame=frame_i,
                        last_pred_frame=-10**9,
                    )
                    assigned.append((tid, box))
                    used_track_ids.add(tid)
                else:
                    assigned.append((best_id, box))
                    used_track_ids.add(best_id)

            # update assigned tracks (box + last_seen)
            for tid, box in assigned:
                tr = tracks[tid]
                tr.box = box
                tr.last_seen_frame = frame_i

            # cleanup old tracks
            to_del = [tid for tid, tr in tracks.items() if frame_i - tr.last_seen_frame > args.ttl]
            for tid in to_del:
                del tracks[tid]

            # --- predict for each visible track every N frames ---
            if frame_i % args.every == 0:
                for tid, tr in tracks.items():
                    # predict only if the track is currently visible (seen recently)
                    if frame_i - tr.last_seen_frame > 0:
                        continue

                    x1, y1, x2, y2 = tr.box
                    face = frame[y1:y2, x1:x2].copy()
                    if face.size == 0:
                        continue

                    pred, p_men, p_women = predict_face(model, tfm, device, face, class_to_idx)

                    tr.label = pred
                    tr.p_men = p_men
                    tr.p_women = p_women
                    tr.last_pred_frame = frame_i

                    if pred == "women":
                        tr.color = (180, 105, 255)  # BGR pink-ish
                    else:
                        tr.color = (120, 220, 80)   # BGR green-ish

            # --- draw ---
            for tid, tr in tracks.items():
                # draw only currently visible tracks
                if frame_i - tr.last_seen_frame > 0:
                    continue

                x1, y1, x2, y2 = tr.box
                color = tr.color

                if tr.label == "women":
                    txt = f"ID {tid}: women ({tr.p_women:.2f})"
                elif tr.label == "men":
                    txt = f"ID {tid}: men ({tr.p_men:.2f})"
                else:
                    txt = f"ID {tid}: ..."

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    txt,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Gender Classifier (webcam)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()