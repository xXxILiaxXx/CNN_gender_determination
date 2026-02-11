from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class UtkSample:
    path: Path
    gender_idx: int  # 0=male, 1=female


class UTKFaceGenderDataset(Dataset):
    """
    UTKFace: filename формат обычно такой:
      age_gender_race_date.jpg  (например: 25_0_0_20170116174525125.jpg)
    gender: 0=male, 1=female (в большинстве реализаций UTKFace)   [oai_citation:1‡dfighter1985](https://dfighter1985.wordpress.com/2024/05/20/converting-the-utkface-computer-vision-dataset-to-the-yolo-format/?utm_source=chatgpt.com)
    """

    def __init__(self, root: str | Path, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples: List[UtkSample] = self._index_files(self.root)

        if len(self.samples) == 0:
            raise RuntimeError(
                f"UTKFaceGenderDataset: не нашёл ни одного .jpg/.png в {self.root} (искал рекурсивно)."
            )

    def _index_files(self, root: Path) -> List[UtkSample]:
        exts = {".jpg", ".jpeg", ".png"}
        paths = [p for p in root.rglob("*") if p.suffix.lower() in exts]

        samples: List[UtkSample] = []
        for p in paths:
            name = p.stem  # без расширения
            parts = name.split("_")
            if len(parts) < 3:
                continue

            # parts[1] -> gender
            try:
                gender = int(parts[1])
            except ValueError:
                continue

            if gender not in (0, 1):
                continue

            samples.append(UtkSample(path=p, gender_idx=gender))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, s.gender_idx