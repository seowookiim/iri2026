from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

from soundclf.data import AudioDataset


def collate_drop_none(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return default_collate(batch)


class WaveformESC50Dataset(Dataset):
    """ESC-50 dataset that returns raw waveform as (1, 1, T)."""

    def __init__(
        self,
        csv_path: str,
        root: str,
        sample_rate: int,
        clip_seconds: float,
        training: bool = False,
        waveform_augment: bool = False,
    ):
        self.root = Path(root)
        self.sample_rate = int(sample_rate)
        self.clip_samples = int(float(clip_seconds) * self.sample_rate)
        self.training = bool(training)
        self.waveform_augment = bool(waveform_augment and training)
        self.rows: List[Dict[str, str]] = []
        self._resamplers: Dict[int, torchaudio.transforms.Resample] = {}

        with Path(csv_path).open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.rows.append(row)
        if not self.rows:
            raise ValueError(f"Empty csv: {csv_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def _resample(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        if sr == self.sample_rate:
            return waveform
        if sr not in self._resamplers:
            self._resamplers[sr] = torchaudio.transforms.Resample(sr, self.sample_rate)
        return self._resamplers[sr](waveform)

    def _crop_or_pad(self, waveform: torch.Tensor) -> torch.Tensor:
        length = waveform.size(-1)
        target = self.clip_samples
        if length >= target:
            if self.training:
                start = random.randint(0, length - target)
            else:
                start = (length - target) // 2
            return waveform[..., start : start + target]
        return F.pad(waveform, (0, target - length), mode="constant", value=0.0)

    def _augment(self, waveform: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.8:
            waveform = waveform * random.uniform(0.8, 1.2)
        if random.random() < 0.5:
            waveform = waveform + torch.randn_like(waveform) * random.uniform(0.001, 0.01)
        return waveform.clamp(-1.0, 1.0)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = self.root / row["path"]
        wave, sr = torchaudio.load(str(path))
        if wave.size(0) > 1:
            wave = wave.mean(dim=0, keepdim=True)
        wave = self._resample(wave, sr)
        wave = self._crop_or_pad(wave)
        if self.waveform_augment:
            wave = self._augment(wave)

        wave = wave.unsqueeze(1)  # (1, 1, T)
        y = torch.tensor(int(row["class_index"]), dtype=torch.long)
        return wave, y


def infer_input_shape(dataset: Dataset, max_scan: int = 64) -> Tuple[int, ...]:
    scan = min(len(dataset), max_scan)
    for idx in range(scan):
        item = dataset[idx]
        if item is None:
            continue
        x, _ = item
        return (1, *tuple(x.shape))
    raise RuntimeError("Could not infer input shape from dataset")


def build_dataset(
    data_cfg: dict,
    model_cfg: dict,
    csv_path: str,
    training: bool,
) -> Dataset:
    input_rep = (model_cfg.get("input_representation", "mel") or "mel").lower()
    if input_rep in {"wave", "waveform", "raw"}:
        return WaveformESC50Dataset(
            csv_path=csv_path,
            root=data_cfg["root"],
            sample_rate=int(data_cfg.get("sample_rate", 16000)),
            clip_seconds=float(data_cfg.get("clip_seconds", 5.0)),
            training=training,
            waveform_augment=bool(data_cfg.get("waveform_augment", True)),
        )

    return AudioDataset(
        csv_path=csv_path,
        root=data_cfg["root"],
        sample_rate=int(data_cfg.get("sample_rate", 16000)),
        clip_seconds=float(data_cfg.get("clip_seconds", 5.0)),
        n_mels=int(model_cfg.get("n_mels", 128)),
        n_fft=int(model_cfg.get("n_fft", 1024)),
        hop_length=int(model_cfg.get("hop_length", 256)),
        win_length=int(model_cfg.get("win_length", model_cfg.get("n_fft", 1024))),
        training=training,
        fixed_length=True,
        segment_long=False,
        paper_augment=bool(data_cfg.get("paper_augment", False)) if training else False,
        waveform_aug=bool(data_cfg.get("waveform_augment", True)) if training else False,
        specaugment=bool(data_cfg.get("specaugment", True)) if training else False,
        time_frames=data_cfg.get("time_frames", None),
    )


def build_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    training: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(training),
        num_workers=int(num_workers),
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_drop_none,
    )


def discover_folds(folds_dir: str) -> List[Dict[str, str]]:
    base = Path(folds_dir)
    if not base.exists():
        raise FileNotFoundError(f"folds_dir not found: {folds_dir}")

    folds: List[Dict[str, str]] = []
    seen = set()

    for train_csv in sorted(base.glob("*_train.csv")):
        prefix = train_csv.name[: -len("_train.csv")]
        val_csv = base / f"{prefix}_val.csv"
        test_csv = base / f"{prefix}_test.csv"
        if val_csv.exists() and test_csv.exists():
            folds.append(
                {
                    "name": prefix,
                    "train_csv": str(train_csv),
                    "val_csv": str(val_csv),
                    "test_csv": str(test_csv),
                }
            )
            seen.add(prefix)

    for fold_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
        name = fold_dir.name
        if name in seen:
            continue
        train_csv = fold_dir / "train.csv"
        val_csv = fold_dir / "val.csv"
        test_csv = fold_dir / "test.csv"
        if train_csv.exists() and val_csv.exists() and test_csv.exists():
            folds.append(
                {
                    "name": name,
                    "train_csv": str(train_csv),
                    "val_csv": str(val_csv),
                    "test_csv": str(test_csv),
                }
            )

    if not folds:
        raise RuntimeError(
            f"No folds discovered in {folds_dir}. Expected *_train.csv layout or fold subdirectories."
        )
    return folds

