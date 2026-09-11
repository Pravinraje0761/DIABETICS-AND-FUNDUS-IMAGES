"""
PyTorch Dataset / DataLoader for the preprocessed fundus images (Phase 3).

Reads a manifest CSV produced by scripts/phase3_preprocess.py:
    id_code, diagnosis, split, ..., source_path, processed_path

Images on disk are already 224x224 uint8 RGB. This module only adds tensor
conversion, ImageNet/RETFound normalisation, and (for training) light
augmentation. Same normalisation for the baseline CNN and for RETFound.

Used by Phases 4 (baseline), 5 (RETFound), 6 (external eval), 8 (MC-dropout).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import IMAGENET_MEAN, IMAGENET_STD, RETFOUND_INPUT_SIZE  # noqa: E402

NUM_CLASSES = 5
PROJECT_DIR = Path(__file__).resolve().parents[1]

# Optional: relocate the (large, read-every-epoch) processed-image tree onto a
# fast local disk without editing manifests. If DR_DATA_ROOT is set and it holds
# the referenced relative path, it overrides PROJECT_DIR. Manifests stay portable;
# unset -> identical behaviour. Motivation: OneDrive Files-On-Demand dehydrating
# data_processed/ between epochs stalled the RETFound fine-tune (see PHASE5_REPORT).
_DATA_ROOT = os.environ.get("DR_DATA_ROOT") or None


def _resolve(path_str: str) -> Path:
    """Manifest paths are relative to PROJECT_DIR (portable); absolute is honoured.
    DR_DATA_ROOT, if set and it actually holds the file, wins over PROJECT_DIR."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    if _DATA_ROOT:
        cand = Path(_DATA_ROOT) / p
        if cand.exists():
            return cand
    return PROJECT_DIR / p


def build_transform(train: bool, size: int = RETFOUND_INPUT_SIZE) -> transforms.Compose:
    """Eval = resize-safety + normalise. Train = + light fundus-safe augmentation.

    Augmentation is deliberately mild: DR lesions are small and localised, so
    aggressive crops/distortions can remove the very signal being graded.
    """
    norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if train:
        return transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                norm,
            ]
        )
    return transforms.Compose(
        [transforms.Resize((size, size)), transforms.ToTensor(), norm]
    )


class FundusDataset(Dataset):
    """One row per gradable image with a valid processed_path.

    __getitem__ returns (image: FloatTensor[3,H,W], label: int, id_code: str).
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        split: str | None = None,
        train: bool = False,
        size: int = RETFOUND_INPUT_SIZE,
        transform: transforms.Compose | None = None,
        train_frac: float | None = None,
        frac_seed: int = 42,
    ) -> None:
        self.manifest_csv = Path(manifest_csv)
        df = pd.read_csv(self.manifest_csv)
        df["id_code"] = df["id_code"].astype(str)
        df = df[df["processed_path"].notna() & (df["processed_path"].astype(str) != "")]
        if split is not None:
            if "split" not in df.columns:
                raise ValueError(f"{self.manifest_csv} has no 'split' column")
            df = df[df["split"] == split]
        if train_frac is not None and split == "train":
            # Training-data-size sensitivity: stratified subsample of the train
            # split only (val/test/external are never touched). Fixed seed ->
            # reproducible; fractions are independently re-sampled, not nested.
            if not (0.0 < train_frac <= 1.0):
                raise ValueError(f"train_frac must be in (0, 1], got {train_frac}")
            df = df.groupby("diagnosis", group_keys=False).sample(
                frac=train_frac, random_state=frac_seed)
        df = df.reset_index(drop=True)
        if df.empty:
            raise ValueError(
                f"no usable rows in {self.manifest_csv} for split={split!r}"
            )
        paths = [_resolve(p) for p in df["processed_path"].astype(str)]
        missing = [p for p in paths[:5] if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                f"processed images referenced by {self.manifest_csv} are missing, "
                f"e.g. {missing[0]} - run scripts/phase3_preprocess.py"
            )

        self.df = df
        self.split = split
        self.transform = transform or build_transform(train=train, size=size)
        self.labels = df["diagnosis"].astype(int).tolist()
        self.id_codes = df["id_code"].tolist()
        self.paths = [str(p) for p in paths]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        with Image.open(self.paths[idx]) as im:
            img = self.transform(im.convert("RGB"))
        return img, int(self.labels[idx]), self.id_codes[idx]

    # -- helpers -------------------------------------------------------------- #
    def class_counts(self) -> torch.Tensor:
        counts = torch.zeros(NUM_CLASSES, dtype=torch.long)
        for y in self.labels:
            counts[y] += 1
        return counts

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights, mean-normalised. For a weighted CE loss."""
        counts = self.class_counts().float().clamp(min=1.0)
        w = counts.sum() / (NUM_CLASSES * counts)
        return w / w.mean()


def build_dataloader(
    manifest_csv: str | Path,
    split: str | None,
    batch_size: int = 32,
    train: bool = False,
    num_workers: int = 0,
    shuffle: bool | None = None,
    size: int = RETFOUND_INPUT_SIZE,
    train_frac: float | None = None,
    frac_seed: int = 42,
) -> DataLoader:
    ds = FundusDataset(manifest_csv, split=split, train=train, size=size,
                       train_frac=train_frac, frac_seed=frac_seed)
    if shuffle is None:
        shuffle = train
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


if __name__ == "__main__":
    # Smoke test against whatever manifests exist.
    proj = Path(__file__).resolve().parents[1]
    for name, split in [
        ("aptos_manifest.csv", "train"),
        ("aptos_manifest.csv", "val"),
        ("aptos_manifest.csv", "test"),
        ("messidor2_manifest.csv", "external"),
        ("idrid_manifest.csv", "external"),
    ]:
        path = proj / "data_splits" / name
        if not path.is_file():
            print(f"skip {name} ({split}) - not found")
            continue
        ds = FundusDataset(path, split=split)
        img, label, idc = ds[0]
        print(f"{name} [{split}]: n={len(ds)}  sample={idc} y={label} "
              f"shape={tuple(img.shape)} min={img.min():.2f} max={img.max():.2f}  "
              f"counts={ds.class_counts().tolist()}")
