from __future__ import annotations

from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from src.config import Config


def _infer_label(filename: str) -> int:
    """
    Odvodí label z názvu súboru pomocou Config.WS3D_LABEL_MAPPING.

    Príklady:
        ses_a03.wav  → prefix "a"  → angry   → 1
        ses_n04.wav  → prefix "n"  → neutral  → 0
        ses_sa01.wav → prefix "sa" → sadness  → 1
        ses38.m4a    → žiadny prefix → -1
    """
    stem = Path(filename).stem.lower()

    if "_" not in stem:
        return -1

    after_underscore = stem.split("_", 1)[1]  # "a03", "sa01", ...

    emotion_prefix = ""
    for ch in after_underscore:
        if ch.isalpha():
            emotion_prefix += ch
        else:
            break

    if not emotion_prefix:
        return -1

    mapping = Config.WS3D_LABEL_MAPPING

    if emotion_prefix in mapping:
        return mapping[emotion_prefix]
    if emotion_prefix[0] in mapping:
        return mapping[emotion_prefix[0]]

    return -1


def _to_posix(path_str: str) -> str:
    try:
        return PureWindowsPath(path_str).as_posix()
    except Exception:
        return path_str


class Ws3dDataset(Dataset):
    """
    WS3D dataset loader kompatibilný s existujúcim labels.csv.

    Používa stratifikovaný split (rovnako ako TessDataset) pretože
    dataset má príliš málo vzoriek (14) na speaker-independent split.

    mode="features"     → tensor (120, T)   pre MLP
    mode="spectrograms" → tensor (1, 128, T) pre CNN
    """

    def __init__(
        self,
        mode: str = "features",
        split: str = "train",
        transform=None,
        val_size: float | None = None,
        test_size: float | None = None,
        seed: int | None = None,
    ):
        assert mode in ("features", "spectrograms"), f"Neznámy mode: {mode}"
        assert split in ("train", "val", "test"), f"Neznámy split: {split}"

        self.mode      = mode
        self.split     = split
        self.transform = transform

        _val_size  = val_size  if val_size  is not None else Config.VAL_SIZE
        _test_size = test_size if test_size is not None else Config.TEST_SIZE
        _seed      = seed      if seed      is not None else Config.RANDOM_SEED

        # ── načítaj labels.csv ──────────────────────────────────────────────
        labels_path = Config.WS3D_PROCESSED_PATH / "labels.csv"
        if not labels_path.exists():
            raise FileNotFoundError(
                f"labels.csv nenájdený: {labels_path}\n"
                "Spusti najprv: python scripts/prepare_ws3d.py"
            )

        df = pd.read_csv(labels_path)
        df.columns = [c.strip().lower() for c in df.columns]

        # ── odvoď labely z názvu súboru ─────────────────────────────────────
        df["label"] = df["filename"].apply(_infer_label)

        # ── vyhoď riadky bez platného labelu (ses38.m4a a pod.) ─────────────
        before = len(df)
        df = df[df["label"].isin([0, 1])].reset_index(drop=True)
        after = len(df)

        if after == 0:
            raise ValueError(
                "Žiadne vzorky s platným labelom (0 alebo 1).\n"
                "Skontroluj Config.WS3D_LABEL_MAPPING a názvy súborov v labels.csv."
            )

        print(f"[Ws3dDataset] Celkom: {before}  |  S labelom: {after}  |  "
              f"Vyhodených: {before - after}  |  "
              f"stress={(df['label'] == 1).sum()}  "
              f"no-stress={(df['label'] == 0).sum()}")

        # ── oprav Windows cesty na POSIX ────────────────────────────────────
        for col in ("feat_path", "spec_path"):
            df[col] = df[col].apply(_to_posix)

        # ── stratifikovaný split podľa labelu ───────────────────────────────
        # (rovnaká logika ako TessDataset — zaručí obe triedy v každom splite)
        train_val_df, test_df = train_test_split(
            df,
            test_size=_test_size,
            random_state=_seed,
            stratify=df["label"],
        )

        val_ratio_adjusted = _val_size / (1.0 - _test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio_adjusted,
            random_state=_seed,
            stratify=train_val_df["label"],
        )

        if split == "train":
            self.metadata = train_df.reset_index(drop=True)
        elif split == "val":
            self.metadata = val_df.reset_index(drop=True)
        else:
            self.metadata = test_df.reset_index(drop=True)

        print(f"[Ws3dDataset] split='{split}'  vzorky={len(self.metadata)}  "
              f"stress={(self.metadata['label'] == 1).sum()}  "
              f"no-stress={(self.metadata['label'] == 0).sum()}")

        if len(self.metadata) == 0:
            raise ValueError(f"Split '{split}' je prázdny.")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.item()

        row = self.metadata.iloc[idx]

        rel_path = row["feat_path"] if self.mode == "features" else row["spec_path"]
        file_path = Config.ROOT_DIR / rel_path

        if not file_path.exists():
            raise FileNotFoundError(f"Súbor nenájdený: {file_path}")

        data = np.load(file_path).astype(np.float32)

        # CNN vstup: (1, n_mels, T)
        if self.mode == "spectrograms" and data.ndim == 2:
            data = np.expand_dims(data, axis=0)

        x = torch.from_numpy(data).float()
        y = torch.tensor(int(row["label"]), dtype=torch.long)

        if self.transform is not None:
            x = self.transform(x)

        return x, y