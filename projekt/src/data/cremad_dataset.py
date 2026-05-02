from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from src.config import Config


class CremadDataset(Dataset):
    """
    CREMA-D dataset loader kompatibilný s pipeline TESS + WS3D.

    Načíta metadata.csv vygenerovaný skriptom scripts/prepare_cremad.py.

    mode='features'      -> MFCC tensor (120, T)   pre MLP
    mode='spectrograms'  -> tensor (1, 128, T)     pre CNN

    Split: stratifikovaný podľa labelu (rovnaká logika ako TessDataset).
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
        assert split in ("train", "val", "test"),    f"Neznámy split: {split}"

        self.mode      = mode
        self.split     = split
        self.transform = transform

        _val_size  = val_size  if val_size  is not None else Config.VAL_SIZE
        _test_size = test_size if test_size is not None else Config.TEST_SIZE
        _seed      = seed      if seed      is not None else Config.RANDOM_SEED

        # ── načítaj metadata.csv ──────────────────────────────────────────
        metadata_path = Config.CREMAD_PROCESSED_PATH / "metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"metadata.csv nenájdený: {metadata_path}\n"
                "Spusti najprv: python scripts/prepare_cremad.py"
            )

        df = pd.read_csv(metadata_path)
        df = df[df["label"].isin([0, 1])].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError("metadata.csv neobsahuje žiadne platné vzorky (label 0 alebo 1).")

        print(
            f"[CremadDataset] Celkom: {len(df)}  |  "
            f"stress={(df['label'] == 1).sum()}  "
            f"no-stress={(df['label'] == 0).sum()}  |  "
            f"hercov={df['actor_id'].nunique()}"
        )

        # ── stratifikovaný split podľa labelu ─────────────────────────────
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

        print(
            f"[CremadDataset] split='{split}'  vzorky={len(self.metadata)}  "
            f"stress={(self.metadata['label'] == 1).sum()}  "
            f"no-stress={(self.metadata['label'] == 0).sum()}"
        )

        if len(self.metadata) == 0:
            raise ValueError(f"Split '{split}' je prázdny.")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.item()

        row = self.metadata.iloc[idx]

        rel_path  = row["feat_path"] if self.mode == "features" else row["spec_path"]
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