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
        ses_a03.wav  → prefix "a"  → angry  → 1
        ses_n04.wav  → prefix "n"  → neutral → 0
        ses_sa01.wav → prefix "sa" → sadness → 1
        ses38.m4a    → žiadny prefix → -1 (neznámy)
    """
    stem = Path(filename).stem.lower()  # napr. "ses_a03" alebo "ses38"

    # Odstrán "ses_" prefix
    if "_" in stem:
        after_underscore = stem.split("_", 1)[1]  # napr. "a03" alebo "sa01"
    else:
        return -1  # napr. "ses38" → žiadna emócia v názve

    # Extrahuj len písmená zo začiatku
    emotion_prefix = ""
    for ch in after_underscore:
        if ch.isalpha():
            emotion_prefix += ch
        else:
            break  # narazili sme na číslo

    if not emotion_prefix:
        return -1

    mapping = Config.WS3D_LABEL_MAPPING

    # Skús celý prefix (napr. "sa"), potom prvý znak (napr. "s")
    if emotion_prefix in mapping:
        return mapping[emotion_prefix]
    if emotion_prefix[0] in mapping:
        return mapping[emotion_prefix[0]]

    return -1


def _to_posix(path_str: str) -> str:
    """Konvertuje Windows cestu (aj na Linuxe) na POSIX formát."""
    try:
        return PureWindowsPath(path_str).as_posix()
    except Exception:
        return path_str


class Ws3dDataset(Dataset):
    """
    WS3D dataset loader kompatibilný s existujúcim labels.csv.

    Label sa odvodí priamo z názvu súboru cez Config.WS3D_LABEL_MAPPING
    — labels.csv môže mať label=-1, to nevadí.

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
                "Spusti najprv: python src/data/prepare_ws3d.py"
            )

        df = pd.read_csv(labels_path)
        df.columns = [c.strip().lower() for c in df.columns]

        # ── oprav labely z názvu súboru ─────────────────────────────────────
        df["label"] = df["filename"].apply(_infer_label)

        # ── vyhoď riadky kde label == -1 (ses38.m4a a pod.) ────────────────
        before = len(df)
        df = df[df["label"].isin([0, 1])].reset_index(drop=True)
        after = len(df)

        if after == 0:
            raise ValueError(
                "Žiadne vzorky s platným labelom (0 alebo 1).\n"
                "Skontroluj Config.WS3D_LABEL_MAPPING a názvy súborov v labels.csv."
            )

        print(f"[Ws3dDataset] Načítaných: {before}  |  S labelom: {after}  |  "
              f"Vyhodených: {before - after}  |  "
              f"stress={( df['label']==1).sum()}  no-stress={(df['label']==0).sum()}")

        # ── oprav Windows cesty na POSIX ────────────────────────────────────
        for col in ("feat_path", "spec_path"):
            df[col] = df[col].apply(_to_posix)

        # ── speaker-independent split podľa subject ─────────────────────────
        subjects = df["subject"].unique()

        train_subj, test_subj = train_test_split(
            subjects, test_size=_test_size, random_state=_seed,
        )
        val_ratio_adjusted = _val_size / (1.0 - _test_size)
        train_subj, val_subj = train_test_split(
            train_subj, test_size=val_ratio_adjusted, random_state=_seed,
        )

        if split == "train":
            mask = df["subject"].isin(train_subj)
        elif split == "val":
            mask = df["subject"].isin(val_subj)
        else:
            mask = df["subject"].isin(test_subj)

        self.metadata = df[mask].reset_index(drop=True)

        if len(self.metadata) == 0:
            raise ValueError(
                f"Split '{split}' je prázdny po rozdelení "
                f"(celkovo {len(subjects)} subjektov). "
                "Skontroluj veľkosti splitov v Config."
            )

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.item()

        row = self.metadata.iloc[idx]

        if self.mode == "features":
            rel_path = row["feat_path"]
        else:
            rel_path = row["spec_path"]

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