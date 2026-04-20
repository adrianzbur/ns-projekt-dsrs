"""
src/data/ws3d_dataset.py

PyTorch Dataset trieda pre WS3D 

Predpokladaná štruktúra data/processed/ws3d/:
    features/
        S2_audio_label1.npy
        S3_audio_label0.npy
        ...
    spectrograms/
        S2_audio_label1.npy
        S3_audio_label0.npy
        ...

Labely:
    0 = nestres
    1 = stres
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (  # noqa: E402
    WS3D_FEATURES,
    WS3D_SPECTROGRAMS,
    TEST_SIZE,
    VAL_SIZE,
    SEED,
)


class Ws3dDataset(Dataset):
    """
    Načítava predspracované WS3D príznaky z .npy súborov.

    Rozdelenie train/val/test je speaker-independent:
    celé subjekty idú iba do jednej split sady.
    """

    def __init__(
        self,
        mode: str = "features",
        split: str = "train",
        transform=None,
    ):
        assert mode in ("features", "spectrograms"), "mode musí byť 'features' alebo 'spectrograms'"
        assert split in ("train", "val", "test"), "split musí byť 'train', 'val' alebo 'test'"

        self.mode = mode
        self.split = split
        self.transform = transform

        data_dir = WS3D_FEATURES if mode == "features" else WS3D_SPECTROGRAMS
        all_files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))

        if len(all_files) == 0:
            raise FileNotFoundError(
                f"Žiadne .npy súbory v {data_dir}\n"
                f"Skontroluj preprocessing skript a cesty v config.py."
            )

        labels = np.array([self._parse_label(f) for f in all_files], dtype=np.int32)
        subjects = np.array([self._parse_subject(f) for f in all_files], dtype=object)

        # Speaker-independent split podľa subjektov
        unique_subjects = np.array(sorted(set(subjects)), dtype=object)
        if len(unique_subjects) < 3:
            raise ValueError(
                f"Málo subjektov pre train/val/test split: {len(unique_subjects)}. "
                f"Potrebné sú aspoň 3 subjekty."
            )

        n_test = max(1, int(round(len(unique_subjects) * TEST_SIZE)))
        n_val = max(1, int(round(len(unique_subjects) * VAL_SIZE)))

        # nech zostane aspoň 1 subjekt pre train
        if n_test + n_val >= len(unique_subjects):
            n_val = max(1, len(unique_subjects) - n_test - 1)
            if n_val < 1:
                n_test = max(1, len(unique_subjects) - 2)
                n_val = 1

        rng = np.random.default_rng(SEED)
        shuffled = rng.permutation(unique_subjects)

        test_subs = set(shuffled[:n_test])
        val_subs = set(shuffled[n_test:n_test + n_val])
        train_subs = set(shuffled[n_test + n_val:])

        split_subjects = {"train": train_subs, "val": val_subs, "test": test_subs}
        chosen_subs = split_subjects[split]
        mask = np.array([s in chosen_subs for s in subjects], dtype=bool)

        self.files = [f for f, m in zip(all_files, mask) if m]
        self.labels = labels[mask]

        if len(self.files) == 0:
            raise ValueError(
                f"Split '{split}' je prázdny. "
                f"Skontroluj TEST_SIZE/VAL_SIZE a počet subjektov."
            )

        print(
            f"Ws3dDataset [{mode}][{split}]: "
            f"{len(self.files)} vzoriek  "
            f"subjekty={sorted(chosen_subs)}  "
            f"(nestres={int((self.labels == 0).sum())}, "
            f"stres={int((self.labels == 1).sum())})"
        )

    @staticmethod
    def _parse_label(filepath: str) -> int:
        stem = os.path.basename(filepath).replace(".npy", "")
        for part in reversed(stem.split("_")):
            if part.startswith("label"):
                value = int(part.replace("label", ""))
                if value not in (0, 1):
                    raise ValueError(f"Neočakávaný label {value} v súbore: {filepath}")
                return value
        raise ValueError(f"Label sa nedá parsovať z: {filepath}")

    @staticmethod
    def _parse_subject(filepath: str) -> str:
        # S2_audio_label1.npy -> S2
        return os.path.basename(filepath).split("_")[0]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        data = np.load(self.files[idx]).astype(np.float32)
        label = int(self.labels[idx])

        x = torch.from_numpy(data)
        y = torch.tensor(label, dtype=torch.long)

        if self.transform is not None:
            x = self.transform(x)

        return x, y