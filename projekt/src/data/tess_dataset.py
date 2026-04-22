import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from src.config import Config


class TessDataset(Dataset):
    """
    TESS dataset loader for:
    - mode='features'      -> MFCC
    - mode='spectrograms'  -> Mel spectrogram

    Split:
    - train / val / test
    """

    def __init__(self, mode="features", split="train", transform=None):
        assert mode in ("features", "spectrograms")
        assert split in ("train", "val", "test")

        self.mode = mode
        self.split = split
        self.transform = transform

        metadata = pd.read_csv(Config.TESS_PROCESSED_PATH / "metadata.csv")

        # stratified split by label
        train_val_df, test_df = train_test_split(
            metadata,
            test_size=Config.TEST_SIZE,
            random_state=Config.RANDOM_SEED,
            stratify=metadata["label"],
        )

        val_ratio_adjusted = Config.VAL_SIZE / (1.0 - Config.TEST_SIZE)

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio_adjusted,
            random_state=Config.RANDOM_SEED,
            stratify=train_val_df["label"],
        )

        if split == "train":
            self.metadata = train_df.reset_index(drop=True)
        elif split == "val":
            self.metadata = val_df.reset_index(drop=True)
        else:
            self.metadata = test_df.reset_index(drop=True)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        row = self.metadata.iloc[idx]

        if self.mode == "features":
            file_path = Config.ROOT_DIR / row["mfcc_path"]
        else:
            file_path = Config.ROOT_DIR / row["spec_path"]

        data = np.load(file_path).astype(np.float32)

        # CNN input: (1, n_mels, time)
        if self.mode == "spectrograms" and data.ndim == 2:
            data = np.expand_dims(data, axis=0)

        x = torch.from_numpy(data).float()
        y = torch.tensor(int(row["label"]), dtype=torch.long)

        if self.transform is not None:
            x = self.transform(x)

        return x, y