import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from src.config import Config

class TessDataset(Dataset):
    def __init__(self, feature_type='mfcc', transform=None):
        """
        Args:
            feature_type (str): 'mfcc' alebo 'spectrogram'
            transform (callable, optional): Augmentácie alebo normalizácia
        """
        self.metadata = pd.read_csv(Config.TESS_PROCESSED_PATH / "metadata.csv")
        self.feature_type = feature_type
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.metadata.iloc[idx]
        
        # Výber správnej cesty k súboru na základe typu príznakov
        if self.feature_type == 'mfcc':
            file_path = Config.ROOT_DIR / row['mfcc_path']
        else:
            file_path = Config.ROOT_DIR / row['spec_path']

        # Načítanie uložených dát
        data = np.load(file_path)
        
        # Pre CNN (spectrogram) potrebujeme pridať kanál (Channel dimension)
        # Tvar: (N_MELS, TIME) -> (1, N_MELS, TIME)
        if self.feature_type == 'spectrogram':
            data = np.expand_dims(data, axis=0)

        # Prevod na Tensor
        data = torch.from_numpy(data).float()
        label = torch.tensor(row['label'], dtype=torch.long)

        # Voliteľné transformácie
        if self.transform:
            data = self.transform(data)

        return data, label

# --- Testovací blok (spustí sa len ak spustíš tento súbor priamo) ---
if __name__ == "__main__":
    dataset = TessDataset(feature_type='mfcc')
    print(f"Veľkosť datasetu: {len(dataset)}")
    
    sample_data, sample_label = dataset[0]
    print(f"Tvar MFCC vzorky: {sample_data.shape}")
    print(f"Label vzorky: {sample_label}")
    
    dataset_spec = TessDataset(feature_type='spectrogram')
    sample_spec, _ = dataset_spec[0]
    print(f"Tvar Spectrogram vzorky: {sample_spec.shape}")