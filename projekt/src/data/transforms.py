"""
src/data/transforms.py

Augmentácie pre trénovaciu fázu.

Používajú sa iba počas trénovania, NIE pri validácii a teste.
Každá trieda je callable — kompatibilná s Dataset.transform.

Augmentácie pre feature vektory (MLP):
    GaussianNoise     – pridá šum do feature vektora

Augmentácie pre spektrogramy (CNN):
    SpecAugment       – časové a frekvenčné maskovanie (Park et al. 2019)
    SpectrogramNoise  – pridá šum do log-Mel spektrogramu
"""

import torch
import numpy as np


class GaussianNoise:
    """
    Pridá gaussovský šum do feature vektora.
    Zlepšuje robustnosť MLP voči malým variáciám príznakov.

    Parametre
    ----------
    std : smerodajná odchýlka šumu (relatívne k std vektora)
    """
    def __init__(self, std: float = 0.01):
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * self.std
        return x + noise


class SpecAugment:
    """
    SpecAugment – časové a frekvenčné maskovanie spektrogramu.
    Park et al. (2019): https://arxiv.org/abs/1904.08779

    Parametre
    ----------
    freq_mask_param : max. šírka frekvenčnej masky (v Mel kanáloch)
    time_mask_param : max. šírka časovej masky (v rámcoch)
    n_freq_masks    : počet frekvenčných masiek
    n_time_masks    : počet časových masiek
    """
    def __init__(
        self,
        freq_mask_param: int = 10,
        time_mask_param: int = 20,
        n_freq_masks:    int = 2,
        n_time_masks:    int = 2,
    ):
        self.F  = freq_mask_param
        self.T  = time_mask_param
        self.nF = n_freq_masks
        self.nT = n_time_masks

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : tvar (1, n_mels, time_frames)
        """
        x = x.clone()

        # Podpora aj pre tvar (n_mels, time_frames)
        squeeze_back = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_back = True

        _, n_mels, n_frames = x.shape
        fill_value = x.mean()

        # Frekvenčné masky
        for _ in range(self.nF):
            max_f = min(self.F, n_mels)
            if max_f <= 0:
                continue
            f = np.random.randint(0, max_f + 1)  # 0..max_f
            if f == 0:
                continue
            f0 = np.random.randint(0, n_mels - f + 1)
            x[:, f0: f0 + f, :] = fill_value

        # Časové masky
        for _ in range(self.nT):
            max_t = min(self.T, n_frames)
            if max_t <= 0:
                continue
            t = np.random.randint(0, max_t + 1)  # 0..max_t
            if t == 0:
                continue
            t0 = np.random.randint(0, n_frames - t + 1)
            x[:, :, t0: t0 + t] = fill_value

        if squeeze_back:
            x = x.squeeze(0)

        return x


class SpectrogramNoise:
    """
    Pridá gaussovský šum do log-Mel spektrogramu.

    Parametre
    ----------
    std : smerodajná odchýlka šumu
    """
    def __init__(self, std: float = 0.05):
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.std


class Compose:
    """Reťazí niekoľko transformácií za sebou (ako torchvision.transforms.Compose)."""
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x


# ─── Predpripravené sady augmentácií ─────────────────────────────────────────

def get_train_transforms_mlp() -> Compose:
    return Compose([GaussianNoise(std=0.01)])


def get_train_transforms_cnn() -> Compose:
    return Compose([
        SpectrogramNoise(std=0.03),
        SpecAugment(freq_mask_param=8, time_mask_param=15, n_freq_masks=2, n_time_masks=2),
    ])