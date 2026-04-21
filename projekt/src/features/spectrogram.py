from __future__ import annotations

import librosa
import numpy as np


def compute_mel_spectrogram(
    y: np.ndarray,
    sr: int,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 512,
    fmin: float = 0.0,
    fmax: float | None = None,
    to_db: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute Mel spectrogram from audio signal.

    Returns:
        np.ndarray of shape [n_mels, time]
    """
    if y is None or len(y) == 0:
        return np.zeros((n_mels, 1), dtype=np.float32)

    spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )

    if to_db:
        spec = librosa.power_to_db(spec, ref=np.max)

    spec = spec.astype(np.float32)

    if normalize:
        mean = spec.mean()
        std = spec.std()
        if std > 1e-8:
            spec = (spec - mean) / std
        else:
            spec = spec - mean

    return spec


def spectrogram_to_tensor(spec: np.ndarray) -> np.ndarray:
    """
    Convert spectrogram [H, W] to CNN-ready tensor [1, H, W].
    """
    spec = np.asarray(spec, dtype=np.float32)

    if spec.ndim != 2:
        raise ValueError(f"Expected 2D spectrogram, got shape {spec.shape}")

    return np.expand_dims(spec, axis=0).astype(np.float32)