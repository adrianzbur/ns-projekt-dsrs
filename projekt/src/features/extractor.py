from __future__ import annotations

from typing import Dict, Optional

import librosa
import numpy as np


def _safe_stats(values: np.ndarray, prefix: str) -> Dict[str, float]:
    """
    Compute robust statistics for a 1D array.
    """
    values = np.asarray(values, dtype=np.float32)

    if values.size == 0 or np.all(np.isnan(values)):
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
        }

    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
        }

    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
    }


def extract_pitch_features(
    y: np.ndarray,
    sr: int,
    fmin: float = 50.0,
    fmax: float = 500.0,
) -> Dict[str, float]:
    """
    Extract pitch-related statistics using librosa.yin.
    """
    if y is None or len(y) == 0:
        return _safe_stats(np.array([]), "pitch")

    try:
        pitch = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr)
        pitch = pitch[np.isfinite(pitch)]
        return _safe_stats(pitch, "pitch")
    except Exception:
        return _safe_stats(np.array([]), "pitch")


def extract_energy_features(y: np.ndarray) -> Dict[str, float]:
    """
    Extract signal energy / RMS statistics.
    """
    if y is None or len(y) == 0:
        return _safe_stats(np.array([]), "energy")

    try:
        rms = librosa.feature.rms(y=y).flatten()
        return _safe_stats(rms, "energy")
    except Exception:
        return _safe_stats(np.array([]), "energy")


def extract_mfcc_features(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
) -> Dict[str, float]:
    """
    Extract MFCC statistics.
    For each MFCC coefficient computes mean/std/min/max.
    """
    features: Dict[str, float] = {}

    if y is None or len(y) == 0:
        for i in range(n_mfcc):
            features[f"mfcc_{i+1}_mean"] = 0.0
            features[f"mfcc_{i+1}_std"] = 0.0
            features[f"mfcc_{i+1}_min"] = 0.0
            features[f"mfcc_{i+1}_max"] = 0.0
        return features

    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        for i in range(n_mfcc):
            coeff = mfcc[i]
            stats = _safe_stats(coeff, f"mfcc_{i+1}")
            features.update(stats)

        return features
    except Exception:
        for i in range(n_mfcc):
            features[f"mfcc_{i+1}_mean"] = 0.0
            features[f"mfcc_{i+1}_std"] = 0.0
            features[f"mfcc_{i+1}_min"] = 0.0
            features[f"mfcc_{i+1}_max"] = 0.0
        return features


def extract_feature_vector(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    include_pitch: bool = True,
    include_energy: bool = True,
    include_mfcc: bool = True,
    return_dict: bool = False,
) -> np.ndarray | Dict[str, float]:
    """
    Main feature extractor for MLP.

    Combines:
    - pitch statistics
    - energy statistics
    - MFCC statistics

    Returns either:
    - numpy vector
    - ordered dict of feature_name -> value
    """
    feature_dict: Dict[str, float] = {}

    if include_pitch:
        feature_dict.update(extract_pitch_features(y, sr))

    if include_energy:
        feature_dict.update(extract_energy_features(y))

    if include_mfcc:
        feature_dict.update(extract_mfcc_features(y, sr, n_mfcc=n_mfcc))

    if return_dict:
        return feature_dict

    return np.array(list(feature_dict.values()), dtype=np.float32)