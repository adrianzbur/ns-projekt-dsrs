import os
from pathlib import Path


class Config:
    # --- PROJEKTOVÁ ŠTRUKTÚRA ---
    ROOT_DIR      = Path(__file__).resolve().parent.parent
    DATA_DIR      = ROOT_DIR / "data"
    RAW_DIR       = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"
    OUTPUTS_DIR   = ROOT_DIR / "outputs"
    MODELS_DIR    = OUTPUTS_DIR / "models"
    PLOTS_DIR     = OUTPUTS_DIR / "plots"
    LOGS_DIR      = OUTPUTS_DIR / "logs"

    for path in [PROCESSED_DIR, MODELS_DIR, PLOTS_DIR, LOGS_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    # --- PARAMETRE AUDIA ---
    SAMPLE_RATE = 16000
    DURATION    = 3.0

    N_MFCC     = 13
    N_MELS     = 128
    HOP_LENGTH = 512
    N_FFT      = 2048

    # --- ŠPECIFIKÁ DATASETOV ---

    # TESS
    TESS_RAW_PATH       = RAW_DIR / "tess"
    TESS_PROCESSED_PATH = PROCESSED_DIR / "tess"
    TESS_LABEL_MAPPING = {
        "neutral": 0,
        "calm":    0,
        "angry":   1,
        "fear":    1,
        "disgust": 1,
        "ps":      0,   # pleasant surprise
        "sad":     1,
    }

    # CREMA-D
    CREMAD_RAW_PATH       = RAW_DIR / "cremad" / "AudioWAV"
    CREMAD_PROCESSED_PATH = PROCESSED_DIR / "cremad"
    # Mapovanie kódu emócie z názvu súboru na binárny label
    # stress=1 : ANG, DIS, FEA, SAD
    # no-stress=0 : HAP, NEU
    CREMAD_LABEL_MAPPING = {
        "ANG": 1,   # Anger     → stress
        "DIS": 1,   # Disgust   → stress
        "FEA": 1,   # Fear      → stress
        "SAD": 1,   # Sad       → stress
        "HAP": 0,   # Happy     → no-stress
        "NEU": 0,   # Neutral   → no-stress
    }
    CREMAD_LABELS = {
        0: "no-stress",
        1: "stress",
    }

    # --- TRÉNING ---
    BATCH_SIZE    = 32
    LEARNING_RATE = 0.001
    EPOCHS        = 50
    RANDOM_SEED   = 42

    TEST_SIZE = 0.2
    VAL_SIZE  = 0.2

    # --- ZARIADENIE ---
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Module-level aliasy ───────────────────────────────────────────────────────
SEED   = Config.RANDOM_SEED
DEVICE = Config.DEVICE

TEST_SIZE = Config.TEST_SIZE
VAL_SIZE  = Config.VAL_SIZE

TESS_FEATURES      = str(Config.TESS_PROCESSED_PATH / "features")
TESS_SPECTROGRAMS  = str(Config.TESS_PROCESSED_PATH / "spectrograms")

CREMAD_FEATURES     = str(Config.CREMAD_PROCESSED_PATH / "features")
CREMAD_SPECTROGRAMS = str(Config.CREMAD_PROCESSED_PATH / "spectrograms")

MLP_BATCH_SIZE   = Config.BATCH_SIZE
MLP_EPOCHS       = Config.EPOCHS
MLP_LR           = Config.LEARNING_RATE
MLP_WEIGHT_DECAY = 1e-4

CNN_BATCH_SIZE   = Config.BATCH_SIZE
CNN_EPOCHS       = Config.EPOCHS
CNN_LR           = Config.LEARNING_RATE
CNN_WEIGHT_DECAY = 1e-4

TESS_FEATURE_DIM   = Config.N_MFCC   # 13
CREMAD_FEATURE_DIM = 120             # 40 MFCC + delta + delta2

FEATURE_DIM = TESS_FEATURE_DIM