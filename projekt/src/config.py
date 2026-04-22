import os
from pathlib import Path

class Config:
    # --- PROJEKTOVÁ ŠTRUKTÚRA ---
    ROOT_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = ROOT_DIR / "data"
    RAW_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"
    
    OUTPUTS_DIR = ROOT_DIR / "outputs"
    MODELS_DIR = OUTPUTS_DIR / "models"
    PLOTS_DIR = OUTPUTS_DIR / "plots"
    LOGS_DIR = OUTPUTS_DIR / "logs"

    # Vytvorenie potrebných priečinkov, ak neexistujú
    for path in [PROCESSED_DIR, MODELS_DIR, PLOTS_DIR, LOGS_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    # --- PARAMETRE AUDIA ---
    SAMPLE_RATE = 16000
    DURATION = 3.0
    
    # Príznaky (Features)
    N_MFCC = 13
    N_MELS = 128
    HOP_LENGTH = 512
    N_FFT = 2048

    # --- ŠPECIFIKÁ DATASETOV ---
    # WorkStress3D (ws3d)
    WS3D_RAW_PATH = RAW_DIR / "ws3d"
    WS3D_PROCESSED_PATH = PROCESSED_DIR / "ws3d"
    WS3D_LABELS = {
        0: "no-stress",
        1: "stress"
    }

    # TESS
    TESS_RAW_PATH = RAW_DIR / "tess"
    TESS_PROCESSED_PATH = PROCESSED_DIR / "tess"
    TESS_LABEL_MAPPING = {
        "neutral": 0,
        "calm": 0,
        "angry": 1,
        "fear": 1,
        "disgust": 1,
        "ps": 0,  # pleasant surprise
        "sad": 1
    }

    # --- TRÉNING ---
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 50
    RANDOM_SEED = 42

    # split pomery
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2

    # --- ZARIADENIE (CPU/GPU) ---
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Module-level aliasy kvôli kompatibilite s run.py / dataset modulmi
# ============================================================================

# základné
SEED = Config.RANDOM_SEED
DEVICE = Config.DEVICE

# split
TEST_SIZE = Config.TEST_SIZE
VAL_SIZE = Config.VAL_SIZE

# cesty WS3D
WS3D_FEATURES = str(Config.WS3D_PROCESSED_PATH / "features")
WS3D_SPECTROGRAMS = str(Config.WS3D_PROCESSED_PATH / "spectrograms")

# cesty TESS (ak ich používaš v tess_dataset.py)
TESS_FEATURES = str(Config.TESS_PROCESSED_PATH / "features")
TESS_SPECTROGRAMS = str(Config.TESS_PROCESSED_PATH / "spectrograms")

# model/tréning parametre – zatiaľ rovnaké pre MLP/CNN
MLP_BATCH_SIZE = Config.BATCH_SIZE
MLP_EPOCHS = Config.EPOCHS
MLP_LR = Config.LEARNING_RATE
MLP_WEIGHT_DECAY = 1e-4

CNN_BATCH_SIZE = Config.BATCH_SIZE
CNN_EPOCHS = Config.EPOCHS
CNN_LR = Config.LEARNING_RATE
CNN_WEIGHT_DECAY = 1e-4

# vstupný rozmer feature vektora (podľa MFCC)
# vstupné rozmery
TESS_FEATURE_DIM = Config.N_MFCC     # 13
WS3D_FEATURE_DIM = 120               # 40 MFCC + delta + delta2

# spätná kompatibilita
FEATURE_DIM = TESS_FEATURE_DIM