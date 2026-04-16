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
    SAMPLE_RATE = 16000  # Štandard pre reč, stačí na zachytenie stresových nuans
    DURATION = 3.0       # Sekundy (všetky nahrávky zarovnáme na túto dĺžku)
    
    # Príznaky (Features)
    N_MFCC = 13          # Počet koeficientov MFCC
    N_MELS = 128         # Počet Mel-filtrov pre spektrogramy
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
    # Mapovanie pre TESS (ak chceš binárnu klasifikáciu)
    TESS_LABEL_MAPPING = {
        "neutral": 0,
        "calm": 0,
        "angry": 1,
        "fear": 1,
        "disgust": 1,
        "ps": 0, # pleasant surprise
        "sad": 1
    }

    # --- TRÉNING ---
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 50
    RANDOM_SEED = 42

    # --- ZARIADENIE (CPU/GPU) ---
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    