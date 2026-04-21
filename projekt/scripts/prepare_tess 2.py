import os
import sys
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Pridanie koreňového priečinka do sys.path, aby sme mohli importovať src
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import Config

def process_audio(file_path):
    """Načíta audio, oreže/doplní na fixnú dĺžku a vráti signál."""
    y, sr = librosa.load(file_path, sr=Config.SAMPLE_RATE)
    
    # Fixná dĺžka (zarovnanie na Config.DURATION)
    target_samples = int(Config.DURATION * Config.SAMPLE_RATE)
    if len(y) > target_samples:
        y = y[:target_samples]
    else:
        y = np.pad(y, (0, target_samples - len(y)))
    return y

def extract_features(y):
    """Extrahuje MFCC a Mel-spektrogram podľa parametrov v Config."""
    mfcc = librosa.feature.mfcc(
        y=y, sr=Config.SAMPLE_RATE, n_mfcc=Config.N_MFCC, n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH
    )
    
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=Config.SAMPLE_RATE, n_mels=Config.N_MELS, n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mfcc, mel_spec_db

def main():
    print(f"--- Spúšťam spracovanie TESS datasetu ---")
    
    # Vytvorenie cieľových priečinkov
    features_dir = Config.TESS_PROCESSED_PATH / "features"
    specs_dir = Config.TESS_PROCESSED_PATH / "spectrograms"
    features_dir.mkdir(parents=True, exist_ok=True)
    specs_dir.mkdir(parents=True, exist_ok=True)

    data_list = []
    
    # TESS má štruktúru: TESS_RAW_PATH / Folder_Emócia / Súbor.wav
    # Príklad priečinka: OAF_angry, YAF_neutral
    audio_files = list(Config.TESS_RAW_PATH.rglob("*.wav"))
    
    if not audio_files:
        print(f"Chyba: Nenašli sa žiadne .wav súbory v {Config.TESS_RAW_PATH}")
        return

    for file_path in tqdm(audio_files, desc="Spracovávam audio"):
        # Zistenie emócie z názvu priečinka alebo súboru
        # TESS súbory sú väčšinou v tvare: Actor_Word_Emotion.wav
        parts = file_path.stem.split("_")
        emotion = parts[-1].lower()
        
        # Mapovanie na 0 (no-stress) a 1 (stress) podľa Configu
        label = Config.TESS_LABEL_MAPPING.get(emotion)
        
        if label is None:
            continue  # Preskočíme emócie, ktoré nemáme v mape

        try:
            # 1. Audio processing
            y = process_audio(file_path)
            
            # 2. Extrakcia
            mfcc, mel_spec = extract_features(y)
            
            # 3. Uloženie ako .npy (rýchle načítanie pri tréningu)
            file_id = file_path.stem
            mfcc_path = features_dir / f"{file_id}_mfcc.npy"
            spec_path = specs_dir / f"{file_id}_spec.npy"
            
            np.save(mfcc_path, mfcc)
            np.save(spec_path, mel_spec)
            
            # 4. Záznam do metadát
            data_list.append({
                "file_id": file_id,
                "emotion": emotion,
                "label": label,
                "mfcc_path": str(mfcc_path.relative_to(Config.ROOT_DIR)),
                "spec_path": str(spec_path.relative_to(Config.ROOT_DIR))
            })
            
        except Exception as e:
            print(f"Chyba pri súbore {file_path}: {e}")

    # Uloženie metadát do CSV pre jednoduchý prístup v Dataset triede
    df = pd.DataFrame(data_list)
    df.to_csv(Config.TESS_PROCESSED_PATH / "metadata.csv", index=False)
    print(f"--- Hotovo! Spracovaných {len(df)} vzoriek. ---")

if __name__ == "__main__":
    main()