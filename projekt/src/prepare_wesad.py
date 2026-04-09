import os
import zipfile
import shutil
import numpy as np
import pandas as pd
from scipy.signal import spectrogram
from kaggle.api.kaggle_api_extended import KaggleApi

RAW_PATH = "projekt/data/raw/wesad"
FEATURES_PATH = "projekt/data/processed/wesad/features"
SPECTRO_PATH = "projekt/data/processed/wesad/spectrograms"
ZIP_FILE = os.path.join(RAW_PATH, "wesad-wearable-stress-affect-detection-dataset.zip")

os.makedirs(RAW_PATH, exist_ok=True)
os.makedirs(FEATURES_PATH, exist_ok=True)
os.makedirs(SPECTRO_PATH, exist_ok=True)

api = KaggleApi()
api.authenticate()
dataset_name = "orvile/wesad-wearable-stress-affect-detection-dataset"

if not os.path.exists(ZIP_FILE):
    print("Downloading ZIP from Kaggle...")
    api.dataset_download_files(dataset_name, path=RAW_PATH, unzip=False)
    print("Download complete.")

subfolders_exist = any(d.startswith("S") for d in os.listdir(RAW_PATH) if os.path.isdir(os.path.join(RAW_PATH, d)))

if not subfolders_exist:
    print(f"Unzipping {ZIP_FILE} ...")
    with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
        zip_ref.extractall(RAW_PATH)
    extracted_root = os.path.join(RAW_PATH, "WESAD")
    if os.path.exists(extracted_root):
        for sub in os.listdir(extracted_root):
            shutil.move(os.path.join(extracted_root, sub), RAW_PATH)
        shutil.rmtree(extracted_root)
    print("Unzip complete.")

subject_dirs = [d for d in os.listdir(RAW_PATH) if os.path.isdir(os.path.join(RAW_PATH, d)) and d.startswith("S") and d != "S1"]

total_feat = 0
total_spec = 0

for sub in sorted(subject_dirs):
    sub_path = os.path.join(RAW_PATH, sub)
    files = [f for f in os.listdir(sub_path) if f.endswith(".csv")]
    print(f"Processing subject {sub}, {len(files)} CSV files found.")

    for f in files:
        file_path = os.path.join(sub_path, f)
        try:
            df = pd.read_csv(file_path, delimiter=';', engine='python')
        except pd.errors.ParserError:
            print(f"Skipping {f}, could not parse.")
            continue

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if numeric_cols.empty:
            print(f"Skipping {f}, no numeric columns.")
            continue

        base_name = f.replace(f"{sub}_", "").replace(".csv", ".npy")
        feat_file = os.path.join(FEATURES_PATH, f"{sub}_{base_name}")
        np.save(feat_file, df[numeric_cols].values)
        total_feat += 1

        for col in numeric_cols:
            signal = df[col].values.astype(float)
            signal = signal[np.isfinite(signal)]
            if len(signal) < 32 or np.all(signal == signal[0]):
                continue
            f_spec, t_spec, Sxx = spectrogram(signal, fs=256, nperseg=64)
            Sxx_log = np.log1p(Sxx)
            spec_file = os.path.join(SPECTRO_PATH, f"{sub}_{base_name.replace('.npy','')}_{col}_spec.npy")
            np.save(spec_file, Sxx_log)
            total_spec += 1

print("All subjects processed successfully!")
print(f"Features stored in: {FEATURES_PATH} ({total_feat} files)")
print(f"Spectrograms stored in: {SPECTRO_PATH} ({total_spec} files)")