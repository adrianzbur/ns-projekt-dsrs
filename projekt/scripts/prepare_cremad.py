"""
prepare_cremad.py – Predspracovanie CREMA-D datasetu.

CREMA-D štruktúra súborov:
    AudioWAV/
        1001_DFA_ANG_XX.wav
        1001_DFA_DIS_LO.wav
        ...

Formát názvu: {ActorID}_{Sentence}_{Emotion}_{Intensity}.wav

Emócie (6):
    ANG – Anger
    DIS – Disgust
    FEA – Fear
    HAP – Happy
    NEU – Neutral
    SAD – Sad

Intenzity (4):
    LO  – Low
    MD  – Medium
    HI  – High
    XX  – Unspecified

Mapovanie na binárny label (rovnaká logika ako TESS):
    stress=1 : ANG, DIS, FEA, SAD
    no-stress=0 : HAP, NEU

Spustenie:
    python scripts/prepare_cremad.py

Očakávaná štruktúra vstupných dát:
    data/raw/cremad/AudioWAV/*.wav

Výstup:
    data/processed/cremad/features/         <- MFCC .npy súbory
    data/processed/cremad/spectrograms/     <- Log-Mel .npy súbory
    data/processed/cremad/metadata.csv      <- tabuľka so všetkými metadátami
"""

import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Cesty ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR      = PROJECT_ROOT / "data" / "raw"  / "cremad" / "AudioWAV"
PROC_DIR     = PROJECT_ROOT / "data" / "processed" / "cremad"
PROC_FEAT    = PROC_DIR / "features"
PROC_SPEC    = PROC_DIR / "spectrograms"
LOGS_DIR     = PROJECT_ROOT / "outputs" / "logs"

LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "prepare_cremad.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── Audio parametre (rovnaké ako v Config) ────────────────────────────────────
SAMPLE_RATE = 16_000
DURATION    = 3.0
N_MFCC      = 40        # rovnaké ako prepare_ws3d (MFCC + delta + delta2 = 120)
N_MELS      = 128
HOP_LENGTH  = 512
N_FFT       = 2048

# ── Label mapovanie ───────────────────────────────────────────────────────────
# Kód emócie v názve súboru → binárny label
EMOTION_TO_LABEL = {
    "ANG": 1,   # Anger     → stress
    "DIS": 1,   # Disgust   → stress
    "FEA": 1,   # Fear      → stress
    "SAD": 1,   # Sad       → stress
    "HAP": 0,   # Happy     → no-stress
    "NEU": 0,   # Neutral   → no-stress
}

# Platné kódy intenzity – "XX" (unspecified) záámerne zahrňujeme,
# pretože CREMA-D ho používa pre väčšinu HAP/NEU vzoriek.
VALID_INTENSITIES = {"LO", "MD", "HI", "XX"}


# ── Parsovanie názvu súboru ───────────────────────────────────────────────────

def parse_filename(stem: str) -> dict | None:
    """
    Rozparsuje stem súboru CREMA-D do slovníka.

    Príklad: '1001_DFA_ANG_XX'
        actor_id  = '1001'
        sentence  = 'DFA'
        emotion   = 'ANG'
        intensity = 'XX'
        label     = 1

    Vráti None ak formát nie je rozpoznaný.
    """
    parts = stem.split("_")
    if len(parts) != 4:
        return None

    actor_id, sentence, emotion, intensity = parts

    if emotion not in EMOTION_TO_LABEL:
        return None

    if intensity not in VALID_INTENSITIES:
        return None

    return {
        "actor_id":  actor_id,
        "sentence":  sentence,
        "emotion":   emotion,
        "intensity": intensity,
        "label":     EMOTION_TO_LABEL[emotion],
    }


# ── Audio spracovanie ─────────────────────────────────────────────────────────

def load_audio(path: Path) -> np.ndarray | None:
    """Načíta WAV cez librosa; vráti None pri chybe."""
    try:
        y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        return y
    except Exception as e:
        log.warning(f"[load] FAIL {path.name}: {e}")
        return None


def pad_or_trim(y: np.ndarray) -> np.ndarray:
    """Orezanie alebo doplnenie signálu na fixnú dĺžku."""
    target = int(SAMPLE_RATE * DURATION)
    if len(y) >= target:
        return y[:target]
    return np.pad(y, (0, target - len(y)))


def normalize(y: np.ndarray) -> np.ndarray:
    """Peak normalizácia."""
    peak = np.max(np.abs(y))
    return y / peak if peak > 0 else y


def extract_mfcc(y: np.ndarray) -> np.ndarray:
    """
    Extrahuje MFCC + delta + delta2.
    Výstup: (120, T) – rovnaký formát ako prepare_ws3d.
    """
    mfcc    = librosa.feature.mfcc(
        y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    delta   = librosa.feature.delta(mfcc)
    delta2  = librosa.feature.delta(mfcc, order=2)
    return np.vstack([mfcc, delta, delta2]).astype(np.float32)


def extract_melspec(y: np.ndarray) -> np.ndarray:
    """
    Extrahuje log-Mel spektrogram.
    Výstup: (128, T) – rovnaký formát ako prepare_ws3d.
    """
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS,
        n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    return librosa.power_to_db(mel).astype(np.float32)


# ── Hlavná logika ─────────────────────────────────────────────────────────────

def ensure_dirs():
    for d in [PROC_FEAT, PROC_SPEC]:
        d.mkdir(parents=True, exist_ok=True)


def collect_files() -> list[dict]:
    """
    Prejde RAW_DIR a vráti zoznam slovníkov pre každý platný súbor.
    Preskočí súbory s nerozpoznaným formátom názvu.
    """
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Priečinok s audio dátami nenájdený: {RAW_DIR}\n"
            "Uisti sa, že CREMA-D WAV súbory sú v: data/raw/cremad/AudioWAV/"
        )

    records = []
    skipped = 0

    for wav_path in sorted(RAW_DIR.glob("*.wav")):
        meta = parse_filename(wav_path.stem)
        if meta is None:
            skipped += 1
            log.debug(f"Preskočený (nerozpoznaný formát): {wav_path.name}")
            continue
        meta["path"] = wav_path
        records.append(meta)

    log.info(f"Nájdené súbory: {len(records)} platných, {skipped} preskočených")
    return records


def process_file(record: dict) -> dict | None:
    """
    Spracuje jeden audio súbor: načíta, normalizuje, extrahuje príznaky,
    uloží .npy a vráti aktualizovaný slovník so cestami.
    """
    path: Path = record["path"]
    stem = path.stem

    feat_path = PROC_FEAT / f"{stem}.npy"
    spec_path = PROC_SPEC / f"{stem}.npy"

    # Preskočíme ak už spracované (cache)
    if feat_path.exists() and spec_path.exists():
        return {
            **{k: v for k, v in record.items() if k != "path"},
            "filename":  path.name,
            "feat_path": str(feat_path.relative_to(PROJECT_ROOT)),
            "spec_path": str(spec_path.relative_to(PROJECT_ROOT)),
        }

    y = load_audio(path)
    if y is None:
        return None

    y = normalize(y)
    y = pad_or_trim(y)

    mfcc = extract_mfcc(y)
    mel  = extract_melspec(y)

    np.save(feat_path, mfcc)
    np.save(spec_path, mel)

    return {
        **{k: v for k, v in record.items() if k != "path"},
        "filename":  path.name,
        "feat_path": str(feat_path.relative_to(PROJECT_ROOT)),
        "spec_path": str(spec_path.relative_to(PROJECT_ROOT)),
    }


def save_metadata(processed: list[dict]):
    df = pd.DataFrame(processed)

    # Zoradiť stĺpce pre prehľadnosť
    col_order = ["filename", "actor_id", "sentence", "emotion",
                 "intensity", "label", "feat_path", "spec_path"]
    df = df[[c for c in col_order if c in df.columns]]

    out_path = PROC_DIR / "metadata.csv"
    df.to_csv(out_path, index=False)
    log.info(f"Metadáta uložené: {out_path}  ({len(df)} vzoriek)")
    return df


def print_summary(df: pd.DataFrame):
    """Vypíše štatistiky datasetu po spracovaní."""
    n_stress    = (df["label"] == 1).sum()
    n_no_stress = (df["label"] == 0).sum()
    n_actors    = df["actor_id"].nunique()

    log.info("─" * 50)
    log.info(f"Celkom vzoriek : {len(df)}")
    log.info(f"  stress (1)   : {n_stress}")
    log.info(f"  no-stress (0): {n_no_stress}")
    log.info(f"Počet hercov   : {n_actors}")
    log.info("Rozloženie emócií:")
    for emotion, count in df["emotion"].value_counts().items():
        label = EMOTION_TO_LABEL[emotion]
        log.info(f"  {emotion}  label={label}  count={count}")
    log.info("Rozloženie intenzít:")
    for intensity, count in df["intensity"].value_counts().items():
        log.info(f"  {intensity}  count={count}")
    log.info("─" * 50)


def main():
    log.info("=== CREMA-D preprocessing start ===")
    ensure_dirs()

    records = collect_files()
    if not records:
        log.error("Žiadne platné súbory. Skontroluj cestu a obsah data/raw/cremad/AudioWAV/")
        sys.exit(1)

    processed = []
    failed    = []

    for record in tqdm(records, desc="Spracovávam CREMA-D"):
        result = process_file(record)
        if result is not None:
            processed.append(result)
        else:
            failed.append(record["path"].name)

    if not processed:
        log.error("Žiadne súbory sa nepodarilo spracovať.")
        sys.exit(1)

    df = save_metadata(processed)
    print_summary(df)

    if failed:
        log.warning(f"Neúspešné súbory ({len(failed)}):")
        for name in failed:
            log.warning(f"  {name}")

    log.info(f"=== HOTOVO ===  OK: {len(processed)}  FAIL: {len(failed)} ===")


if __name__ == "__main__":
    main()