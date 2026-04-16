import sys
import logging
import warnings
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ─── Paths ─────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_AUDIO = PROJECT_ROOT / "data" / "raw" / "ws3d" / "AudioData"
PROC_FEAT = PROJECT_ROOT / "data" / "processed" / "ws3d" / "features"
PROC_SPEC = PROJECT_ROOT / "data" / "processed" / "ws3d" / "spectrograms"
LOGS_DIR  = PROJECT_ROOT / "outputs" / "logs"

LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ─── Logging (LEN DO SUBORU) ─────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("outputs/logs/prepare_ws3d.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)


# ─── Audio params ─────────────────────────────
SAMPLE_RATE = 16000
DURATION    = 3.0
N_MFCC      = 40
N_MELS      = 128
HOP_LENGTH  = 512
N_FFT       = 2048
FMIN        = 0
FMAX        = 8000


# ─── Dirs ─────────────────────────────
def ensure_dirs():
    for d in [PROC_FEAT, PROC_SPEC]:
        d.mkdir(parents=True, exist_ok=True)


# ─── AUDIO LOAD ─────────────────────────────
def load_via_ffmpeg(path: Path) -> np.ndarray | None:
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(path),
            "-vn",
            "-ac", "1",
            "-ar", str(SAMPLE_RATE),
            tmp_path,
        ]

        result = subprocess.run(cmd, capture_output=True)

        if result.returncode != 0:
            log.warning(f"[ffmpeg] FAIL {path.name}")
            return None

        if not Path(tmp_path).exists():
            return None

        y, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
        return y

    except Exception as e:
        log.warning(f"[ffmpeg exception] {path.name}: {e}")
        return None

    finally:
        if tmp_path and Path(tmp_path).exists():
            Path(tmp_path).unlink()


def load_via_librosa(path: Path) -> np.ndarray | None:
    try:
        y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        return y
    except:
        return None


def load_via_soundfile(path: Path) -> np.ndarray | None:
    try:
        import soundfile as sf

        y, sr = sf.read(str(path), always_2d=False)

        if y.ndim > 1:
            y = y.mean(axis=1)

        if sr != SAMPLE_RATE:
            y = librosa.resample(
                y.astype(np.float32),
                orig_sr=sr,
                target_sr=SAMPLE_RATE
            )

        return y.astype(np.float32)

    except:
        return None


def load_audio(path: Path) -> np.ndarray | None:
    if path.suffix.lower() == ".wav":
        y = load_via_librosa(path)
        if y is not None:
            return y

    y = load_via_ffmpeg(path)
    if y is not None:
        return y

    y = load_via_librosa(path)
    if y is not None:
        return y

    y = load_via_soundfile(path)
    if y is not None:
        return y

    log.error(f"FAILED LOAD: {path.name}")
    return None


# ─── FEATURES ─────────────────────────────
def pad_or_trim(y):
    target = int(SAMPLE_RATE * DURATION)

    if len(y) >= target:
        return y[:target]

    return np.pad(y, (0, target - len(y)))


def normalize(y):
    peak = np.max(np.abs(y))

    if peak > 0:
        return y / peak

    return y


def extract_mfcc(y):
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    return np.vstack([mfcc, delta, delta2]).astype(np.float32)


def extract_melspec(y):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        fmin=FMIN,
        fmax=FMAX,
    )

    return librosa.power_to_db(mel).astype(np.float32)


# ─── COLLECT ─────────────────────────────
def collect_audio_files():
    if not RAW_AUDIO.exists():
        raise FileNotFoundError(RAW_AUDIO)

    records = []

    for fpath in RAW_AUDIO.rglob("*"):
        if not fpath.is_file():
            continue

        if fpath.suffix.lower() not in {".m4a", ".wav", ".mp3", ".flac", ".ogg"}:
            continue

        records.append({
            "path": fpath,
            "subject": f"session_{fpath.stem.replace('ses', '')}",
            "label": -1,
            "filename": fpath.name,
        })

    return records


# ─── PROCESS ─────────────────────────────
def process_file(record):
    path = record["path"]
    subject = record["subject"]
    stem = path.stem

    feat_dir = PROC_FEAT / subject
    spec_dir = PROC_SPEC / subject

    feat_dir.mkdir(parents=True, exist_ok=True)
    spec_dir.mkdir(parents=True, exist_ok=True)

    feat_path = feat_dir / f"{stem}.npy"
    spec_path = spec_dir / f"{stem}.npy"

    if feat_path.exists() and spec_path.exists():
        return {
            **record,
            "feat_path": str(feat_path.relative_to(PROJECT_ROOT)),
            "spec_path": str(spec_path.relative_to(PROJECT_ROOT)),
        }

    y = load_audio(path)

    if y is None:
        return None

    y = normalize(y)
    y = pad_or_trim(y)

    mfcc = extract_mfcc(y)
    mel = extract_melspec(y)

    np.save(feat_path, mfcc)
    np.save(spec_path, mel)

    return {
        **record,
        "feat_path": str(feat_path.relative_to(PROJECT_ROOT)),
        "spec_path": str(spec_path.relative_to(PROJECT_ROOT)),
    }


# ─── SAVE ─────────────────────────────
def save_labels(processed):
    df = pd.DataFrame(processed)

    out_csv = PROJECT_ROOT / "data" / "processed" / "ws3d" / "labels.csv"

    df.to_csv(out_csv, index=False)

    log.info(f"Saved labels: {len(df)}")


# ─── MAIN ─────────────────────────────
def main():
    log.info("=== WS3D preprocessing start ===")
    ensure_dirs()

    records = collect_audio_files()
    processed = []
    failed = []

    for i, record in enumerate(tqdm(records, desc="Processing"), start=1):
        result = process_file(record)

        if result:
            processed.append(result)
        else:
            failed.append(record)

    save_labels(processed)

    print(f"\n=== HOTOVO ===  OK: {len(processed)}  FAIL: {len(failed)}")
    log.info(f"=== HOTOVO ===  OK: {len(processed)}  FAIL: {len(failed)}")

    if failed:
        for r in failed:
            log.warning(f"FAILED: {r['path']}")


if __name__ == "__main__":
    main()