# ns-projekt-dsrs
Detekcia stresu z rečového signálu, Python – Neurónové siete

Projekt porovnáva dva prístupy k detekcii stresu z reči:

- **Nepriama detekcia:** TESS → stres aproximovaný z emócií (angry/fear/disgust/sad → stres)
- **Priama detekcia:** CREMA-D → emócie zakódované v názve súboru (ANG/DIS/FEA/SAD → stres)

Pre každý dataset trénujeme dva modely:
- **MLP** – nad ručne extrahovanými príznakmi (MFCC + delta + delta²), shape `(120, T)`
- **CNN** – nad log-Mel spektrogramom, shape `(1, 128, T)`

---

## Štruktúra projektu

```
ns-projekt-dsrs/
│
├── data/
│   ├── raw/
│   │   ├── cremad/
│   │   │   └── AudioWAV/          ← 1001_DFA_ANG_XX.wav, ...
│   │   └── tess/                  ← OAF_angry/, YAF_neutral/, ...
│   └── processed/
│       ├── cremad/
│       │   ├── metadata.csv       ← vygenerovaný prepare_cremad.py
│       │   ├── features/          ← .npy shape (120, T)
│       │   └── spectrograms/      ← .npy shape (128, T)
│       └── tess/
│           ├── metadata.csv       ← vygenerovaný prepare_tess.py
│           ├── features/          ← .npy shape (120, T)
│           └── spectrograms/      ← .npy shape (128, T)
│
├── scripts/
│   ├── prepare_cremad.py          ← preprocessing CREMA-D datasetu
│   └── prepare_tess.py            ← preprocessing TESS datasetu
│
├── src/
│   ├── config.py                  ← jeden zdroj pravdy pre všetky parametre
│   │
│   ├── data/
│   │   ├── tess_dataset.py        ← TessDataset (stratified split)
│   │   ├── cremad_dataset.py      ← CremadDataset (stratified split)
│   │   └── transforms.py         ← GaussianNoise, SpecAugment, SpectrogramNoise
│   │
│   ├── models/
│   │   ├── mlp.py                 ← MLP: vstup (B, D) alebo (B, D, T) → logit (B,)
│   │   └── cnn.py                 ← CNN so SE blokmi: vstup (B, 1, 128, T) → logit (B,)
│   │
│   ├── training/
│   │   ├── trainer.py             ← Trainer trieda (fit + evaluate)
│   │   └── metrics.py             ← compute_metrics, print_metrics, compare_metrics
│   │
│   └── evaluation/
│       └── evaluate.py            ← plot_training_curves, plot_confusion_matrix,
│                                     plot_roc_curves, plot_comparison_bar
│
├── outputs/
│   ├── models/                    ← najlepšie checkpointy (.pt)
│   │   ├── tess_mlp_best.pt
│   │   ├── tess_cnn_best.pt
│   │   ├── cremad_mlp_best.pt
│   │   └── cremad_cnn_best.pt
│   ├── plots/                     ← všetky grafy
│   ├── logs/                      ← logy z preprocessingu
│   └── summary_metrics.json       ← metriky všetkých experimentov
│
├── notebooks/
│   ├── 01_eda_tess.ipynb
│   ├── 02_eda_cremad.ipynb
│   └── 03_results_comparison.ipynb
│
├── requirements.txt
├── README.md
└── run.py                         ← hlavný vstupný bod
```

---

## Inštalácia

```bash
git clone <repo-url>
cd ns-projekt-dsrs
pip install -r requirements.txt
```

---

## Použitie

### 1. Príprava dát

**TESS** – nahrajte `.wav` súbory do `data/raw/tess/`, potom:
```bash
python scripts/prepare_tess.py
```

**CREMA-D** – nahrajte audio súbory do `data/raw/cremad/AudioWAV/`, potom:
```bash
python scripts/prepare_cremad.py
```

> Po preprocessingu vzniknú súbory `data/processed/tess/metadata.csv`
> a `data/processed/cremad/metadata.csv` ktoré používajú Dataset triedy.

---

### 2. Tréning

```bash
# Jednotlivé kombinácie
python run.py --dataset tess --model mlp
python run.py --dataset tess --model cnn
python run.py --dataset cremad --model mlp
python run.py --dataset cremad --model cnn

# Všetky 4 kombinácie naraz
python run.py --dataset all --model all

# S vlastnými parametrami
python run.py --dataset tess --model mlp --epochs 100 --device cuda

# Deterministický režim (reprodukovateľnosť)
python run.py --dataset all --model all --deterministic
```

---

### 3. Výsledky

Grafy sa automaticky ukladajú do `outputs/plots/`:

| Súbor | Obsah |
|---|---|
| `tess_mlp_training_curves.png` | Loss a accuracy krivky |
| `tess_mlp_confusion_matrix.png` | Confusion matrix s ACC a AUC |
| `cremad_cnn_training_curves.png` | ... (analogicky pre každú kombináciu) |
| `roc_curves.png` | ROC krivky všetkých modelov |
| `comparison_bar.png` | Porovnanie ACC/F1/Precision/Recall |

Súhrnné metriky všetkých experimentov: `outputs/summary_metrics.json`

---

## Datasety

### TESS (Toronto Emotional Speech Set)
- 2 800 nahrávok, 2 herečky, 7 emócií
- Sample rate: 24 414 Hz → resampleujeme na 16 000 Hz
- Split: stratifikovaný náhodný (train 60% / val 20% / test 20%)
- Mapovanie emócií na label:

| Emócia | Label |
|---|---|
| neutral, calm, ps | 0 (bez stresu) |
| angry, fear, disgust, sad | 1 (stres) |

### CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)
- 7 442 nahrávok, 91 hercov, 6 emócií, 4 úrovne intenzity
- Sample rate: 16 000 Hz
- Split: stratifikovaný náhodný (train 60% / val 20% / test 20%)
- Formát názvu súboru: `{ActorID}_{Sentence}_{Emotion}_{Intensity}.wav`
  - napr. `1001_DFA_ANG_XX.wav`
- Mapovanie emócií z názvu súboru na label:

| Kód | Emócia | Intenzita | Label |
|-----|--------|-----------|-------|
| ANG | Anger | LO / MD / HI / XX | 1 (stres) |
| DIS | Disgust | LO / MD / HI / XX | 1 (stres) |
| FEA | Fear | LO / MD / HI / XX | 1 (stres) |
| SAD | Sad | LO / MD / HI / XX | 1 (stres) |
| HAP | Happy | LO / MD / HI / XX | 0 (bez stresu) |
| NEU | Neutral | LO / MD / HI / XX | 0 (bez stresu) |

---

## Modely

### MLP
- **Vstup:** MFCC + delta + delta², shape `(120, T)` → priemeruje cez časovú os → `(B, 120)`
- **Architektúra:** BN → [Linear → BN → ReLU → Dropout] × 3 → Linear
- **Skryté vrstvy:** 256 → 128 → 64
- **Výstup:** logit `(B,)` → BCEWithLogitsLoss

### CNN
- **Vstup:** log-Mel spektrogram, shape `(B, 1, 128, T)`
- **Architektúra:** [Conv2d → BN → ReLU → SEBlock → MaxPool] × 3 → GAP → Dropout → Linear
- **Kanály:** 32 → 64 → 128
- **Výstup:** logit `(B,)` → BCEWithLogitsLoss

---

## Audio parametre

| Parameter | Hodnota |
|---|---|
| Sample rate | 16 000 Hz |
| Dĺžka nahrávky | 3.0 s |
| N_MFCC | 40 (+ delta + delta² = 120) |
| N_MELS | 128 |
| HOP_LENGTH | 512 |
| N_FFT | 2048 |