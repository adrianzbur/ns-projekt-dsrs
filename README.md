# ns-projekt-dsrs
Detekcia stresu z rečového signálu, Python – Neurónové siete

Projekt porovnáva dva prístupy k detekcii stresu z reči:

- **Nepriama detekcia:** TESS → stres aproximovaný z emócií (angry/fear/disgust/sad → stres)
- **Priama detekcia:** WS3D → emócie zakódované v názve súboru (angry/sadness → stres)

Pre každý dataset trénujeme dva modely:
- **MLP** – nad ručne extrahovanými príznakmi (MFCC + delta + delta2, shape 120×T)
- **CNN** – nad log-Mel spektrogramom (shape 1×128×T)

---

## Štruktúra projektu

```
ns-projekt-dsrs/
│
├── data/
│   ├── raw/
│   │   ├── ws3d/                  ← AudioData/ (ses_a03.wav, ses_n04.wav, ...)
│   │   └── tess/                  ← OAF_angry/, YAF_neutral/, ...
│   └── processed/
│       ├── ws3d/
│       │   ├── labels.csv         ← vygenerovaný prepare_ws3d.py
│       │   ├── features/          ← .npy shape (120, T)
│       │   └── spectrograms/      ← .npy shape (128, T)
│       └── tess/
│           ├── metadata.csv       ← vygenerovaný prepare_tess.py
│           ├── features/          ← .npy shape (13, T)
│           └── spectrograms/      ← .npy shape (128, T)
│
├── scripts/
│   ├── prepare_ws3d.py            ← preprocessing WS3D datasetu
│   └── prepare_tess.py            ← preprocessing TESS datasetu
│
├── src/
│   ├── config.py                  ← jeden zdroj pravdy pre všetky parametre
│   │
│   ├── data/
│   │   ├── tess_dataset.py        ← TessDataset (stratified split)
│   │   ├── ws3d_dataset.py        ← Ws3dDataset (speaker-independent split)
│   │   └── transforms.py          ← GaussianNoise, SpecAugment, SpectrogramNoise
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
│   │   ├── ws3d_mlp_best.pt
│   │   └── ws3d_cnn_best.pt
│   ├── plots/                     ← všetky grafy
│   ├── logs/                      ← logy z preprocessingu
│   └── summary_metrics.json       ← metriky všetkých experimentov
│
├── notebooks/
│   ├── 01_eda_tess.ipynb
│   ├── 02_eda_ws3d.ipynb
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

**WS3D** – nahrajte audio súbory do `data/raw/ws3d/AudioData/`, potom:
```bash
python scripts/prepare_ws3d.py
```

> Po preprocessingu vzniknú súbory `data/processed/tess/metadata.csv`
> a `data/processed/ws3d/labels.csv` ktoré používajú Dataset triedy.

---

### 2. Tréning

```bash
# Jednotlivé kombinácie
python run.py --dataset tess --model mlp
python run.py --dataset tess --model cnn
python run.py --dataset ws3d --model mlp
python run.py --dataset ws3d --model cnn

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
| `ws3d_cnn_training_curves.png` | ... (analogicky pre každú kombináciu) |
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

### WS3D
- Audio nahrávky s emóciami zakódovanými v názve súboru
- Sample rate: 16 000 Hz
- Split: speaker-independent (train / val / test podľa subjektu)
- Mapovanie emócií z názvu súboru (`ses_a03.wav` → prefix `a` → angry → 1):

| Prefix | Emócia | Label |
|--------|--------|-------|
|   n    | neutral|  0    |
|   h    |  happy |  0    |
|   c    | calm,  |       |
|   ps   |pleasant|  0    |
|        |surprise|       |
|   a    | angry  |  1    |
|   sa   | sadness|  1    |
|   d,f  | disgust, fear | 1 |

---

## Modely

### MLP
- **Vstup:** MFCC + delta + delta2, shape `(120, T)` → priemeruje cez časovú os → `(B, 120)`
- **Architektúra:** BN → [Linear → BN → ReLU → Dropout] × 3 → Linear
- **Skryté vrstvy:** 256 → 128 → 64
- **Výstup:** logit `(B,)` → BCEWithLogitsLoss

### CNN
- **Vstup:** log-Mel spektrogram, shape `(B, 1, 128, T)`
- **Architektúra:** [Conv2d → BN → ReLU → SEBlock → MaxPool] × 3 → GAP → Dropout → Linear
- **Kanály:** 32 → 64 → 128
- **Výstup:** logit `(B,)` → BCEWithLogitsLoss
