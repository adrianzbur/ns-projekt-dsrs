# ns-projekt-dsrs
Detekcia stresu z recoveho signalu, Python - Neuronove siete

Projekt porovnáva dva prístupy k detekcii stresu z reči:

Priama detekcia: ws3d -> explicitné anotácie stresu (baseline/stress/amusement)
Nepriama detekcia: TESS -> stres aproximovaný z emócií (anger/fear → stres)

Pre každý dataset trénujeme dva modely:
MLP – nad ručne extrahovanými príznakmi (MFCC + delta + energia + ZCR)
CNN – nad log-Mel spektrogramom


ns-projekt-dsrs/
│
├── data/
│   ├── raw/
│   │   ├── ws3d/              ← S2/, S3/, ... (z Kaggle)
│   │   └── tess/               ← OAF_angry/, YAF_neutral/, ...
│   └── processed/
│       ├── ws3d/
│       │   ├── features/       ← .npy MFCC vektory
│       │   └── spectrograms/   ← .npy Mel spektrogramy
│       └── tess/
│           ├── features/
│           └── spectrograms/
│
├── scripts/                    ← jednorazové skripty (spúšťajú sa raz)
│   ├── prepare_ws3d.py        ← stiahnutie + preprocessing ws3d
│   └── prepare_tess.py         ← resample + preprocessing TESS
│
├── src/                        ← opakovane použiteľný kód
│   ├── __init__.py
│   ├── config.py               ← JEDEN zdroj pravdy pre všetky parametre
│   │
│   ├── data/                   ← načítanie + Dataset triedy
│   │   ├── __init__.py
│   │   ├── tess_dataset.py     ← TessDataset(torch.utils.data.Dataset)
│   │   ├── ws3d_dataset.py     ← ws3dDataset(...)
│   │   └── transforms.py       ← augmentácie, normalizácia
│   │
│   ├── features/               ← extrakcia príznakov
│   │   ├── __init__.py
│   │   ├── extractor.py        ← MFCC, pitch, energia (librosa)
│   │   └── spectrogram.py      ← Mel spektrogram
│   │
│   ├── models/                 ← definície architektúr
│   │   ├── __init__.py
│   │   ├── mlp.py
│   │   └── cnn.py
│   │
│   ├── training/               ← tréningová logika
│   │   ├── __init__.py
│   │   ├── trainer.py          ← univerzálny Trainer (funguje pre oba datasety)
│   │   └── metrics.py          ← accuracy, F1, AUC, confusion matrix
│   │
│   └── evaluation/
│       ├── __init__.py
│       └── evaluate.py         ← porovnanie TESS vs WESAD, MLP vs CNN
│
├── outputs/
│   ├── models/                 ← uložené .pt checkpointy
│   │   ├── tess_mlp_best.pt
│   │   ├── tess_cnn_best.pt
│   │   ├── ws3d_mlp_best.pt
│   │   └── ws3d_cnn_best.pt
│   ├── plots/                  ← všetky grafy na jednom mieste
│   └── logs/                   ← CSV tréningové logy
│
├── notebooks/
│   ├── 01_eda_tess.ipynb
│   ├── 02_eda_ws3d.ipynb
│   └── 03_results_comparison.ipynb
│
├── report/
│   └── report_dsrs.pdf         ← váš aktuálny report
│
├── requirements.txt
├── README.md
└── run.py                      ← hlavný vstupný bod (spúšťa celý pipeline)


# Inštalácia:
bashgit clone <repo-url>
cd ns-projekt-dsrs
pip install -r requirements.txt

Použitie
1. Príprava dát
TESS – nahrajte .wav súbory do data/raw/tess/, potom:
bashpython scripts/prepare_tess.py
WS3D – potrebujete Kaggle API token (~/.kaggle/kaggle.json):
bashpython scripts/prepare_wesad.py

2. Tréning
bash# Jeden model
python run.py --dataset tess  --model mlp
python run.py --dataset tess  --model cnn
python run.py --dataset ws3d --model mlp
python run.py --dataset ws3d --model cnn

# Všetky 4 kombinácie naraz
python run.py --dataset all --model all

# S vlastnými parametrami
python run.py --dataset tess --model mlp --epochs 100 --device cuda
3. Výsledky
Grafy sa automaticky ukladajú do outputs/plots/:

tess_mlp_training_curves.png
tess_cnn_confusion_matrix.png
roc_curves_comparison.png
comparison_bar_chart.png


Datasety
TESS (Toronto Emotional Speech Set)
2 800 nahrávok, 2 herečky, 7 emócií
Sample rate: 24 414 Hz → resampleujeme na 16 000 Hz
Mapovanie: neutral/happy → 0, angry/fear/disgust/sad/ps → 1

ws3d (Wearable Stress and Affect Detection)
15 subjektov, nositeľné senzory + mikrofón
Sample rate: 16 000 Hz (audio)
Mapovanie: baseline/amusement → 0, stress → 1
Rozdelenie je speaker-independent (celí subjekti ↔ train/test)


Modely
# MLP

Vstup: 164-dimenzionálny vektor (MFCC-40 × 4 + RMS × 2 + ZCR × 2)
Architektúra: BN → [Linear→BN→ReLU→Dropout] × 3 → Linear
Skryté vrstvy: 256 → 128 → 64

# CNN
Vstup: log-Mel spektrogram (1, 64, 128)
Architektúra: [Conv2d→BN→ReLU→SEBlock→MaxPool] × 3 → GAP → Linear
Kanály: 32 → 64 → 128