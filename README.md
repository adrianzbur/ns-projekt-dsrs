# ns-projekt-dsrs
Detekcia stresu z recoveho signalu, Python - Neuronove siete

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