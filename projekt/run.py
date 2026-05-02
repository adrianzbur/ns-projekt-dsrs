"""
run.py – Hlavný vstupný bod projektu.

Spúšťa tréning a evaluáciu modelov pre vybrané kombinácie dataset × model.

Použitie:
    python run.py --dataset tess   --model mlp
    python run.py --dataset cremad --model cnn
    python run.py --dataset all    --model all    # všetky 4 kombinácie

Príklady:
    python run.py --dataset tess   --model mlp --epochs 50
    python run.py --dataset cremad --model all --device cuda
    python run.py --dataset all    --model all --deterministic
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.config import (
    SEED,
    DEVICE,
    MLP_BATCH_SIZE,
    MLP_EPOCHS,
    MLP_LR,
    MLP_WEIGHT_DECAY,
    CNN_BATCH_SIZE,
    CNN_EPOCHS,
    CNN_LR,
    CNN_WEIGHT_DECAY,
    TESS_FEATURE_DIM,
    CREMAD_FEATURE_DIM,
)
from src.data.tess_dataset   import TessDataset
from src.data.cremad_dataset import CremadDataset
from src.data.transforms import get_train_transforms_mlp, get_train_transforms_cnn
from src.models.mlp import MLP
from src.models.cnn import CNN
from src.training.trainer  import Trainer
from src.training.metrics  import compute_metrics, print_metrics, compare_metrics
from src.evaluation.evaluate import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_comparison_bar,
)

torch.manual_seed(SEED)
np.random.seed(SEED)

# Mapovanie názvu datasetu → trieda a feature dim pre MLP
DATASET_CLASS = {
    "tess":   TessDataset,
    "cremad": CremadDataset,
}

DATASET_FEATURE_DIM = {
    "tess":   TESS_FEATURE_DIM,
    "cremad": CREMAD_FEATURE_DIM,
}

# Všetky dostupné datasety pre --dataset all
ALL_DATASETS = ["tess", "cremad"]


def resolve_device(requested_device: str) -> str:
    req = requested_device.lower().strip()
    if req.startswith("cuda"):
        if torch.cuda.is_available():
            return "cuda"
        print("[WARN] CUDA nie je dostupná, prepínam na CPU.")
        return "cpu"
    return "cpu"


def configure_reproducibility(deterministic: bool):
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    else:
        torch.backends.cudnn.benchmark = True


def dataloader_runtime_params(device: str) -> dict:
    if device.startswith("cuda"):
        return {"num_workers": 2, "pin_memory": True}
    return {"num_workers": 0, "pin_memory": False}


def get_dataloaders(dataset_name: str, model_type: str, batch_size: int, device: str):
    mode = "features" if model_type == "mlp" else "spectrograms"
    train_transform = (
        get_train_transforms_mlp() if model_type == "mlp"
        else get_train_transforms_cnn()
    )

    DatasetClass = DATASET_CLASS[dataset_name]

    train_ds = DatasetClass(mode=mode, split="train", transform=train_transform)
    val_ds   = DatasetClass(mode=mode, split="val")
    test_ds  = DatasetClass(mode=mode, split="test")

    dl_params = dataloader_runtime_params(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **dl_params)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **dl_params)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **dl_params)

    return train_loader, val_loader, test_loader


def build_model(dataset_name: str, model_type: str):
    if model_type == "mlp":
        input_dim = DATASET_FEATURE_DIM[dataset_name]
        return MLP(input_dim=input_dim)
    return CNN()


def run_single(dataset_name: str, model_type: str, epochs: int, device: str) -> dict:
    tag = f"{dataset_name}_{model_type}"
    print(f"\n{'═' * 60}")
    print(f"  Spúšťam: {tag.upper()} | device={device} | epochs={epochs}")
    print(f"{'═' * 60}")

    bs = MLP_BATCH_SIZE if model_type == "mlp" else CNN_BATCH_SIZE
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset_name, model_type, bs, device
    )

    model = build_model(dataset_name, model_type)
    lr = MLP_LR if model_type == "mlp" else CNN_LR
    wd = MLP_WEIGHT_DECAY if model_type == "mlp" else CNN_WEIGHT_DECAY

    trainer = Trainer(
        model=model,
        dataset_name=dataset_name,
        model_type=model_type,
        lr=lr,
        weight_decay=wd,
        epochs=epochs,
        device=device,
    )

    history     = trainer.fit(train_loader, val_loader)
    test_result = trainer.evaluate(test_loader)

    metrics = compute_metrics(
        test_result["all_labels"],
        test_result["all_preds"],
        test_result["all_probs"],
    )
    print_metrics(metrics, title=f"{tag.upper()} – Testové výsledky")

    plot_training_curves(history, tag, test_acc=metrics["accuracy"])
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        tag,
        accuracy=metrics["accuracy"],
        auc=metrics["roc_auc"],
    )

    return {
        "tag":     tag,
        "history": history,
        "metrics": metrics,
        "y_true":  test_result["all_labels"],
        "y_prob":  test_result["all_probs"],
        "roc_auc": metrics["roc_auc"],
    }


def to_serializable_metrics(metrics: dict) -> dict:
    out = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def save_summary_json(all_results: dict, out_dir: str = "outputs"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "summary_metrics.json")
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "results": {
            tag: to_serializable_metrics(res["metrics"])
            for tag, res in all_results.items()
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Súhrn metrík uložený do: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Detekcia stresu z rečového signálu – tréning modelov"
    )
    parser.add_argument(
        "--dataset",
        choices=["tess", "cremad", "all"],
        default="tess",
        help="Dataset: tess | cremad | all (= tess + cremad)",
    )
    parser.add_argument(
        "--model",
        choices=["mlp", "cnn", "all"],
        default="mlp",
    )
    parser.add_argument("--epochs",      type=int, default=None)
    parser.add_argument("--device",      default=DEVICE)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    if args.epochs is not None and args.epochs <= 0:
        raise ValueError("--epochs musí byť kladné celé číslo.")

    device = resolve_device(args.device)
    configure_reproducibility(args.deterministic)

    # --dataset all spúšťa tess + cremad (ws3d bol nahradený)
    if args.dataset == "all":
        datasets = ALL_DATASETS
    else:
        datasets = [args.dataset]

    models = ["mlp", "cnn"] if args.model == "all" else [args.model]
    combos = [(d, m) for d in datasets for m in models]

    all_results = {}
    for dataset_name, model_type in combos:
        ep = args.epochs if args.epochs is not None else (
            MLP_EPOCHS if model_type == "mlp" else CNN_EPOCHS
        )
        result = run_single(dataset_name, model_type, ep, device)
        all_results[result["tag"]] = result

    if len(all_results) > 1:
        print(f"\n{'═' * 60}")
        print("  SÚHRNNÉ POROVNANIE")
        compare_metrics({tag: r["metrics"] for tag, r in all_results.items()})

        plot_roc_curves({
            tag: {"y_true": r["y_true"], "y_prob": r["y_prob"], "roc_auc": r["roc_auc"]}
            for tag, r in all_results.items()
        })
        plot_comparison_bar({tag: r["metrics"] for tag, r in all_results.items()})

    save_summary_json(all_results, out_dir="outputs")

    print(f"\n{'═' * 60}")
    print("  Hotovo! Výstupy sú v priečinku outputs/")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()