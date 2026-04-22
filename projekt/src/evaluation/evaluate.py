from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

# ── výstupný priečinok pre grafy ─────────────────────────────────────────────
_PLOTS_DIR = Path("outputs/plots")
_PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT FUNKCIE  (volané z run.py)
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_curves(
    history: Dict[str, list],
    tag: str,
    test_acc: float | None = None,
) -> None:
    """
    Kreslí loss a accuracy krivky pre train/val.

    Args:
        history : dict so kľúčmi train_loss, val_loss, train_accuracy, val_accuracy ...
        tag     : názov kombinácie (napr. 'tess_mlp') – použije sa v názve súboru
        test_acc: ak je zadané, pridá horizontálnu čiaru s test accuracy
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ── Loss ──────────────────────────────────────────────────────────────
    axes[0].plot(epochs, history["train_loss"], label="Train loss")
    axes[0].plot(epochs, history["val_loss"],   label="Val loss", linestyle="--")
    axes[0].set_title(f"{tag} – Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── Accuracy ──────────────────────────────────────────────────────────
    axes[1].plot(epochs, history["train_accuracy"], label="Train acc")
    axes[1].plot(epochs, history["val_accuracy"],   label="Val acc", linestyle="--")
    if test_acc is not None:
        axes[1].axhline(y=test_acc, color="red", linestyle=":", label=f"Test acc={test_acc:.3f}")
    axes[1].set_title(f"{tag} – Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out = _PLOTS_DIR / f"{tag}_training_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Training curves → {out}")


def plot_confusion_matrix(
    cm: np.ndarray | list,
    tag: str,
    class_names: list[str] | None = None,
    accuracy: float | None = None,
    auc: float | None = None,
) -> None:
    """
    Vizualizuje confusion matrix.

    Args:
        cm          : 2D numpy array alebo list (výstup z compute_metrics)
        tag         : názov kombinácie – použije sa v názve súboru
        class_names : zoznam názvov tried (default ["0","1"])
        accuracy    : ak je zadané, zobrazí sa v titulku
        auc         : ak je zadané, zobrazí sa v titulku
    """
    cm = np.array(cm)
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
            )

    title = f"{tag} – Confusion Matrix"
    if accuracy is not None:
        title += f"\nAcc={accuracy:.3f}"
    if auc is not None:
        title += f"  AUC={auc:.3f}"

    ax.set_title(title)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()

    out = _PLOTS_DIR / f"{tag}_confusion_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Confusion matrix → {out}")


def plot_roc_curves(
    results: Dict[str, Dict[str, Any]],
) -> None:
    """
    Kreslí ROC krivky pre viacero modelov do jedného grafu.

    Args:
        results : {
            "tess_mlp": {"y_true": np.ndarray, "y_prob": np.ndarray, "roc_auc": float},
            ...
        }
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    for tag, r in results.items():
        y_true = np.array(r["y_true"])
        y_prob = np.array(r["y_prob"])
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = r.get("roc_auc", auc(fpr, tpr))
        ax.plot(fpr, tpr, label=f"{tag}  (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = _PLOTS_DIR / "roc_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] ROC curves → {out}")


def plot_comparison_bar(
    all_metrics: Dict[str, Dict[str, Any]],
    metrics_to_plot: list[str] | None = None,
) -> None:
    """
    Skupinový stĺpcový graf porovnávajúci metriky naprieč modelmi.

    Args:
        all_metrics     : {"tess_mlp": {"accuracy": ..., "f1": ..., ...}, ...}
        metrics_to_plot : ktoré metriky zobraziť (default: accuracy, precision, recall, f1)
    """
    if metrics_to_plot is None:
        metrics_to_plot = ["accuracy", "precision", "recall", "f1"]

    tags = list(all_metrics.keys())
    x = np.arange(len(metrics_to_plot))
    width = 0.8 / max(len(tags), 1)

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, tag in enumerate(tags):
        values = [float(all_metrics[tag].get(m, 0.0)) for m in metrics_to_plot]
        offset = (i - len(tags) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=tag)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out = _PLOTS_DIR / "comparison_bar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Comparison bar → {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  PÔVODNÉ FUNKCIE (zachované pre spätnú kompatibilitu)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1]
    if isinstance(batch, dict):
        if "x" in batch and "y" in batch:
            return batch["x"], batch["y"]
        if "inputs" in batch and "labels" in batch:
            return batch["inputs"], batch["labels"]
    raise ValueError("Unsupported batch format.")


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: Iterable,
    device: str | torch.device = "cpu",
    criterion: Optional[torch.nn.Module] = None,
    class_names: Optional[list[str]] = None,
) -> Dict[str, Any]:
    model.eval()
    model.to(device)

    total_loss = 0.0
    total_samples = 0
    all_targets = []
    all_preds = []

    for batch in dataloader:
        inputs, labels = _extract_batch(batch)
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)

        if outputs.ndim == 1 or (outputs.ndim == 2 and outputs.shape[1] == 1):
            logits = outputs.squeeze(-1)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            if criterion is not None:
                loss = criterion(logits, labels.float())
        else:
            preds = torch.argmax(outputs, dim=1)
            if criterion is not None:
                loss = criterion(outputs, labels.long())

        if criterion is not None:
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        all_targets.extend(_to_numpy(labels).astype(int).tolist())
        all_preds.extend(_to_numpy(preds).astype(int).tolist())

    y_true = np.array(all_targets, dtype=int)
    y_pred = np.array(all_preds, dtype=int)
    avg_loss = total_loss / total_samples if criterion is not None and total_samples > 0 else None

    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1        = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    labels_sorted = sorted(np.unique(np.concatenate([y_true, y_pred])))
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)

    if class_names is None:
        class_names = [str(label) for label in labels_sorted]

    return {
        "loss": avg_loss,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "class_names": class_names,
    }


def print_evaluation_summary(results: Dict[str, Any]) -> None:
    print("\n=== Evaluation summary ===")
    if results.get("loss") is not None:
        print(f"Loss:      {results['loss']:.4f}")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-score:  {results['f1']:.4f}")
    print("Confusion matrix:")
    print(np.array(results["confusion_matrix"]))


def save_evaluation_results(
    results: Dict[str, Any],
    output_dir: str | Path,
    prefix: str = "eval",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / f"{prefix}_metrics.json"
    metrics_to_save = {k: v for k, v in results.items() if k not in {"y_true", "y_pred"}}
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_to_save, f, indent=4, ensure_ascii=False)

    cm = np.array(results["confusion_matrix"])
    tag = prefix
    plot_confusion_matrix(cm, tag, class_names=results.get("class_names"))