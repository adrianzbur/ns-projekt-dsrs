from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def _extract_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Podporované formáty batchu:
    1. (inputs, labels)
    2. {"x": inputs, "y": labels}
    3. {"inputs": inputs, "labels": labels}
    """
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1]

    if isinstance(batch, dict):
        if "x" in batch and "y" in batch:
            return batch["x"], batch["y"]
        if "inputs" in batch and "labels" in batch:
            return batch["inputs"], batch["labels"]

    raise ValueError(
        "Unsupported batch format. Expected (inputs, labels), "
        '{"x": ..., "y": ...}, or {"inputs": ..., "labels": ...}.'
    )


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
    """
    Vyhodnotí klasifikačný model na dataloaderi.

    Predpokladá:
    - multiclass logits tvaru [B, C]
      alebo
    - binary logits tvaru [B] / [B, 1]

    Returns dict:
    - loss
    - accuracy
    - precision
    - recall
    - f1
    - confusion_matrix
    - y_true
    - y_pred
    - class_names
    """
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

        # Binary classification: output [B] alebo [B, 1]
        if outputs.ndim == 1 or (outputs.ndim == 2 and outputs.shape[1] == 1):
            logits = outputs.squeeze(-1)
            preds = (torch.sigmoid(logits) >= 0.5).long()

            if criterion is not None:
                # BCEWithLogitsLoss očakáva float labels
                loss = criterion(logits, labels.float())
        else:
            # Multiclass classification: output [B, C]
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

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    labels_sorted = sorted(np.unique(np.concatenate([y_true, y_pred])))
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)

    if class_names is None:
        class_names = [str(label) for label in labels_sorted]

    results = {
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

    return results


def print_evaluation_summary(results: Dict[str, Any]) -> None:
    """
    Pekný print metrík do konzoly.
    """
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
    """
    Uloží:
    - metrics JSON
    - confusion matrix PNG
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / f"{prefix}_metrics.json"
    cm_path = output_dir / f"{prefix}_confusion_matrix.png"

    # JSON bez raw predikcií, aby súbor nebol zbytočne veľký
    metrics_to_save = {
        key: value
        for key, value in results.items()
        if key not in {"y_true", "y_pred"}
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_to_save, f, indent=4, ensure_ascii=False)

    cm = np.array(results["confusion_matrix"])
    class_names = results.get("class_names", [str(i) for i in range(cm.shape[0])])

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=200, bbox_inches="tight")
    plt.close()