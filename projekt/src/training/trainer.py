from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.config import Config


class Trainer:
    """
    Generic trainer for binary classification.

    Expected model output:
        logits of shape (B,) or (B, 1)

    Expected labels:
        0/1
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dataset_name: str,
        model_type: str,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 20,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.dataset_name = dataset_name
        self.model_type = model_type
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.device = device

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.save_dir = Path(Config.MODELS_DIR)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.best_model_path = self.save_dir / f"{dataset_name}_{model_type}_best.pt"

    def _move_batch(self, batch: Any):
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        elif isinstance(batch, dict):
            if "x" in batch and "y" in batch:
                x, y = batch["x"], batch["y"]
            elif "inputs" in batch and "labels" in batch:
                x, y = batch["inputs"], batch["labels"]
            else:
                raise ValueError("Unsupported dict batch format.")
        else:
            raise ValueError("Unsupported batch format.")

        x = x.to(self.device)
        y = y.to(self.device).float().view(-1)
        return x, y

    def _run_epoch(self, dataloader, training: bool = True) -> Dict[str, float]:
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_samples = 0

        all_probs = []
        all_preds = []
        all_labels = []

        progress = tqdm(dataloader, leave=False, desc="Train" if training else "Val")

        for batch in progress:
            x, y = self._move_batch(batch)

            if training:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(training):
                logits = self.model(x).view(-1)
                loss = self.criterion(logits, y)

                if training:
                    loss.backward()
                    self.optimizer.step()

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()

            bs = y.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

            all_probs.extend(probs.detach().cpu().numpy().tolist())
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(y.detach().cpu().numpy().astype(int).tolist())

            progress.set_postfix(loss=f"{total_loss / max(total_samples, 1):.4f}")

        avg_loss = total_loss / max(total_samples, 1)

        # local import to avoid circular imports
        from src.training.metrics import compute_metrics

        metrics = compute_metrics(
            y_true=np.array(all_labels),
            y_pred=np.array(all_preds),
            y_prob=np.array(all_probs),
        )
        metrics["loss"] = float(avg_loss)
        return metrics

    def fit(self, train_loader, val_loader) -> Dict[str, list]:
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "train_precision": [],
            "train_recall": [],
            "train_f1": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
        }

        best_f1 = -1.0
        best_state = copy.deepcopy(self.model.state_dict())

        for epoch in range(1, self.epochs + 1):
            print(f"\nEpoch [{epoch}/{self.epochs}]")

            train_metrics = self._run_epoch(train_loader, training=True)
            val_metrics = self._run_epoch(val_loader, training=False)

            history["train_loss"].append(train_metrics["loss"])
            history["train_accuracy"].append(train_metrics["accuracy"])
            history["train_precision"].append(train_metrics["precision"])
            history["train_recall"].append(train_metrics["recall"])
            history["train_f1"].append(train_metrics["f1"])

            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["val_precision"].append(val_metrics["precision"])
            history["val_recall"].append(val_metrics["recall"])
            history["val_f1"].append(val_metrics["f1"])

            print(
                f"Train | loss={train_metrics['loss']:.4f} "
                f"acc={train_metrics['accuracy']:.4f} "
                f"prec={train_metrics['precision']:.4f} "
                f"rec={train_metrics['recall']:.4f} "
                f"f1={train_metrics['f1']:.4f}"
            )
            print(
                f"Val   | loss={val_metrics['loss']:.4f} "
                f"acc={val_metrics['accuracy']:.4f} "
                f"prec={val_metrics['precision']:.4f} "
                f"rec={val_metrics['recall']:.4f} "
                f"f1={val_metrics['f1']:.4f}"
            )

            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                best_state = copy.deepcopy(self.model.state_dict())
                torch.save(best_state, self.best_model_path)
                print(f"  -> uložený nový best model: {self.best_model_path}")

        self.model.load_state_dict(best_state)
        return history

    @torch.no_grad()
    def evaluate(self, dataloader) -> Dict[str, np.ndarray]:
        self.model.eval()

        all_probs = []
        all_preds = []
        all_labels = []

        for batch in tqdm(dataloader, leave=False, desc="Test"):
            x, y = self._move_batch(batch)

            logits = self.model(x).view(-1)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()

            all_probs.extend(probs.detach().cpu().numpy().tolist())
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(y.detach().cpu().numpy().astype(int).tolist())

        return {
            "all_probs": np.array(all_probs, dtype=float),
            "all_preds": np.array(all_preds, dtype=int),
            "all_labels": np.array(all_labels, dtype=int),
        }