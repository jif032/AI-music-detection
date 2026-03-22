# =============================================================================
# evaluate.py
# =============================================================================
# Evaluation utilities for both chunk-level and track-level metrics.
#
# Chunk-level  : each 10-second chunk is treated as an independent prediction.
#
# Track-level  : chunk probabilities for the same track are aggregated
#                via mean pooling (soft voting) before thresholding.
#                This is more meaningful for real-world use where you
#                have access to the full track.
#
# Per-source breakdown is always computed so you can see how the model
# performs separately on FMA, Suno, and Udio.
# =============================================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from pathlib import Path
import logging
import json

log = logging.getLogger(__name__)


# ── Core metric functions ─────────────────────────────────────────────────────


def compute_chunk_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute classification metrics at chunk level.

    Parameters
    ----------
    labels    : (N,) int — ground truth (0=human, 1=ai)
    probs     : (N,) float — P(AI) predicted probability
    threshold : decision boundary for accuracy / F1

    Returns
    -------
    dict with keys: auroc, auprc, accuracy, f1
    """
    preds = (probs >= threshold).astype(int)

    # Guard against single-class edge cases
    if len(np.unique(labels)) < 2:
        auroc = auprc = 0.5
    elif len(np.unique(labels)) == 2:
        auroc = float(roc_auc_score(labels, probs))
        auprc = float(average_precision_score(labels, probs))
    else:
        raise ValueError(
            "Labels have more than 2 classes — check data loading or model output"
        )

    return {
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }


def compute_track_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    track_ids: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Aggregate chunk probabilities per track (soft voting / mean pooling)
    then compute metrics at track level.

    Parameters
    ----------
    labels    : (N,) int — chunk-level ground truth
    probs     : (N,) float — chunk-level P(AI)
    track_ids : (N,) int — which track each chunk belongs to

    Returns
    -------
    dict with keys: auroc, auprc, accuracy, f1, n_tracks
    """
    track_data = {}
    for label, prob, tid in zip(labels, probs, track_ids):
        if tid not in track_data:
            track_data[tid] = {"labels": [], "probs": []}
        track_data[tid]["labels"].append(int(label))
        track_data[tid]["probs"].append(float(prob))

    track_labels = []
    track_probs = []
    for tid, data in track_data.items():
        # All chunks of a track must share the same label
        track_labels.append(data["labels"][0])
        track_probs.append(float(np.mean(data["probs"])))

    track_labels = np.array(track_labels)
    track_probs = np.array(track_probs)
    metrics = compute_chunk_metrics(track_labels, track_probs, threshold)
    metrics["n_tracks"] = len(track_labels)
    return metrics


def compute_per_source_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    sources: list[str],
    track_ids: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, dict]:
    """
    Compute chunk-level and track-level metrics broken down by source.

    Returns
    -------
    dict keyed by source_name, each value is
    {"chunk": metrics_dict, "track": metrics_dict}
    """
    sources_arr = np.array(sources)
    results = {}

    for source in np.unique(sources_arr):
        mask = sources_arr == source
        if mask.sum() == 0:
            continue
        chunk_m = compute_chunk_metrics(labels[mask], probs[mask], threshold)
        track_m = compute_track_metrics(
            labels[mask], probs[mask], track_ids[mask], threshold
        )
        results[source] = {"chunk": chunk_m, "track": track_m}

    return results


# ── Full evaluation pass ──────────────────────────────────────────────────────


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    split_name: str = "test",
    threshold: float = 0.5,
) -> dict:
    """
    Run a full evaluation pass over a DataLoader.

    Parameters
    ----------
    model      : trained model (already on device)
    loader     : DataLoader for the split to evaluate
    device     : torch device
    split_name : label for logging (e.g. "test", "val")
    threshold  : decision threshold for accuracy/F1

    Returns
    -------
    results dict with keys:
        labels, probs, track_ids, sources  — raw arrays
        chunk   — chunk-level metric dict
        track   — track-level metric dict
        per_source — per-source breakdown dict
    """
    model.eval()
    all_labels = []
    all_probs = []
    all_track_ids = []
    all_sources = []

    with torch.no_grad():
        for batch_inputs, batch_labels, batch_metas in loader:
            batch_inputs = batch_inputs.to(device)
            logits = model(batch_inputs)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

            all_probs.extend(probs.tolist())
            all_labels.extend(batch_labels.numpy().tolist())
            all_track_ids.extend([m["track_id"] for m in batch_metas])
            all_sources.extend([m["source_name"] for m in batch_metas])

    labels = np.array(all_labels, dtype=np.int32)
    probs = np.array(all_probs, dtype=np.float32)
    track_ids = np.array(all_track_ids, dtype=np.int64)  # ← int32 → int64

    chunk_metrics = compute_chunk_metrics(labels, probs, threshold)
    track_metrics = compute_track_metrics(labels, probs, track_ids, threshold)
    source_metrics = compute_per_source_metrics(
        labels, probs, all_sources, track_ids, threshold
    )

    results = {
        "labels": labels,
        "probs": probs,
        "track_ids": track_ids,
        "sources": all_sources,
        "chunk": chunk_metrics,
        "track": track_metrics,
        "per_source": source_metrics,
    }

    _log_evaluation_results(results, split_name)
    return results


# ── Logging ───────────────────────────────────────────────────────────────────


def _log_evaluation_results(results: dict, split_name: str) -> None:
    log.info(f"\n{'=' * 55}")
    log.info(f"Evaluation results — split='{split_name}'")
    log.info(f"{'=' * 55}")

    c = results["chunk"]
    t = results["track"]

    log.info(f"\nChunk-level  ({len(results['labels'])} chunks):")
    log.info(f"  AUROC    = {c['auroc']:.4f}")
    log.info(f"  AUPRC    = {c['auprc']:.4f}")
    log.info(f"  Accuracy = {c['accuracy']:.4f}")
    log.info(f"  F1       = {c['f1']:.4f}")

    log.info(f"\nTrack-level  ({t['n_tracks']} tracks):")
    log.info(f"  AUROC    = {t['auroc']:.4f}")
    log.info(f"  AUPRC    = {t['auprc']:.4f}")
    log.info(f"  Accuracy = {t['accuracy']:.4f}")
    log.info(f"  F1       = {t['f1']:.4f}")

    log.info(f"\nPer-source breakdown (chunk level):")
    for source, metrics in results["per_source"].items():
        cm = metrics["chunk"]
        tm = metrics["track"]
        log.info(
            f"  {source:8s} | "
            f"chunk: auroc={cm['auroc']:.4f} "
            f"auprc={cm['auprc']:.4f} "
            f"acc={cm['accuracy']:.4f} | "
            f"track: auroc={tm['auroc']:.4f} "
            f"acc={tm['accuracy']:.4f} "
            f"n={tm['n_tracks']}"
        )


# ── Results saving ────────────────────────────────────────────────────────────


def save_evaluation_results(
    results: dict,
    output_dir: Path,
    run_name: str,
    split_name: str,
) -> None:
    """
    Save evaluation results to disk.
      - predictions CSV  (chunk_id, label, prob, track_id, source)
      - metrics JSON     (chunk + track + per_source)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Predictions CSV
    preds_df = pd.DataFrame(
        {
            "label": results["labels"],
            "prob_ai": results["probs"],
            "track_id": results["track_ids"],
            "source_name": results["sources"],
            "pred": (results["probs"] >= 0.5).astype(int),
        }
    )
    preds_path = output_dir / f"{run_name}_{split_name}_predictions.csv"
    preds_df.to_csv(preds_path, index=False)

    # Metrics JSON
    metrics = {
        "chunk": results["chunk"],
        "track": results["track"],
        "per_source": {
            src: {
                "chunk": m["chunk"],
                "track": m["track"],
            }
            for src, m in results["per_source"].items()
        },
    }
    metrics_path = output_dir / f"{run_name}_{split_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    log.info(f"Predictions → {preds_path}")
    log.info(f"Metrics     → {metrics_path}")


# ── Training curve plotting ───────────────────────────────────────────────────


def plot_training_history(
    history_df: pd.DataFrame,
    output_path: Path | None = None,
) -> None:
    """
    Plot loss, AUROC, and AUPRC curves for train and val splits.
    Saves to output_path if given, otherwise displays inline.
    """
    import matplotlib.pyplot as plt

    train_h = history_df[history_df["split"] == "train"]
    val_h = history_df[history_df["split"] == "val"]

    metrics_to_plot = [
        ("loss", "Loss"),
        ("chunk_auroc", "Chunk AUROC"),
        ("chunk_auprc", "Chunk AUPRC"),
        ("track_auroc", "Track AUROC"),
        ("track_auprc", "Track AUPRC"),
    ]

    fig, axes = plt.subplots(
        1, len(metrics_to_plot), figsize=(4 * len(metrics_to_plot), 4)
    )

    for ax, (col, title) in zip(axes, metrics_to_plot):
        ax.plot(
            train_h["epoch"],
            train_h[col],
            label="train",
            color="#4C72B0",
            linewidth=1.5,
        )
        ax.plot(
            val_h["epoch"],
            val_h[col],
            label="val",
            color="#DD8452",
            linewidth=1.5,
            linestyle="--",
        )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("Training History", fontsize=12)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        log.info(f"Training curve saved → {output_path}")
    plt.show()
