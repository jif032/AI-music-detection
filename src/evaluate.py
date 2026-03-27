# =============================================================================
# evaluate.py
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
    Chunk-level metrics.

    Returns auroc, auprc (AI=positive), auprc_human (human=positive),
    accuracy, f1.
    """
    preds = (probs >= threshold).astype(int)

    if len(np.unique(labels)) < 2:
        return {
            "auroc": 0.5,
            "auprc": float(labels.mean()),
            "auprc_human": float(1 - labels.mean()),
            "accuracy": float(accuracy_score(labels, preds)),
            "f1": float(f1_score(labels, preds, zero_division=0)),
            "f1_human": float(f1_score(1 - labels, 1 - preds, zero_division=0)),
        }

    # AI = positive class
    auroc = float(roc_auc_score(labels, probs))
    auprc = float(average_precision_score(labels, probs))
    # Human = positive class (invert labels and probs)
    auprc_human = float(average_precision_score(1 - labels, 1 - probs))

    return {
        "auroc": auroc,
        "auprc": auprc,
        "auprc_human": auprc_human,
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "f1_human": float(f1_score(1 - labels, 1 - preds, zero_division=0)),
    }


def compute_track_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    track_ids: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Track-level metrics using mean-pooled chunk probabilities.
    """
    track_data = {}
    for label, prob, tid in zip(labels, probs, track_ids):
        if tid not in track_data:
            track_data[tid] = {"label": int(label), "probs": []}
        track_data[tid]["probs"].append(float(prob))

    track_labels = np.array([d["label"] for d in track_data.values()])
    track_probs = np.array([np.mean(d["probs"]) for d in track_data.values()])

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
    Compute metrics broken down by source.

    For sources with only one class (e.g. FMA=all human, Udio=all AI),
    AUROC is undefined. In this case we compute pairwise metrics:
    each AI source is paired with all human (FMA) samples.

    Returns dict keyed by source_name with chunk and track metrics.
    Also includes pairwise keys like "FMA_vs_Udio", "FMA_vs_Suno".
    """
    sources_arr = np.array(sources)
    results = {}

    # ── Per-source metrics (may be single-class) ──────────────────────────
    for source in np.unique(sources_arr):
        mask = sources_arr == source
        if mask.sum() == 0:
            continue
        src_labels = labels[mask]
        src_probs = probs[mask]
        src_tids = track_ids[mask]

        n_classes = len(np.unique(src_labels))
        if n_classes < 2:
            # Single class — AUROC undefined, report accuracy only
            preds = (src_probs >= threshold).astype(int)
            results[source] = {
                "chunk": {
                    "auroc": None,
                    "auprc": None,
                    "auprc_human": None,
                    "accuracy": float(accuracy_score(src_labels, preds)),
                    "f1": float(f1_score(src_labels, preds, zero_division=0)),
                    "f1_human": float(
                        f1_score(1 - src_labels, 1 - preds, zero_division=0)
                    ),
                    "n_chunks": int(mask.sum()),
                    "note": "single_class",
                },
                "track": {
                    "auroc": None,
                    "auprc": None,
                    "accuracy": None,
                    "n_tracks": len(np.unique(src_tids)),
                    "note": "single_class",
                },
            }
        else:
            results[source] = {
                "chunk": {
                    **compute_chunk_metrics(src_labels, src_probs, threshold),
                    "n_chunks": int(mask.sum()),
                },
                "track": compute_track_metrics(
                    src_labels, src_probs, src_tids, threshold
                ),
            }

    # ── Pairwise metrics: each AI source vs human (FMA) ───────────────────
    # Find human source(s)
    human_mask = labels == 0
    ai_sources = [
        s
        for s in np.unique(sources_arr)
        if (sources_arr == s).sum() > 0 and labels[sources_arr == s].mean() > 0.5
    ]

    if human_mask.sum() > 0 and len(ai_sources) > 0:
        for ai_src in ai_sources:
            ai_mask = sources_arr == ai_src
            pair_mask = human_mask | ai_mask

            if pair_mask.sum() == 0:
                continue

            pair_labels = labels[pair_mask]
            pair_probs = probs[pair_mask]
            pair_tids = track_ids[pair_mask]
            pair_key = f"human_vs_{ai_src}"

            if len(np.unique(pair_labels)) < 2:
                continue

            results[pair_key] = {
                "chunk": {
                    **compute_chunk_metrics(pair_labels, pair_probs, threshold),
                    "n_chunks": int(pair_mask.sum()),
                },
                "track": compute_track_metrics(
                    pair_labels, pair_probs, pair_tids, threshold
                ),
            }

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
    Full evaluation pass. Returns raw arrays + all metrics.
    """
    model.eval()
    all_labels, all_probs, all_track_ids, all_sources = [], [], [], []

    with torch.no_grad():
        for batch_inputs, batch_labels, batch_metas in loader:
            labels_np = batch_labels.numpy().tolist()
            batch_inputs = batch_inputs.to(device)
            logits = model(batch_inputs)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

            all_probs.extend(probs.tolist())
            all_labels.extend(labels_np)
            all_track_ids.extend([m["track_id"] for m in batch_metas])
            all_sources.extend([m["source_name"] for m in batch_metas])

    labels = np.array(all_labels, dtype=np.int64)
    probs_arr = np.array(all_probs, dtype=np.float32)
    track_ids = np.array(all_track_ids, dtype=np.int64)

    chunk_metrics = compute_chunk_metrics(labels, probs_arr, threshold)
    track_metrics = compute_track_metrics(labels, probs_arr, track_ids, threshold)
    source_metrics = compute_per_source_metrics(
        labels, probs_arr, all_sources, track_ids, threshold
    )

    results = {
        "labels": labels,
        "probs": probs_arr,
        "track_ids": track_ids,
        "sources": all_sources,
        "chunk": chunk_metrics,
        "track": track_metrics,
        "per_source": source_metrics,
        "split": split_name,
    }

    _log_results(results, split_name)
    return results


# ── Result row builder (for comparison table) ─────────────────────────────────


def build_result_row(
    run_name: str,
    results: dict,
    model_type: str,
    train_cfg,
    split_strategy: str,
    extra_tags: dict | None = None,
) -> dict:
    c = results["chunk"]
    t = results["track"]
    ps = results["per_source"]

    row = {
        "run_name": run_name,
        "split_strategy": split_strategy,
        "model_type": model_type,
        "learning_rate": train_cfg.learning_rate,
        "batch_size": train_cfg.batch_size,
        # Overall chunk-level
        "chunk_auroc": round(c["auroc"], 4),
        "chunk_auprc_ai": round(c["auprc"], 4),
        "chunk_auprc_hum": round(c["auprc_human"], 4),
        "chunk_acc": round(c["accuracy"], 4),
        "chunk_f1_ai": round(c["f1"], 4),
        "chunk_f1_hum": round(c["f1_human"], 4),
        # Overall track-level
        "track_auroc": round(t["auroc"], 4),
        "track_auprc_ai": round(t["auprc"], 4),
        "track_auprc_hum": round(t["auprc_human"], 4),
        "track_acc": round(t["accuracy"], 4),
        "track_f1_ai": round(t["f1"], 4),
        "track_f1_hum": round(t["f1_human"], 4),
        "n_tracks": t["n_tracks"],
    }

    # Per-source accuracy (single-class sources)
    for source in ["FMA", "Suno", "Udio"]:
        if source in ps:
            m = ps[source]
            row[f"{source.lower()}_chunk_acc"] = round(m["chunk"]["accuracy"], 4)
            row[f"{source.lower()}_n_tracks"] = m["track"]["n_tracks"]

    # Pairwise AUROC (meaningful cross-class comparison)
    for ai_src in ["Suno", "Udio"]:
        key = f"human_vs_{ai_src}"
        if key in ps:
            row[f"vs_{ai_src.lower()}_chunk_auroc"] = round(
                ps[key]["chunk"]["auroc"], 4
            )
            row[f"vs_{ai_src.lower()}_track_auroc"] = round(
                ps[key]["track"]["auroc"], 4
            )
        else:
            row[f"vs_{ai_src.lower()}_chunk_auroc"] = None
            row[f"vs_{ai_src.lower()}_track_auroc"] = None

    if extra_tags:
        row.update(extra_tags)

    return row


# ── Logging ───────────────────────────────────────────────────────────────────


def _log_results(results: dict, split_name: str) -> None:
    log.info(f"\n{'=' * 55}")
    log.info(f"Evaluation — split='{split_name}'")
    log.info(f"{'=' * 55}")
    c = results["chunk"]
    t = results["track"]
    log.info(f"\nChunk-level ({len(results['labels'])} chunks):")
    log.info(f"  AUROC        = {c['auroc']:.4f}")
    log.info(f"  AUPRC (AI)   = {c['auprc']:.4f}")
    log.info(f"  AUPRC (hum)  = {c['auprc_human']:.4f}")
    log.info(f"  Accuracy     = {c['accuracy']:.4f}")
    log.info(f"  F1 (AI)      = {c['f1']:.4f}")
    log.info(f"  F1 (human)   = {c['f1_human']:.4f}")
    log.info(f"\nTrack-level ({t['n_tracks']} tracks):")
    log.info(f"  AUROC        = {t['auroc']:.4f}")
    log.info(f"  AUPRC (AI)   = {t['auprc']:.4f}")
    log.info(f"  Accuracy     = {t['accuracy']:.4f}")

    log.info(f"\nPer-source breakdown:")
    for src, m in results["per_source"].items():
        cm = m["chunk"]
        tm = m["track"]
        auroc_str = (
            f"{cm['auroc']:.4f}" if cm["auroc"] is not None else "N/A (single class)"
        )
        log.info(
            f"  {src:20s}: chunk auroc={auroc_str} "
            f"acc={cm['accuracy']:.4f} | "
            f"track n={tm['n_tracks']}"
        )


# ── Saving ────────────────────────────────────────────────────────────────────


def save_evaluation_results(
    results: dict,
    output_dir: Path,
    run_name: str,
    split_name: str,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preds_df = pd.DataFrame(
        {
            "label": results["labels"],
            "prob_ai": results["probs"],
            "pred": (results["probs"] >= 0.5).astype(int),
            "track_id": results["track_ids"],
            "source_name": results["sources"],
        }
    )
    preds_path = output_dir / f"{run_name}_{split_name}_predictions.csv"
    preds_df.to_csv(preds_path, index=False)

    # Serializable metrics only
    metrics = {
        "chunk": results["chunk"],
        "track": results["track"],
        "per_source": {
            src: {"chunk": m["chunk"], "track": m["track"]}
            for src, m in results["per_source"].items()
        },
    }
    metrics_path = output_dir / f"{run_name}_{split_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    log.info(f"Predictions → {preds_path}")
    log.info(f"Metrics     → {metrics_path}")


# ── Training curve plot ───────────────────────────────────────────────────────


def plot_training_history(
    history_df: pd.DataFrame,
    output_path: Path | None = None,
    title: str = "Training History",
) -> None:
    import matplotlib.pyplot as plt

    train_h = history_df[history_df["split"] == "train"]
    val_h = history_df[history_df["split"] == "val"]

    metrics = [
        ("loss", "Loss"),
        ("chunk_auroc", "Chunk AUROC"),
        ("chunk_auprc", "Chunk AUPRC (AI)"),
        ("track_auroc", "Track AUROC"),
        ("track_auprc", "Track AUPRC (AI)"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    for ax, (col, label) in zip(axes, metrics):
        ax.plot(
            train_h["epoch"],
            train_h[col],
            color="#4C72B0",
            linewidth=1.5,
            label="train",
        )
        ax.plot(
            val_h["epoch"],
            val_h[col],
            color="#DD8452",
            linewidth=1.5,
            linestyle="--",
            label="val",
        )
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        log.info(f"Saved → {output_path}")
    plt.show()
