# =============================================================================
# train.py
# =============================================================================
# Training loop with:
#   - Mixed precision (autocast) where available
#   - LR scheduler (ReduceLROnPlateau on val AUPRC)
#   - Early stopping on val AUPRC
#   - Checkpoint saving (best model + last model)
#   - Per-epoch logging of loss, accuracy, AUROC, AUPRC
#     at both chunk level and track level
#   - Full training history returned as DataFrame for plotting
# =============================================================================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import time
import json

log = logging.getLogger(__name__)


def _compute_metrics(
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    all_track_ids: np.ndarray,
    all_sources: list[str],
    threshold: float = 0.5,
) -> dict:
    from evaluate import compute_chunk_metrics, compute_track_metrics

    chunk_metrics = compute_chunk_metrics(all_labels, all_probs, threshold)
    track_metrics = compute_track_metrics(
        all_labels, all_probs, all_track_ids, threshold
    )
    return {"chunk": chunk_metrics, "track": track_metrics}


def _run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    is_train,
    use_amp,
    scaler,
) -> tuple:
    model.train(is_train)
    total_loss = 0.0
    all_labels = []
    all_probs = []
    all_track_ids = []
    all_sources = []

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for batch_inputs, batch_labels, batch_metas in loader:
            # Collect BEFORE device move
            labels_np = batch_labels.numpy().tolist()
            track_ids_np = [m["track_id"] for m in batch_metas]
            sources_np = [m["source_name"] for m in batch_metas]

            # Zero gradients BEFORE forward
            if is_train:
                optimizer.zero_grad(set_to_none=True)

            batch_inputs = batch_inputs.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)

            if use_amp and scaler is not None:
                with torch.autocast(device_type=device.type):
                    logits = model(batch_inputs)
                    loss = criterion(logits, batch_labels)
            else:
                logits = model(batch_inputs)
                loss = criterion(logits, batch_labels)

            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            total_loss += loss.item() * batch_inputs.size(0)

            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels_np)
            all_track_ids.extend(track_ids_np)
            all_sources.extend(sources_np)

    avg_loss = total_loss / len(loader.dataset)
    return (
        avg_loss,
        np.array(all_labels, dtype=np.int64),
        np.array(all_probs, dtype=np.float32),
        np.array(all_track_ids, dtype=np.int64),
        all_sources,
    )


def _format_metrics(metrics: dict, prefix: str) -> str:
    c = metrics["chunk"]
    t = metrics["track"]
    return (
        f"{prefix} | "
        f"chunk — acc={c['accuracy']:.4f} auroc={c['auroc']:.4f} "
        f"auprc={c['auprc']:.4f} | "
        f"track — acc={t['accuracy']:.4f} auroc={t['auroc']:.4f} "
        f"auprc={t['auprc']:.4f}"
    )


def _epoch_row(epoch, split, loss, metrics) -> dict:
    c = metrics["chunk"]
    t = metrics["track"]
    return {
        "epoch": epoch,
        "split": split,
        "loss": loss,
        "chunk_accuracy": c["accuracy"],
        "chunk_auroc": c["auroc"],
        "chunk_auprc": c["auprc"],
        "chunk_f1": c["f1"],
        "track_accuracy": t["accuracy"],
        "track_auroc": t["auroc"],
        "track_auprc": t["auprc"],
        "track_f1": t["f1"],
    }


def _save_checkpoint(model, optimizer, epoch, metrics, path, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "val_auprc": metrics["chunk"]["auprc"],
            "config": {
                "model_type": config.MODEL_TYPE,
                "n_mels": config.N_MELS,
                "spec_shape": config.SPEC_SHAPE,
                "cnn_channels": config.CNN_CHANNELS,
            },
        },
        path,
    )
    log.info(f"Checkpoint saved → {path}")


def load_checkpoint(model, path, device, optimizer=None) -> dict:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])
    log.info(
        f"Loaded checkpoint from {path} "
        f"(epoch={ckpt['epoch']}, val_auprc={ckpt['val_auprc']:.4f})"
    )
    return ckpt


def train(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    config,
    device: torch.device,
    manifest: pd.DataFrame,
    run_name: str = "run",
) -> pd.DataFrame:
    """
    Full training loop.

    Parameters
    ----------
    model    : instantiated model on device
    loaders  : dict with keys "train", "val"
    config   : config module
    device   : torch device
    manifest : full manifest DataFrame — used to build augment-free
               train eval loader for honest metric computation
    run_name : checkpoint / log filename prefix
    """
    from dataset import build_eval_loader

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=config.LR_PATIENCE,
        factor=config.LR_FACTOR,
        min_lr=config.LR_MIN,
    )

    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Augment-free loader for honest train metrics
    train_eval_loader = build_eval_loader(
        manifest=manifest,
        config=config,
        split="train",
        model_type=config.MODEL_TYPE,
    )

    best_val_auprc = -1.0
    epochs_no_improve = 0
    history = []

    best_ckpt_path = config.CHECKPOINT_DIR / f"{run_name}_best.pt"
    last_ckpt_path = config.CHECKPOINT_DIR / f"{run_name}_last.pt"

    log.info(f"\n{'=' * 55}")
    log.info(f"Training start — run='{run_name}'")
    log.info(f"  Model     : {config.MODEL_TYPE}")
    log.info(f"  Device    : {device}")
    log.info(f"  Epochs    : {config.NUM_EPOCHS}")
    log.info(f"  Batch     : {config.BATCH_SIZE}")
    log.info(f"  LR        : {config.LEARNING_RATE}")
    log.info(f"  AMP       : {use_amp}")
    log.info(f"  Early stop: patience={config.EARLY_STOP_PATIENCE}")
    log.info(f"{'=' * 55}\n")

    for epoch in range(1, config.NUM_EPOCHS + 1):
        t0 = time.time()

        # ── Weight update pass (with augmentation) ─────────────────────────
        _run_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            is_train=True,
            use_amp=use_amp,
            scaler=scaler,
        )

        # ── Train metrics (no augmentation — honest evaluation) ────────────
        train_loss, tr_labels, tr_probs, tr_tids, tr_srcs = _run_epoch(
            model,
            train_eval_loader,
            criterion,
            None,
            device,
            is_train=False,
            use_amp=use_amp,
            scaler=None,
        )
        train_metrics = _compute_metrics(tr_labels, tr_probs, tr_tids, tr_srcs)

        # ── Val metrics ────────────────────────────────────────────────────
        val_loss, val_labels, val_probs, val_tids, val_srcs = _run_epoch(
            model,
            loaders["val"],
            criterion,
            None,
            device,
            is_train=False,
            use_amp=use_amp,
            scaler=None,
        )
        val_metrics = _compute_metrics(val_labels, val_probs, val_tids, val_srcs)

        val_auprc = val_metrics["chunk"]["auprc"]
        elapsed = time.time() - t0

        scheduler.step(val_auprc)
        current_lr = optimizer.param_groups[0]["lr"]

        log.info(
            f"Epoch {epoch:03d}/{config.NUM_EPOCHS} "
            f"[{elapsed:.1f}s | lr={current_lr:.2e}]"
        )
        log.info(_format_metrics(train_metrics, "  TRAIN"))
        log.info(_format_metrics(val_metrics, "  VAL  "))

        history.append(_epoch_row(epoch, "train", train_loss, train_metrics))
        history.append(_epoch_row(epoch, "val", val_loss, val_metrics))

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            epochs_no_improve = 0
            _save_checkpoint(
                model, optimizer, epoch, val_metrics, best_ckpt_path, config
            )
            log.info(f"  ↑ New best val AUPRC: {best_val_auprc:.4f}")
        else:
            epochs_no_improve += 1
            log.info(
                f"  No improvement for "
                f"{epochs_no_improve}/{config.EARLY_STOP_PATIENCE} "
                f"epochs (best={best_val_auprc:.4f})"
            )

        _save_checkpoint(model, optimizer, epoch, val_metrics, last_ckpt_path, config)

        if epochs_no_improve >= config.EARLY_STOP_PATIENCE:
            log.info(
                f"\nEarly stopping at epoch {epoch}. "
                f"Best val AUPRC: {best_val_auprc:.4f}"
            )
            break

    history_df = pd.DataFrame(history)
    history_path = config.LOGS_DIR / f"{run_name}_history.csv"
    history_df.to_csv(history_path, index=False)
    log.info(f"\nTraining history saved → {history_path}")
    log.info(f"\n{'=' * 55}")
    log.info(f"Training complete — best val AUPRC: {best_val_auprc:.4f}")
    log.info(f"Best checkpoint  → {best_ckpt_path}")
    log.info(f"{'=' * 55}")
    return history_df
