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
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field, asdict
import logging
import time
import platform

log = logging.getLogger(__name__)


# ── TrainingConfig ────────────────────────────────────────────────────────────


@dataclass
class TrainingConfig:
    """
    All training hyperparameters in one place.
    Instantiate in the notebook and pass to train().
    Stored inside every checkpoint for reproducibility.

    scheduler_metric  : "val_loss" | "val_auroc" | "val_auprc"
                        What the LR scheduler monitors.
                        val_loss is recommended — smoother signal.
    checkpoint_metric : "val_auroc" | "val_auprc"
                        What triggers best-model saving and early stopping.
                        val_auroc is recommended.
    """

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    num_epochs: int = 30

    # LR schedule
    lr_patience: int = 5
    lr_factor: float = 0.5
    lr_min: float = 1e-6
    warmup_epochs: int = 0

    # Early stopping and checkpointing
    early_stop_patience: int = 10
    scheduler_metric: str = "val_loss"  # monitor loss for LR reduction
    checkpoint_metric: str = "val_auroc"  # save best AUROC model

    # Augmentation (passed through to dataset if needed)
    use_augmentation: bool = True
    freq_mask_param: int = 15
    time_mask_param: int = 50
    n_freq_masks: int = 2
    n_time_masks: int = 2

    def to_dict(self) -> dict:
        return asdict(self)

    def log_summary(self) -> None:
        log.info("Training config:")
        for k, v in self.to_dict().items():
            log.info(f"  {k:<25}: {v}")


# ── Epoch runner ──────────────────────────────────────────────────────────────


def _run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    is_train: bool,
    use_amp: bool,
    scaler,
) -> tuple:
    """
    Run one epoch — either training or evaluation.
    Returns (avg_loss, labels, probs, track_ids, sources).
    """
    model.train(is_train)
    total_loss = 0.0
    all_labels = []
    all_probs = []
    all_track_ids = []
    all_sources = []

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for b_in, b_lab, b_meta in loader:
            # Collect labels BEFORE device move (MPS safety)
            labels_np = b_lab.numpy().tolist()
            tids_np = [m["track_id"] for m in b_meta]
            srcs_np = [m["source_name"] for m in b_meta]

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            # NOTE: no non_blocking=True — causes silent data corruption on MPS
            b_in = b_in.to(device)
            b_lab = b_lab.to(device)

            if use_amp and scaler is not None:
                with torch.autocast(device_type=device.type):
                    logits = model(b_in)
                    loss = criterion(logits, b_lab)
            else:
                logits = model(b_in)
                loss = criterion(logits, b_lab)

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

            total_loss += loss.item() * b_in.size(0)

            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

            all_probs.extend(probs.tolist())
            all_labels.extend(labels_np)
            all_track_ids.extend(tids_np)
            all_sources.extend(srcs_np)

    avg_loss = total_loss / len(loader.dataset)
    return (
        avg_loss,
        np.array(all_labels, dtype=np.int64),
        np.array(all_probs, dtype=np.float32),
        np.array(all_track_ids, dtype=np.int64),
        all_sources,
    )


# ── Metric helpers ────────────────────────────────────────────────────────────


def _compute_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
    track_ids: np.ndarray,
    sources: list,
    threshold: float = 0.5,
) -> dict:
    from evaluate import compute_chunk_metrics, compute_track_metrics

    return {
        "chunk": compute_chunk_metrics(labels, probs, threshold),
        "track": compute_track_metrics(labels, probs, track_ids, threshold),
    }


def _format_metrics(metrics: dict, prefix: str) -> str:
    c = metrics["chunk"]
    t = metrics["track"]
    return (
        f"{prefix} | "
        f"chunk acc={c['accuracy']:.4f} "
        f"auroc={c['auroc']:.4f} "
        f"auprc={c['auprc']:.4f} | "
        f"track auroc={t['auroc']:.4f} "
        f"auprc={t['auprc']:.4f}"
    )


def _epoch_row(epoch, split, loss, metrics) -> dict:
    c = metrics["chunk"]
    t = metrics["track"]
    return {
        "epoch": epoch,
        "split": split,
        "loss": round(loss, 6),
        "chunk_accuracy": c["accuracy"],
        "chunk_auroc": c["auroc"],
        "chunk_auprc": c["auprc"],
        "chunk_f1": c["f1"],
        "track_accuracy": t["accuracy"],
        "track_auroc": t["auroc"],
        "track_auprc": t["auprc"],
        "track_f1": t["f1"],
    }


# ── Checkpoint helpers ────────────────────────────────────────────────────────


def _save_checkpoint(
    model,
    optimizer,
    epoch,
    val_loss,
    val_metrics,
    path,
    run_name,
    train_cfg,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "run_name": run_name,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_auroc": val_metrics["chunk"]["auroc"],
            "val_auprc": val_metrics["chunk"]["auprc"],
            "train_config": train_cfg.to_dict(),
        },
        path,
    )
    log.info(f"Checkpoint → {path}")


def load_checkpoint(
    model,
    path,
    device,
    optimizer=None,
) -> dict:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])
    log.info(
        f"Loaded {path} | "
        f"epoch={ckpt['epoch']} | "
        f"val_auroc={ckpt.get('val_auroc', 'n/a'):.4f}"
    )
    return ckpt


# ── Main training function ────────────────────────────────────────────────────


def train(
    model: nn.Module,
    loaders: dict,
    train_cfg: TrainingConfig,
    base_config,
    device: torch.device,
    manifest: pd.DataFrame,
    run_name: str = "run",
    model_type: str = "cnn",
) -> pd.DataFrame:
    """
    Full training loop.

    Parameters
    ----------
    model       : instantiated model already on device
    loaders     : dict with keys "train" and "val"
                  Built externally in the notebook.
                  loaders["train"] must have augmentation=True.
                  loaders["val"]   must have augmentation=False.
    train_cfg   : TrainingConfig dataclass
    base_config : config module — paths, mel params
    device      : torch device
    manifest    : full manifest DataFrame — used to build augment-free
                  train eval loader for honest metric computation
    run_name    : checkpoint and log filename prefix
    model_type  : "mlp" | "cnn" | "cnn_lstm" | "cnn_gru"

    Returns
    -------
    history_df : DataFrame — one row per (epoch, split)
    """
    from dataset import MelSpecDataset, MelSpecSeqDataset, _collate_fn

    # ── Augment-free train eval loader ────────────────────────────────────
    # Built here — not passed in — so metrics are always on clean data.
    mtype = model_type.lower()
    if mtype in (
        "cnn_lstm",
        "cnn_gru",  # legacy alias for the original BiGRU model
        "cnn_bigru",
        "cnn_bilstm",
        "cnn_bilstm_attention",
    ):
        train_eval_ds = MelSpecSeqDataset(
            manifest=manifest,
            split="train",
            config=base_config,
            segment_frames=43,
            augment=False,
        )
    else:
        # covers "mlp", "cnn", and anything else using 2D spectrograms
        train_eval_ds = MelSpecDataset(
            manifest=manifest,
            split="train",
            config=base_config,
            augment=False,
        )

    train_eval_loader = DataLoader(
        train_eval_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=0 if platform.system() == "Darwin" else 4,
        pin_memory=False,
        drop_last=False,
        collate_fn=_collate_fn,
    )

    # ── Optimizer ─────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    # ── Warmup scheduler ──────────────────────────────────────────────────
    use_warmup = train_cfg.warmup_epochs > 0
    if use_warmup:
        warmup = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=train_cfg.warmup_epochs,
        )

    # ── Plateau scheduler — mode depends on monitored metric ──────────────
    plateau_mode = "min" if train_cfg.scheduler_metric == "val_loss" else "max"
    plateau = ReduceLROnPlateau(
        optimizer,
        mode=plateau_mode,
        patience=train_cfg.lr_patience,
        factor=train_cfg.lr_factor,
        min_lr=train_cfg.lr_min,
    )

    # ── AMP (CUDA only — MPS not supported) ───────────────────────────────
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # ── State ─────────────────────────────────────────────────────────────
    best_metric_value = (
        float("inf") if train_cfg.checkpoint_metric == "val_loss" else -1.0
    )
    epochs_no_improve = 0
    history = []

    best_ckpt = base_config.CHECKPOINT_DIR / f"{run_name}_best.pt"
    last_ckpt = base_config.CHECKPOINT_DIR / f"{run_name}_last.pt"

    log.info(f"\n{'=' * 60}")
    log.info(f"Training  run='{run_name}'  device={device}  model={mtype}")
    log.info(f"  scheduler → {train_cfg.scheduler_metric} (mode={plateau_mode})")
    log.info(f"  checkpoint → best {train_cfg.checkpoint_metric}")
    train_cfg.log_summary()
    log.info(f"{'=' * 60}\n")

    for epoch in range(1, train_cfg.num_epochs + 1):
        t0 = time.time()

        # ── Weight update pass (augmented train loader) ────────────────────
        train_dataset = loaders["train"].dataset
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch - 1)
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

        # ── Train metrics (augment-free) ───────────────────────────────────
        train_loss, tr_l, tr_p, tr_t, tr_s = _run_epoch(
            model,
            train_eval_loader,
            criterion,
            None,
            device,
            is_train=False,
            use_amp=use_amp,
            scaler=None,
        )
        train_metrics = _compute_metrics(tr_l, tr_p, tr_t, tr_s)

        # ── Val metrics ────────────────────────────────────────────────────
        val_loss, vl_l, vl_p, vl_t, vl_s = _run_epoch(
            model,
            loaders["val"],
            criterion,
            None,
            device,
            is_train=False,
            use_amp=use_amp,
            scaler=None,
        )
        val_metrics = _compute_metrics(vl_l, vl_p, vl_t, vl_s)

        val_auroc = val_metrics["chunk"]["auroc"]
        val_auprc = val_metrics["chunk"]["auprc"]
        elapsed = time.time() - t0

        # ── Scheduler step ─────────────────────────────────────────────────
        if use_warmup and epoch <= train_cfg.warmup_epochs:
            warmup.step()
        else:
            sched_value = {
                "val_loss": val_loss,
                "val_auroc": val_auroc,
                "val_auprc": val_auprc,
            }[train_cfg.scheduler_metric]
            plateau.step(sched_value)

        current_lr = optimizer.param_groups[0]["lr"]

        # ── Logging ────────────────────────────────────────────────────────
        log.info(
            f"Epoch {epoch:03d}/{train_cfg.num_epochs} "
            f"[{elapsed:.1f}s | lr={current_lr:.2e}] | "
            f"val_loss={val_loss:.4f} "
            f"val_auroc={val_auroc:.4f} "
            f"val_auprc={val_auprc:.4f}"
        )
        log.info(_format_metrics(train_metrics, "  TRAIN"))
        log.info(_format_metrics(val_metrics, "  VAL  "))

        history.append(_epoch_row(epoch, "train", train_loss, train_metrics))
        history.append(_epoch_row(epoch, "val", val_loss, val_metrics))

        # ── Checkpoint / early stopping ────────────────────────────────────
        ckpt_value = {
            "val_loss": val_loss,
            "val_auroc": val_auroc,
            "val_auprc": val_auprc,
        }[train_cfg.checkpoint_metric]

        # For loss: lower is better. For AUROC/AUPRC: higher is better.
        improved = (
            ckpt_value < best_metric_value
            if train_cfg.checkpoint_metric == "val_loss"
            else ckpt_value > best_metric_value
        )

        if improved:
            best_metric_value = ckpt_value
            epochs_no_improve = 0
            _save_checkpoint(
                model,
                optimizer,
                epoch,
                val_loss,
                val_metrics,
                best_ckpt,
                run_name,
                train_cfg,
            )
            log.info(
                f"  ↑ New best {train_cfg.checkpoint_metric}="
                f"{best_metric_value:.4f} "
                f"(auroc={val_auroc:.4f} auprc={val_auprc:.4f})"
            )
        else:
            epochs_no_improve += 1
            log.info(
                f"  No improvement "
                f"{epochs_no_improve}/{train_cfg.early_stop_patience} "
                f"(best {train_cfg.checkpoint_metric}="
                f"{best_metric_value:.4f})"
            )

        _save_checkpoint(
            model,
            optimizer,
            epoch,
            val_loss,
            val_metrics,
            last_ckpt,
            run_name,
            train_cfg,
        )

        if epochs_no_improve >= train_cfg.early_stop_patience:
            log.info(
                f"\nEarly stopping at epoch {epoch}. "
                f"Best {train_cfg.checkpoint_metric}: "
                f"{best_metric_value:.4f}"
            )
            break

    # ── Save history ───────────────────────────────────────────────────────
    history_df = pd.DataFrame(history)
    history_path = base_config.LOGS_DIR / f"{run_name}_history.csv"
    history_df.to_csv(history_path, index=False)

    log.info(f"\nHistory  → {history_path}")
    log.info(
        f"Best {train_cfg.checkpoint_metric}: {best_metric_value:.4f} → {best_ckpt}"
    )

    return history_df
