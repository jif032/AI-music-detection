"""Prepare data and run the CNN-BiGRU/BiLSTM/attention comparison on CPU.

Run from the repository root with:
    .venv/bin/python src/run_comparison.py --strategy random

The script uses existing prepared artifacts when available. It creates model-
specific checkpoint, log, and result directories and refuses to replace prior
run outputs unless --overwrite is supplied.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import config
from dataset import build_dataloaders
from evaluate import evaluate, save_evaluation_results
from mert_mining import build_mined_chunk_index
from models import build_model
from sampling import build_chunk_index, cap_ai_chunks
from spectrogram import generate_spectrograms, load_manifest
from splits import make_cross_source_split, make_random_split
from train import TrainingConfig, load_checkpoint, train

log = logging.getLogger("model_comparison")

MODEL_TYPES = ("cnn_bigru", "cnn_bilstm", "cnn_bilstm_attention")
MODEL_KWARGS = {
    "n_mels": config.N_MELS,
    "hidden_size": 128,
    "num_layers": 2,
    "segment_frames": 43,
    "dropout": 0.3,
}
TEST_THRESHOLD = 0.5


def make_training_config() -> TrainingConfig:
    """Keep the existing CNN-BiGRU experiment settings unchanged."""
    return TrainingConfig(
        learning_rate=1e-4,
        weight_decay=1e-4,
        batch_size=64,
        num_epochs=50,
        lr_patience=8,
        lr_factor=0.5,
        lr_min=1e-6,
        warmup_epochs=3,
        early_stop_patience=15,
        scheduler_metric="val_loss",
        checkpoint_metric="val_auroc",
        use_augmentation=True,
        freq_mask_param=15,
        time_mask_param=30,
        n_freq_masks=2,
        n_time_masks=2,
    )


def seed_experiment(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_or_create_split(strategy: str) -> pd.DataFrame:
    split_path = config.SPLITS_DIR / f"{strategy}_split.csv"
    if split_path.exists():
        log.info("Using existing split: %s", split_path)
        split_df = pd.read_csv(split_path)
    else:
        meta = pd.read_csv(config.META_PATH)
        if strategy == "random":
            split_df = make_random_split(
                meta=meta,
                train_ratio=config.TRAIN_RATIO,
                val_ratio=config.VAL_RATIO,
                test_ratio=config.TEST_RATIO,
                seed=config.SEED,
                output_path=split_path,
            )
        elif strategy in ("udio_train", "suno_train"):
            train_source = "Udio" if strategy == "udio_train" else "Suno"
            test_source = "Suno" if strategy == "udio_train" else "Udio"
            split_df = make_cross_source_split(
                meta=meta,
                train_source=train_source,
                test_source=test_source,
                val_ratio_of_train=config.VAL_RATIO,
                train_ratio=config.TRAIN_RATIO,
                seed=config.SEED,
                output_path=split_path,
            )
        else:
            raise ValueError(f"Unknown split strategy: {strategy}")

    required = {"file_path", "label", "source_name", "duration_sec", "split"}
    missing = required - set(split_df.columns)
    if missing:
        raise ValueError(f"Split is missing required columns: {sorted(missing)}")
    if split_df["file_path"].duplicated().any():
        raise ValueError("A track appears more than once in the track-level split.")
    return split_df


def _load_or_create_chunks(strategy: str, split_df: pd.DataFrame) -> pd.DataFrame:
    balanced_path = config.CHUNKS_DIR / f"{strategy}_chunks_balanced.csv"
    if balanced_path.exists():
        log.info("Using existing balanced chunks: %s", balanced_path)
        chunks = pd.read_csv(balanced_path)
    else:
        raw_path = config.CHUNKS_DIR / f"{strategy}_chunks_raw.csv"
        if raw_path.exists():
            raw = pd.read_csv(raw_path)
        else:
            raw = build_chunk_index(split_df, config, output_path=raw_path)
        chunks = cap_ai_chunks(raw, config, split="train")
        chunks.to_csv(balanced_path, index=False)
        log.info("Saved balanced chunks: %s", balanced_path)
    if chunks["chunk_id"].duplicated().any():
        raise ValueError("Chunk index contains duplicate chunk_id values.")
    return chunks


def _load_or_create_manifest(strategy: str, chunks: pd.DataFrame) -> pd.DataFrame:
    spec_dir = config.SPECTROGRAMS_DIR / strategy
    manifest_path = spec_dir / "manifest.csv"
    stats_path = spec_dir / "norm_stats.json"
    if manifest_path.exists() and stats_path.exists():
        manifest, _, _ = load_manifest(spec_dir)
        return manifest
    if spec_dir.exists() and any(spec_dir.rglob("*.npy")):
        raise FileExistsError(
            f"Found partial spectrogram files under {spec_dir}, without a complete "
            "manifest and normalization stats. Refusing to overwrite them."
        )
    manifest = generate_spectrograms(chunks, config, output_dir=spec_dir)
    if len(manifest) != len(chunks):
        raise RuntimeError(
            f"Spectrogram generation produced {len(manifest)} of {len(chunks)} chunks; "
            "stopping before mining or training."
        )
    return manifest


def _load_or_create_mined_index(strategy: str, chunks: pd.DataFrame) -> pd.DataFrame:
    mined_path = config.CHUNKS_DIR / f"{strategy}_chunks_mined.csv"
    if mined_path.exists():
        log.info("Using existing mined index: %s", mined_path)
        mined = pd.read_csv(mined_path)
    else:
        embeddings_path = config.CHUNKS_DIR / f"{strategy}_mert_embeddings_train.npy"
        mined = build_mined_chunk_index(
            chunk_df=chunks,
            config=config,
            device=torch.device("cpu"),
            subset_fraction=0.25,
            hard_negative_fraction=0.30,
            neighbor_confusion_threshold=0.30,
            knn_k=10,
            embeddings_cache_path=embeddings_path,
            random_seed=config.SEED,
        )
        mined.to_csv(mined_path, index=False)
        log.info("Saved mined index: %s", mined_path)
    return mined


def prepare_manifest(strategy: str) -> pd.DataFrame:
    """Create or reuse one strategy's split/chunk/spec/MERT artifacts."""
    split_df = _load_or_create_split(strategy)
    chunks = _load_or_create_chunks(strategy, split_df)
    full_manifest = _load_or_create_manifest(strategy, chunks)
    mined = _load_or_create_mined_index(strategy, chunks)

    train_ids = set(mined.loc[mined["split"] == "train", "chunk_id"].astype(int))
    train_manifest = full_manifest[
        (full_manifest["split"] == "train")
        & full_manifest["chunk_id"].isin(train_ids)
    ].copy()
    val_test = full_manifest[full_manifest["split"].isin(["val", "test"])].copy()
    mined_meta = mined.loc[mined["split"] == "train", [
        "chunk_id", "hard_type", "confusion_frac", "is_hard_negative"
    ]]
    manifest = pd.concat([train_manifest, val_test], ignore_index=True)
    manifest = manifest.merge(mined_meta, on="chunk_id", how="left")
    manifest["hard_type"] = manifest["hard_type"].fillna("random")
    manifest["confusion_frac"] = manifest["confusion_frac"].fillna(0.0)
    manifest["is_hard_negative"] = manifest["is_hard_negative"].fillna(False)

    if manifest["chunk_id"].duplicated().any():
        raise ValueError("Prepared manifest contains duplicate chunk IDs.")
    if set(manifest["split"].unique()) != {"train", "val", "test"}:
        raise ValueError("Prepared manifest must contain train, val, and test splits.")
    track_splits = manifest.groupby("file_path")["split"].nunique()
    if (track_splits > 1).any():
        raise ValueError("Track leakage detected across train/validation/test.")
    missing_specs = [p for p in manifest["npy_path"] if not Path(p).is_file()]
    if missing_specs:
        raise FileNotFoundError(f"{len(missing_specs)} spectrogram files are missing.")

    log.info("Prepared comparison manifest: %s", manifest.groupby("split").size().to_dict())
    return manifest


def _model_run_config(model_type: str, strategy: str):
    checkpoint_dir = config.CHECKPOINTS_DIR / model_type / strategy
    log_dir = config.LOGS_DIR / model_type / strategy
    result_dir = config.RESULTS_DIR / model_type / strategy
    for directory in (checkpoint_dir, log_dir, result_dir):
        directory.mkdir(parents=True, exist_ok=True)
    values = {key: value for key, value in vars(config).items() if key.isupper()}
    values["CHECKPOINT_DIR"] = checkpoint_dir
    values["CHECKPOINTS_DIR"] = checkpoint_dir
    values["LOGS_DIR"] = log_dir
    values["RESULTS_DIR"] = result_dir
    values["AUGMENTATION_SEED"] = config.SEED
    return SimpleNamespace(**values), checkpoint_dir, log_dir, result_dir


def _assert_run_outputs_available(model_type: str, strategy: str, overwrite: bool):
    run_name = f"{model_type}_mert_{strategy}"
    run_config, checkpoint_dir, log_dir, result_dir = _model_run_config(model_type, strategy)
    expected = [
        checkpoint_dir / f"{run_name}_best.pt",
        checkpoint_dir / f"{run_name}_last.pt",
        log_dir / f"{run_name}_history.csv",
        result_dir / "summary.json",
    ]
    existing = [path for path in expected if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Existing model outputs would be replaced; rerun with --overwrite only "
            f"if that is intended. Existing: {existing}"
        )
    return run_name, run_config, checkpoint_dir, log_dir, result_dir


def validate_sampling_sequences(manifest: pd.DataFrame, strategy: str) -> None:
    """Verify same per-epoch weighted draws and augmentation across model types."""
    probe_config = SimpleNamespace(**{
        key: value for key, value in vars(config).items() if key.isupper()
    })
    probe_config.AUGMENTATION_SEED = config.SEED
    sampler_sequences = {}
    train_datasets = {}
    for model_type in MODEL_TYPES:
        loaders = build_dataloaders(
            manifest,
            probe_config,
            model_type=model_type,
            batch_size=64,
            sampler_seed=config.SEED,
        )
        sampler = loaders["train"].sampler
        if sampler is None:
            raise RuntimeError("Expected the configured weighted sampler.")
        sampler_sequences[model_type] = [
            list(iter(sampler)) for _ in range(3)
        ]
        train_datasets[model_type] = loaders["train"].dataset

    baseline = sampler_sequences[MODEL_TYPES[0]]
    for model_type in MODEL_TYPES[1:]:
        if sampler_sequences[model_type] != baseline:
            raise RuntimeError(f"Sampler sequences differ for {model_type}.")
    for epoch in range(3):
        for model_type in MODEL_TYPES:
            train_datasets[model_type].set_epoch(epoch)
        sample_idx = baseline[epoch][0]
        tensors = [train_datasets[m][sample_idx][0] for m in MODEL_TYPES]
        if not all(torch.equal(tensors[0], tensor) for tensor in tensors[1:]):
            raise RuntimeError("Seeded SpecAugment differs across model variants.")
    log.info("Identical sampler sequences and augmentation verified for 3 epochs.")


def measure_inference(loader, model, device: torch.device) -> tuple[float, int]:
    """Measure end-to-end CPU loader+model time after one warmup batch."""
    model.eval()
    batches = iter(loader)
    warmup = next(batches, None)
    if warmup is None:
        return float("nan"), 0
    with torch.inference_mode():
        model(warmup[0].to(device))

    chunks = 0
    start = time.perf_counter()
    with torch.inference_mode():
        for inputs, _, _ in batches:
            model(inputs.to(device))
            chunks += len(inputs)
    seconds = time.perf_counter() - start
    return seconds, chunks


def run_comparison(strategy: str = "random", overwrite: bool = False) -> pd.DataFrame:
    """Prepare artifacts and train/evaluate all three models sequentially on CPU."""
    device = torch.device("cpu")
    train_cfg = make_training_config()
    train_cfg.log_summary()

    # Check output collisions before expensive preprocessing or model downloads.
    run_dirs = {
        model_type: _assert_run_outputs_available(model_type, strategy, overwrite)
        for model_type in MODEL_TYPES
    }
    manifest = prepare_manifest(strategy)
    validate_sampling_sequences(manifest, strategy)

    aggregate_dir = config.RESULTS_DIR / "architecture_comparison"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = aggregate_dir / f"{strategy}.csv"
    if aggregate_path.exists() and not overwrite:
        raise FileExistsError(
            f"Aggregate results already exist at {aggregate_path}; use --overwrite "
            "to replace them."
        )
    rows = []

    for model_type in MODEL_TYPES:
        run_name, run_config, checkpoint_dir, log_dir, result_dir = run_dirs[model_type]
        seed_experiment(config.SEED)
        loader_config = SimpleNamespace(**{
            key: value for key, value in vars(config).items() if key.isupper()
        })
        loader_config.AUGMENTATION_SEED = config.SEED
        loaders = build_dataloaders(
            manifest,
            loader_config,
            model_type=model_type,
            batch_size=train_cfg.batch_size,
            sampler_seed=config.SEED,
        )
        model = build_model(model_type, **MODEL_KWARGS).to(device)
        parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info("Starting %s on %s with %s trainable parameters", model_type, device, parameter_count)

        start = time.perf_counter()
        history = train(
            model=model,
            loaders=loaders,
            train_cfg=train_cfg,
            base_config=run_config,
            device=device,
            manifest=manifest,
            run_name=run_name,
            model_type=model_type,
        )
        training_seconds = time.perf_counter() - start
        best_checkpoint = load_checkpoint(
            model,
            checkpoint_dir / f"{run_name}_best.pt",
            device,
        )
        test_results = evaluate(
            model=model,
            loader=loaders["test"],
            device=device,
            split_name="test",
            threshold=TEST_THRESHOLD,
        )
        inference_seconds, inference_chunks = measure_inference(
            loaders["test"], model, device
        )
        save_evaluation_results(test_results, log_dir, run_name, "test")

        summary = {
            "model_type": model_type,
            "run_name": run_name,
            "device": "cpu",
            "seed": config.SEED,
            "trainable_parameters": int(parameter_count),
            "training_seconds": float(training_seconds),
            "inference_seconds_after_warmup": float(inference_seconds),
            "inference_chunks_after_warmup": int(inference_chunks),
            "inference_chunks_per_second": (
                float(inference_chunks / inference_seconds)
                if inference_seconds > 0
                else None
            ),
            "epochs_completed": int(history["epoch"].max()),
            "best_epoch": int(best_checkpoint["epoch"]),
            "best_validation_auroc": float(best_checkpoint["val_auroc"]),
            "test_threshold": TEST_THRESHOLD,
            "chunk_metrics": test_results["chunk"],
            "track_metrics": test_results["track"],
            "per_source_metrics": test_results["per_source"],
        }
        with (result_dir / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
        rows.append({
            "model_type": model_type,
            "run_name": run_name,
            "seed": config.SEED,
            "device": "cpu",
            "trainable_parameters": parameter_count,
            "training_seconds": training_seconds,
            "inference_seconds_after_warmup": inference_seconds,
            "inference_chunks_after_warmup": inference_chunks,
            "inference_chunks_per_second": (
                inference_chunks / inference_seconds if inference_seconds > 0 else None
            ),
            "epochs_completed": int(history["epoch"].max()),
            "best_epoch": int(best_checkpoint["epoch"]),
            "best_validation_auroc": best_checkpoint["val_auroc"],
            "test_threshold": TEST_THRESHOLD,
            "chunk_auroc": test_results["chunk"]["auroc"],
            "chunk_auprc_ai": test_results["chunk"]["auprc"],
            "chunk_accuracy": test_results["chunk"]["accuracy"],
            "chunk_f1_ai": test_results["chunk"]["f1"],
            "track_auroc": test_results["track"]["auroc"],
            "track_auprc_ai": test_results["track"]["auprc"],
            "track_accuracy": test_results["track"]["accuracy"],
            "track_f1_ai": test_results["track"]["f1"],
            "n_chunks": len(test_results["labels"]),
            "n_tracks": test_results["track"]["n_tracks"],
        })
        pd.DataFrame(rows).to_csv(aggregate_path, index=False)
        log.info(
            "Finished %s: %.2f min, chunk AUROC=%.4f, track AUROC=%.4f",
            model_type,
            training_seconds / 60,
            test_results["chunk"]["auroc"],
            test_results["track"]["auroc"],
        )
        del model, loaders

    result_df = pd.DataFrame(rows)
    log.info("All model runs complete. Comparison results: %s", aggregate_path)
    return result_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy",
        choices=("random", "udio_train", "suno_train"),
        default="random",
        help="Existing project split strategy (default: random).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of this strategy's model and comparison outputs.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare/reuse data and validate split/sampling before any model training.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if args.prepare_only:
        manifest = prepare_manifest(args.strategy)
        validate_sampling_sequences(manifest, args.strategy)
        print("Prepared manifest rows:", manifest.groupby("split").size().to_dict())
        print("No model training was run.")
    else:
        result = run_comparison(strategy=args.strategy, overwrite=args.overwrite)
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
