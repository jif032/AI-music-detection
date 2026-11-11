# =============================================================================
# dataset.py
# =============================================================================
# PyTorch Dataset classes for all model types.
#
# Three dataset variants sharing a common base:
#   MelSpecDataset      — for CNN (returns 2D spectrogram as image)
#   MelSpecSeqDataset   — for CNN+LSTM / CNN+GRU (returns sequence of frames)
#   ASTDataset          — for AST fine-tuning (returns raw audio at 16kHz)
#
# All variants:
#   - Load pre-generated .npy spectrograms (or raw audio for AST)
#   - Apply SpecAugment during training if configured
#   - Return (input_tensor, label, meta_dict) so track-level evaluation
#     can group chunks by track_id without a separate lookup
# =============================================================================

# =============================================================================
# dataset.py
# =============================================================================
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import librosa
from pathlib import Path
import logging
import platform

log = logging.getLogger(__name__)


# ── SpecAugment ───────────────────────────────────────────────────────────────


def spec_augment(
    spec: torch.Tensor,
    freq_mask_param: int,
    time_mask_param: int,
    n_freq_masks: int,
    n_time_masks: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    n_mels, n_frames = spec.shape
    for _ in range(n_freq_masks):
        f = torch.randint(0, freq_mask_param + 1, (1,), generator=generator).item()
        f_start = torch.randint(
            0, max(1, n_mels - f), (1,), generator=generator
        ).item()
        spec[f_start : f_start + f, :] = 0.0
    for _ in range(n_time_masks):
        t = torch.randint(0, time_mask_param + 1, (1,), generator=generator).item()
        t_start = torch.randint(
            0, max(1, n_frames - t), (1,), generator=generator
        ).item()
        spec[:, t_start : t_start + t] = 0.0
    return spec


# ── Config attribute helpers ──────────────────────────────────────────────────


def _get(config, attr: str, default):
    """Safely read an attribute from config with a fallback default."""
    return getattr(config, attr, default)


# ── Base Dataset ──────────────────────────────────────────────────────────────


class _BaseDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        split: str,
        config,
        augment: bool | None = None,
    ):
        self.config = config
        self.split = split
        self.is_train = split == "train"
        self.augment = self.is_train if augment is None else augment
        self.augmentation_seed = _get(config, "AUGMENTATION_SEED", None)
        self.epoch = 0

        self.df = manifest[manifest["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No chunks found for split='{split}'")

        # Remap track_id to safe contiguous integers
        unique_tracks = self.df["file_path"].unique()
        track_id_map = {fp: i for i, fp in enumerate(unique_tracks)}
        self.df = self.df.copy()
        self.df["track_id"] = self.df["file_path"].map(track_id_map)

        # Binary labels
        self.labels = (self.df["label"].astype(str) == "ai").astype(int).values

        unique_labels = np.unique(self.labels)
        assert set(unique_labels).issubset({0, 1}), (
            f"Labels must be binary (0/1), got: {unique_labels}"
        )

        log.info(
            f"[{self.__class__.__name__}] split={split} | "
            f"augment={self.augment} | "
            f"{len(self.df)} chunks | "
            f"human={(self.labels == 0).sum()} | "
            f"ai={(self.labels == 1).sum()}"
        )

    def __len__(self) -> int:
        return len(self.df)

    def _get_meta(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        return {
            "chunk_id": int(row["chunk_id"]),
            "track_id": int(row["track_id"]),
            "file_path": str(row["file_path"]),
            "source_name": str(row["source_name"]),
            "label": int(self.labels[idx]),
        }

    def _load_spec(self, idx: int) -> np.ndarray:
        return np.load(self.df.iloc[idx]["npy_path"]).astype(np.float32)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used by seeded per-sample SpecAugment."""
        self.epoch = int(epoch)

    def _maybe_augment(self, spec: torch.Tensor, idx: int) -> torch.Tensor:
        """Apply SpecAugment only when self.augment is True AND config has params."""
        if not self.augment:
            return spec
        # Read augmentation params with safe defaults (off if not present)
        use_aug = _get(self.config, "USE_AUGMENTATION", False)
        if not use_aug:
            return spec
        generator = None
        if self.augmentation_seed is not None:
            seed = (int(self.augmentation_seed) + self.epoch * 1_000_003 + idx) % (
                2**63 - 1
            )
            generator = torch.Generator(device="cpu").manual_seed(seed)
        return spec_augment(
            spec=spec,
            freq_mask_param=_get(self.config, "FREQ_MASK_PARAM", 15),
            time_mask_param=_get(self.config, "TIME_MASK_PARAM", 50),
            n_freq_masks=_get(self.config, "N_FREQ_MASKS", 2),
            n_time_masks=_get(self.config, "N_TIME_MASKS", 2),
            generator=generator,
        )


# ── CNN / MLP Dataset ─────────────────────────────────────────────────────────


class MelSpecDataset(_BaseDataset):
    """
    For CNN and MLP models.
    Returns (1, n_mels, time_frames) — MLP flattens internally.
    """

    def __init__(self, manifest, split, config, augment=None):
        super().__init__(manifest, split, config, augment=augment)

    def __getitem__(self, idx: int):
        spec = torch.from_numpy(self._load_spec(idx))
        spec = self._maybe_augment(spec, idx)
        spec = spec.unsqueeze(0)
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return spec, label, self._get_meta(idx)


# ── CNN + RNN Dataset ─────────────────────────────────────────────────────────


class MelSpecSeqDataset(_BaseDataset):
    """
    For CNN+LSTM / CNN+GRU.
    Returns (n_segments, 1, n_mels, segment_frames).
    """

    def __init__(self, manifest, split, config, segment_frames=43, augment=None):
        super().__init__(manifest, split, config, augment=augment)
        self.segment_frames = segment_frames
        self.n_segments = config.SPEC_TIME_FRAMES // segment_frames
        log.info(f"  Sequence: {self.n_segments} segments × {segment_frames} frames")

    def __getitem__(self, idx: int):
        spec = torch.from_numpy(self._load_spec(idx))
        spec = self._maybe_augment(spec, idx)

        usable = self.n_segments * self.segment_frames
        spec = (
            spec[:, :usable]
            .reshape(self.config.N_MELS, self.n_segments, self.segment_frames)
            .permute(1, 0, 2)
            .unsqueeze(1)
        )

        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return spec, label, self._get_meta(idx)


# ── AST Dataset ───────────────────────────────────────────────────────────────


class ASTDataset(_BaseDataset):
    """For AST fine-tuning — returns raw audio at 16kHz."""

    AST_SR = 16000

    def __init__(self, manifest, split, config, augment=None):
        super().__init__(manifest, split, config, augment=augment)
        self.expected_samples = config.CHUNK_SEC * self.AST_SR

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        audio, _ = librosa.load(
            row["file_path"],
            sr=self.AST_SR,
            mono=True,
            offset=float(row["start_sec"]),
            duration=self.config.CHUNK_SEC,
        )
        rms = np.sqrt(np.mean(audio**2))
        if rms > 1e-8:
            audio = audio * (_get(self.config, "RMS_TARGET", 0.1) / rms)
        if len(audio) < self.expected_samples:
            audio = np.pad(audio, (0, self.expected_samples - len(audio)))
        audio = audio[: self.expected_samples]
        waveform = torch.from_numpy(audio).unsqueeze(0)
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return waveform, label, self._get_meta(idx)


# ── DataLoader factory ────────────────────────────────────────────────────────


def _resolve_dataset_class(mtype: str):
    """Map model type string to (DatasetClass, dataset_kwargs)."""
    if mtype in ("mlp", "cnn"):
        return MelSpecDataset, {}
    elif mtype in (
        "cnn_lstm",
        "cnn_gru",  # legacy alias for the original BiGRU model
        "cnn_bigru",
        "cnn_bilstm",
        "cnn_bilstm_attention",
    ):
        return MelSpecSeqDataset, {"segment_frames": 43}
    elif mtype == "ast":
        return ASTDataset, {}
    else:
        raise ValueError(
            f"Unknown model_type='{mtype}'. "
            f"Choose from: mlp, cnn, cnn_lstm, cnn_gru, ast"
        )


def build_dataloaders(
    manifest: pd.DataFrame,
    config,
    model_type: str | None = None,
    batch_size: int | None = None,
    sampler_seed: int | None = None,
) -> dict[str, DataLoader]:
    """
    Build train / val / test DataLoaders.

    Parameters
    ----------
    manifest   : spectrogram manifest DataFrame
    config     : config module (only fixed params needed — paths, spec shape)
    model_type : "mlp" | "cnn" | "cnn_lstm" | "cnn_gru" | "ast"
                 Supports cnn_bigru, cnn_bilstm, and
                 cnn_bilstm_attention for the sequence comparison.
    batch_size : override batch size. Falls back to config.BATCH_SIZE
                 or 64 if neither is set.
    sampler_seed : optional independent seed for sampler and worker RNGs.
                   Reuse this seed across models for identical chunk draws.
    """
    mtype = (model_type or _get(config, "MODEL_TYPE", "cnn")).lower()
    bs = batch_size or _get(config, "BATCH_SIZE", 64)

    DatasetClass, dataset_kwargs = _resolve_dataset_class(mtype)
    log.info(f"\nBuilding DataLoaders — model_type='{mtype}', batch_size={bs}")

    pin = torch.cuda.is_available()
    n_workers = 0 if platform.system() == "Darwin" else 4
    use_sampler = _get(config, "USE_WEIGHTED_SAMPLER", True)

    loaders = {}
    for split in ["train", "val", "test"]:
        if len(manifest[manifest["split"] == split]) == 0:
            log.warning(f"No chunks for split='{split}' — skipping.")
            continue

        dataset = DatasetClass(
            manifest=manifest,
            split=split,
            config=config,
            **dataset_kwargs,
        )

        loader_generator = None
        sampler_generator = None
        if sampler_seed is not None:
            loader_generator = torch.Generator().manual_seed(int(sampler_seed))
            sampler_generator = torch.Generator().manual_seed(int(sampler_seed))

        if split == "train" and use_sampler:
            sampler = _make_weighted_sampler(dataset, generator=sampler_generator)
            shuffle = False
        else:
            sampler = None
            shuffle = split == "train"

        loaders[split] = DataLoader(
            dataset,
            batch_size=bs,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=n_workers,
            pin_memory=pin,
            drop_last=(split == "train"),
            collate_fn=_collate_fn,
            generator=loader_generator,
        )

        log.info(f"  {split:5s}: {len(dataset)} chunks | {len(loaders[split])} batches")

    return loaders


def build_eval_loader(
    manifest: pd.DataFrame,
    config,
    split: str = "train",
    model_type: str | None = None,
    batch_size: int | None = None,
) -> DataLoader:
    """
    Augmentation-free DataLoader for honest metric computation.
    Always uses augment=False regardless of config settings.
    """
    mtype = (model_type or _get(config, "MODEL_TYPE", "cnn")).lower()
    bs = batch_size or _get(config, "BATCH_SIZE", 64)

    DatasetClass, dataset_kwargs = _resolve_dataset_class(mtype)

    dataset = DatasetClass(
        manifest=manifest,
        split=split,
        config=config,
        augment=False,  # always off
        **dataset_kwargs,
    )

    n_workers = 0 if platform.system() == "Darwin" else 4

    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=_collate_fn,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_weighted_sampler(
    dataset: _BaseDataset,
    generator: torch.Generator | None = None,
) -> WeightedRandomSampler:
    labels = dataset.labels
    n_human = (labels == 0).sum()
    n_ai = (labels == 1).sum()
    n_total = len(labels)

    w_human = n_total / (2 * n_human) if n_human > 0 else 1.0
    w_ai = n_total / (2 * n_ai) if n_ai > 0 else 1.0
    weights = torch.from_numpy(np.where(labels == 0, w_human, w_ai)).double()

    log.info(f"  WeightedSampler: human={w_human:.3f}, ai={w_ai:.3f}")
    return WeightedRandomSampler(
        weights,
        num_samples=n_total,
        replacement=True,
        generator=generator,
    )


def _collate_fn(batch):
    inputs, labels, metas = [], [], []
    for inp, lbl, meta in batch:
        assert isinstance(lbl, torch.Tensor)
        inputs.append(inp)
        labels.append(lbl)
        metas.append(meta)
    return torch.stack(inputs), torch.stack(labels), metas
