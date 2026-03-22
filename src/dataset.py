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
) -> torch.Tensor:
    n_mels, n_frames = spec.shape
    for _ in range(n_freq_masks):
        f = torch.randint(0, freq_mask_param + 1, (1,)).item()
        f_start = torch.randint(0, max(1, n_mels - f), (1,)).item()
        spec[f_start : f_start + f, :] = 0.0
    for _ in range(n_time_masks):
        t = torch.randint(0, time_mask_param + 1, (1,)).item()
        t_start = torch.randint(0, max(1, n_frames - t), (1,)).item()
        spec[:, t_start : t_start + t] = 0.0
    return spec


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

        # augment=None → follow split (True for train, False for val/test)
        # augment=False → always off (used for train metric evaluation)
        self.augment = self.is_train if augment is None else augment

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
        npy_path = self.df.iloc[idx]["npy_path"]
        return np.load(npy_path).astype(np.float32)

    def _maybe_augment(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment only when self.augment is True."""
        if self.augment and self.config.USE_AUGMENTATION:
            spec = spec_augment(
                spec=spec,
                freq_mask_param=self.config.FREQ_MASK_PARAM,
                time_mask_param=self.config.TIME_MASK_PARAM,
                n_freq_masks=self.config.N_FREQ_MASKS,
                n_time_masks=self.config.N_TIME_MASKS,
            )
        return spec


# ── CNN Dataset ───────────────────────────────────────────────────────────────


class MelSpecDataset(_BaseDataset):
    """CNN dataset — returns (1, n_mels, time_frames)."""

    def __init__(self, manifest, split, config, augment=None):
        super().__init__(manifest, split, config, augment=augment)

    def __getitem__(self, idx: int):
        spec = self._load_spec(idx)
        spec = torch.from_numpy(spec)
        spec = self._maybe_augment(spec)
        spec = spec.unsqueeze(0)
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return spec, label, self._get_meta(idx)


# ── CNN + RNN Dataset ─────────────────────────────────────────────────────────


class MelSpecSeqDataset(_BaseDataset):
    """CNN+LSTM/GRU dataset — returns (n_segments, 1, n_mels, segment_frames)."""

    def __init__(self, manifest, split, config, segment_frames=43, augment=None):
        super().__init__(manifest, split, config, augment=augment)
        self.segment_frames = segment_frames
        self.n_segments = config.SPEC_TIME_FRAMES // segment_frames
        log.info(
            f"  Sequence mode: {self.n_segments} segments × "
            f"{segment_frames} frames each"
        )

    def __getitem__(self, idx: int):
        spec = self._load_spec(idx)
        spec = torch.from_numpy(spec)
        spec = self._maybe_augment(spec)

        usable_frames = self.n_segments * self.segment_frames
        spec = spec[:, :usable_frames]
        spec = (
            spec.reshape(self.config.N_MELS, self.n_segments, self.segment_frames)
            .permute(1, 0, 2)
            .unsqueeze(1)
        )

        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return spec, label, self._get_meta(idx)


# ── AST Dataset ───────────────────────────────────────────────────────────────


class ASTDataset(_BaseDataset):
    """AST dataset — returns raw audio at 16kHz."""

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
            audio = audio * (self.config.RMS_TARGET / rms)
        if len(audio) < self.expected_samples:
            audio = np.pad(audio, (0, self.expected_samples - len(audio)))
        audio = audio[: self.expected_samples]
        waveform = torch.from_numpy(audio).unsqueeze(0)
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return waveform, label, self._get_meta(idx)


# ── DataLoader factory ────────────────────────────────────────────────────────


def build_dataloaders(
    manifest: pd.DataFrame,
    config,
    model_type: str | None = None,
) -> dict[str, DataLoader]:

    mtype = (model_type or config.MODEL_TYPE).lower()

    if mtype == "cnn":
        DatasetClass = MelSpecDataset
        dataset_kwargs = {}
    elif mtype in ("cnn_lstm", "cnn_gru"):
        DatasetClass = MelSpecSeqDataset
        dataset_kwargs = {"segment_frames": 43}
    elif mtype == "ast":
        DatasetClass = ASTDataset
        dataset_kwargs = {}
    else:
        raise ValueError(f"Unknown model_type='{mtype}'.")

    log.info(f"\nBuilding DataLoaders — model_type='{mtype}'")

    pin = torch.cuda.is_available()
    n_workers = 0 if platform.system() == "Darwin" else 4

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

        if split == "train" and config.USE_WEIGHTED_SAMPLER:
            sampler = _make_weighted_sampler(dataset)
            shuffle = False
        else:
            sampler = None
            shuffle = split == "train"

        loaders[split] = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=n_workers,
            pin_memory=pin,
            drop_last=(split == "train"),
            collate_fn=_collate_fn,
        )

        log.info(
            f"  {split:5s} loader: {len(dataset)} chunks | "
            f"{len(loaders[split])} batches | "
            f"batch_size={config.BATCH_SIZE}"
        )

    return loaders


def build_eval_loader(
    manifest: pd.DataFrame,
    config,
    split: str = "train",
    model_type: str | None = None,
) -> DataLoader:
    """
    Build a DataLoader for metric evaluation with augmentation disabled.
    Used to compute honest train metrics without SpecAugment corruption.
    """
    mtype = (model_type or config.MODEL_TYPE).lower()

    if mtype == "cnn":
        DatasetClass = MelSpecDataset
        dataset_kwargs = {}
    elif mtype in ("cnn_lstm", "cnn_gru"):
        DatasetClass = MelSpecSeqDataset
        dataset_kwargs = {"segment_frames": 43}
    else:
        DatasetClass = MelSpecDataset
        dataset_kwargs = {}

    dataset = DatasetClass(
        manifest=manifest,
        split=split,
        config=config,
        augment=False,  # ← always off for evaluation
        **dataset_kwargs,
    )

    n_workers = 0 if platform.system() == "Darwin" else 4

    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=_collate_fn,
    )


def _make_weighted_sampler(dataset: _BaseDataset) -> WeightedRandomSampler:
    labels = dataset.labels
    n_human = (labels == 0).sum()
    n_ai = (labels == 1).sum()
    n_total = len(labels)

    weight_human = n_total / (2 * n_human) if n_human > 0 else 1.0
    weight_ai = n_total / (2 * n_ai) if n_ai > 0 else 1.0

    weights = np.where(labels == 0, weight_human, weight_ai)
    weights = torch.from_numpy(weights).double()

    log.info(
        f"  WeightedSampler: human_weight={weight_human:.3f}, ai_weight={weight_ai:.3f}"
    )

    return WeightedRandomSampler(
        weights=weights,
        num_samples=n_total,
        replacement=True,
    )


def _collate_fn(batch):
    inputs, labels, metas = [], [], []
    for inp, lbl, meta in batch:
        assert isinstance(lbl, torch.Tensor)
        inputs.append(inp)
        labels.append(lbl)
        metas.append(meta)
    return torch.stack(inputs), torch.stack(labels), metas
