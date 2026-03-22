# =============================================================================
# spectrogram.py
# =============================================================================
# Generates mel spectrograms from the chunk index and saves them as
# float32 .npy files. Writes a manifest CSV mapping each chunk to its
# .npy file path for use by the PyTorch Dataset.
#
# Design decisions:
#   - One .npy file per chunk for memory-mapped loading
#   - Spectrograms saved in dB scale using a FIXED reference (1.0)
#     rather than per-spectrogram max — this preserves relative loudness
#     differences across chunks so global normalization is meaningful
#   - Global standardization (zero mean / unit variance) computed from
#     training set only — val/test use training statistics
#   - Two-pass design: generate all → compute train stats → normalize all
# =============================================================================

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
import logging
import time
import json

log = logging.getLogger(__name__)


# ── Audio loading ─────────────────────────────────────────────────────────────


def _rms_normalize(audio: np.ndarray, target_rms: float) -> np.ndarray:
    rms = np.sqrt(np.mean(audio**2))
    if rms < 1e-8:
        return audio
    return audio * (target_rms / rms)


def _load_chunk(
    file_path: str,
    start_sec: float,
    chunk_sec: float,
    sr: int,
    rms_target: float,
) -> np.ndarray:
    """
    Load one audio chunk from a wav file.
    Applies RMS normalization and zero-pads if shorter than expected.
    """
    expected_samples = int(chunk_sec * sr)

    audio, _ = librosa.load(
        file_path,
        sr=sr,
        mono=True,
        offset=start_sec,
        duration=chunk_sec,
    )
    audio = _rms_normalize(audio, rms_target)

    if len(audio) < expected_samples:
        audio = np.pad(audio, (0, expected_samples - len(audio)))

    return audio[:expected_samples]


# ── Spectrogram generation ────────────────────────────────────────────────────


def _audio_to_melspec(
    audio: np.ndarray,
    sr: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    f_max: int,
) -> np.ndarray:
    """
    Convert audio waveform to log-mel spectrogram.

    Uses a FIXED reference of 1.0 for power_to_db instead of
    ref=np.max. This is critical: ref=np.max normalizes each
    spectrogram to its own peak, destroying the relative amplitude
    information needed for consistent global normalization.

    Returns
    -------
    spec : np.ndarray of shape (n_mels, time_frames), float32, dB scale
           Values will be in roughly [-80, 0] dB range before
           global normalization is applied.
    """
    S = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmax=f_max,
    )
    # Fixed reference = 1.0 so all spectrograms share the same scale
    S_db = librosa.power_to_db(S, ref=1.0).astype(np.float32)
    return S_db


def _pad_or_crop(spec: np.ndarray, target_frames: int) -> np.ndarray:
    """
    Ensure spectrogram has exactly target_frames time steps.
    Crops if longer, zero-pads if shorter.
    """
    n_frames = spec.shape[1]
    if n_frames == target_frames:
        return spec
    if n_frames > target_frames:
        return spec[:, :target_frames]
    pad_width = target_frames - n_frames
    # Pad with the minimum value (silence in dB scale)
    return np.pad(
        spec, ((0, 0), (0, pad_width)), mode="constant", constant_values=spec.min()
    )


# ── Normalization ─────────────────────────────────────────────────────────────


def compute_normalization_stats(
    manifest: pd.DataFrame,
) -> tuple[float, float]:
    """
    Compute global mean and std across all training spectrograms.
    Uses Welford's online algorithm for numerical stability.
    Only called on training chunks — stats applied to val/test.

    Parameters
    ----------
    manifest : training manifest rows (split == "train")

    Returns
    -------
    (mean, std) as Python floats
    """
    log.info(f"Computing normalization stats from {len(manifest)} training files...")
    t0 = time.time()

    running_sum = 0.0
    running_sq = 0.0
    running_count = 0

    for _, row in manifest.iterrows():
        spec = np.load(row["npy_path"]).astype(np.float64)
        running_sum += float(spec.sum())
        running_sq += float((spec**2).sum())
        running_count += spec.size

    mean = running_sum / running_count
    var = (running_sq / running_count) - (mean**2)
    std = float(np.sqrt(max(var, 1e-8)))

    log.info(f"Stats computed in {time.time() - t0:.1f}s")
    log.info(f"  mean = {mean:.6f}  (should be in [-80, 0] range)")
    log.info(f"  std  = {std:.6f}   (should be > 5 for dB spectrograms)")
    return float(mean), std


def apply_normalization(
    spec: np.ndarray,
    mean: float,
    std: float,
) -> np.ndarray:
    """Standardize a spectrogram to zero mean / unit variance."""
    return ((spec - mean) / std).astype(np.float32)


# ── Main generation pipeline ──────────────────────────────────────────────────


def generate_spectrograms(
    chunk_df: pd.DataFrame,
    config,
    output_dir: Path | None = None,
    log_every: int = 500,
) -> pd.DataFrame:
    """
    Generate and save mel spectrograms for all chunks in chunk_df.

    Two-pass design:
      Pass 1 — generate raw (unnormalized) spectrograms, save as .npy
      Pass 2 — compute train stats, normalize all files, overwrite .npy

    Parameters
    ----------
    chunk_df   : final chunk index (all splits)
    config     : config module
    output_dir : directory for .npy files (default: config.SPECTROGRAMS_DIR)
    log_every  : log progress every N chunks

    Returns
    -------
    manifest : DataFrame with all chunk_df columns plus:
                 npy_path   — absolute path to .npy file
                 spec_mean  — training mean applied during normalization
                 spec_std   — training std applied during normalization
    """
    if output_dir is None:
        output_dir = config.SPECTROGRAMS_DIR
    output_dir = Path(output_dir)

    for split in chunk_df["split"].unique():
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    errors = []

    # ── Pass 1: generate raw spectrograms ─────────────────────────────────
    log.info(f"\n{'=' * 55}")
    log.info(f"Pass 1 — generating raw spectrograms")
    log.info(f"Total chunks : {len(chunk_df)}")
    log.info(f"Output dir   : {output_dir}")
    log.info(f"Spec shape   : {config.SPEC_SHAPE}")
    log.info(
        f"Note: using fixed dB reference (ref=1.0) for consistent cross-chunk scaling"
    )
    log.info(f"{'=' * 55}")
    t0 = time.time()

    for i, (_, row) in enumerate(chunk_df.iterrows()):
        try:
            audio = _load_chunk(
                file_path=row["file_path"],
                start_sec=row["start_sec"],
                chunk_sec=config.CHUNK_SEC,
                sr=config.SAMPLE_RATE,
                rms_target=config.RMS_TARGET,
            )

            spec = _audio_to_melspec(
                audio=audio,
                sr=config.SAMPLE_RATE,
                n_mels=config.N_MELS,
                n_fft=config.N_FFT,
                hop_length=config.HOP_LENGTH,
                f_max=config.F_MAX,
            )

            spec = _pad_or_crop(spec, config.SPEC_TIME_FRAMES)

            fname = f"chunk_{row['chunk_id']:06d}.npy"
            npy_path = output_dir / row["split"] / fname
            np.save(npy_path, spec)

            manifest_rows.append(
                {
                    **row.to_dict(),
                    "npy_path": str(npy_path),
                }
            )

        except Exception as e:
            log.warning(f"  Error on chunk {row['chunk_id']} ({row['file_path']}): {e}")
            errors.append(row["chunk_id"])
            continue

        if (i + 1) % log_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(chunk_df) - i - 1) / rate
            log.info(
                f"  {i + 1}/{len(chunk_df)} | "
                f"{elapsed:.0f}s elapsed | "
                f"~{remaining:.0f}s remaining | "
                f"errors: {len(errors)}"
            )

    manifest = pd.DataFrame(manifest_rows)
    log.info(
        f"\nPass 1 complete — {len(manifest)} spectrograms saved "
        f"({len(errors)} errors) in {time.time() - t0:.1f}s"
    )

    if errors:
        log.warning(f"Failed chunk IDs: {errors}")

    # ── Verify raw spectrogram range before normalization ──────────────────
    sample_specs = np.stack(
        [
            np.load(manifest.iloc[i]["npy_path"])
            for i in np.linspace(0, len(manifest) - 1, 20, dtype=int)
        ]
    )
    log.info(f"\nRaw spectrogram stats (pre-normalization):")
    log.info(
        f"  mean={sample_specs.mean():.3f}  "
        f"std={sample_specs.std():.3f}  "
        f"min={sample_specs.min():.3f}  "
        f"max={sample_specs.max():.3f}"
    )
    log.info(f"  Expected range: roughly [-80, 0] dB")

    # ── Pass 2: compute normalization stats from training set only ─────────
    log.info(f"\n{'=' * 55}")
    log.info("Pass 2 — computing normalization statistics")
    log.info(f"{'=' * 55}")

    train_manifest = manifest[manifest["split"] == "train"]
    if len(train_manifest) == 0:
        raise ValueError("No training chunks — cannot compute stats.")

    spec_mean, spec_std = compute_normalization_stats(train_manifest)

    log.info(f"\nNormalization parameters:")
    log.info(f"  mean = {spec_mean:.4f}")
    log.info(f"  std  = {spec_std:.4f}")

    # ── Pass 3: normalize and overwrite all files ──────────────────────────
    log.info(f"\n{'=' * 55}")
    log.info("Pass 3 — normalizing all splits")
    log.info(f"{'=' * 55}")
    t0 = time.time()

    for i, (_, row) in enumerate(manifest.iterrows()):
        spec = np.load(row["npy_path"])
        spec = apply_normalization(spec, spec_mean, spec_std)
        np.save(row["npy_path"], spec)

        if (i + 1) % log_every == 0:
            log.info(f"  Normalized {i + 1}/{len(manifest)} files...")

    log.info(f"Pass 3 complete in {time.time() - t0:.1f}s")

    # ── Verify final normalization ─────────────────────────────────────────
    sample_specs = np.stack(
        [
            np.load(manifest.iloc[i]["npy_path"])
            for i in np.linspace(0, len(manifest) - 1, 20, dtype=int)
        ]
    )
    log.info(f"\nFinal spectrogram stats (post-normalization):")
    log.info(f"  mean={sample_specs.mean():.4f}  (target: ~0.0)")
    log.info(f"  std ={sample_specs.std():.4f}   (target: ~1.0)")
    log.info(f"  min ={sample_specs.min():.4f}")
    log.info(f"  max ={sample_specs.max():.4f}")

    # ── Add stats to manifest ──────────────────────────────────────────────
    manifest = manifest.copy()
    manifest["spec_mean"] = spec_mean
    manifest["spec_std"] = spec_std

    # ── Save manifest and stats ────────────────────────────────────────────
    manifest_path = output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    log.info(f"\nManifest saved → {manifest_path}")

    stats_path = output_dir / "norm_stats.json"
    with open(stats_path, "w") as f:
        json.dump({"mean": spec_mean, "std": spec_std}, f, indent=2)
    log.info(f"Norm stats  → {stats_path}")

    _log_spec_summary(manifest)
    return manifest


def _log_spec_summary(manifest: pd.DataFrame) -> None:
    log.info(f"\n{'=' * 55}")
    log.info("Spectrogram generation summary")
    log.info(f"{'=' * 55}")
    summary = (
        manifest.groupby(["split", "label", "source_name"])
        .size()
        .rename("chunks")
        .reset_index()
    )
    log.info(f"\n{summary.to_string(index=False)}")
    n = len(manifest)
    try:
        sample_size = Path(manifest.iloc[0]["npy_path"]).stat().st_size
        total_gb = (sample_size * n) / (1024**3)
        log.info(
            f"\nEstimated disk usage: {total_gb:.2f} GB "
            f"({n} files × {sample_size / 1024:.1f} KB each)"
        )
    except Exception:
        pass


# ── Utility: reload manifest for subsequent sessions ─────────────────────────


def load_manifest(spec_dir: Path) -> tuple[pd.DataFrame, float, float]:
    """
    Reload a previously generated manifest and normalization stats.
    Use this in subsequent sessions to skip regeneration.

    Returns
    -------
    (manifest_df, spec_mean, spec_std)
    """
    manifest_path = Path(spec_dir) / "manifest.csv"
    stats_path = Path(spec_dir) / "norm_stats.json"

    manifest = pd.read_csv(manifest_path)
    with open(stats_path) as f:
        stats = json.load(f)

    log.info(f"Loaded manifest: {len(manifest)} chunks")
    log.info(f"Norm stats: mean={stats['mean']:.6f}, std={stats['std']:.6f}")

    return manifest, stats["mean"], stats["std"]
