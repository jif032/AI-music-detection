# =============================================================================
# sampling.py
# =============================================================================
# Generates a chunk index from a track-level split DataFrame.
# No audio is loaded here — we only record (file_path, start_sec, label,
# source, split) for each chunk. Audio loading happens later in spectrogram.py.
#
# Weighted percentile sampling ensures chunks are drawn from musically
# informative regions of each track, with configurable edge probabilities.
# =============================================================================

import numpy as np
import pandas as pd
from pathlib import Path
import logging

log = logging.getLogger(__name__)


# ── Internal helpers ──────────────────────────────────────────────────────────


def _n_chunks_for_source(source_name: str, config) -> int:
    """Return configured chunk count for a given source."""
    mapping = {
        "FMA": config.N_CHUNKS_FMA,
        "Udio": config.N_CHUNKS_UDIO,
        "Suno": config.N_CHUNKS_SUNO,
    }
    return mapping.get(source_name, config.N_CHUNKS_UDIO)


def _sample_start_positions(
    duration_sec: float,
    n_chunks: int,
    chunk_sec: float,
    region_bounds: list,
    region_probs: list,
    rng: np.random.Generator,
) -> list[float]:
    """
    Sample n_chunks start positions (in seconds) from a track of
    duration_sec using weighted percentile region sampling.

    Strategy:
      1. For each chunk, pick a region according to region_probs.
      2. Sample a uniform start position within that region, ensuring
         the full chunk fits within the track.
      3. Retry up to MAX_RETRIES times to avoid overlapping chunks
         (soft constraint — overlap allowed if track is short).

    Returns a list of start positions in seconds (may be < n_chunks
    if the track is too short to fit all non-overlapping chunks).
    """
    MAX_RETRIES = 20
    MIN_GAP_SEC = chunk_sec * 0.5  # allow 50% overlap at most

    usable_duration = duration_sec - chunk_sec
    if usable_duration <= 0:
        return []

    bounds_sec = [b * duration_sec for b in region_bounds]
    positions = []

    for _ in range(n_chunks):
        for attempt in range(MAX_RETRIES):
            # Pick region weighted by region_probs
            region_idx = rng.choice(len(region_probs), p=region_probs)
            lo = bounds_sec[region_idx]
            hi = bounds_sec[region_idx + 1]

            # Clamp so the full chunk fits inside the track
            lo_clamped = max(0.0, lo)
            hi_clamped = min(usable_duration, hi)

            if hi_clamped <= lo_clamped:
                # Region too narrow for this track — try another region
                continue

            start = float(rng.uniform(lo_clamped, hi_clamped))

            # Check minimum gap from already-placed chunks (soft)
            too_close = any(abs(start - p) < MIN_GAP_SEC for p in positions)
            if not too_close or len(positions) == 0:
                positions.append(start)
                break
        # If all retries exhausted, skip this chunk slot

    return positions


# ── Public API ────────────────────────────────────────────────────────────────


def build_chunk_index(
    split_df: pd.DataFrame,
    config,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Build a chunk index DataFrame from a track-level split DataFrame.

    Parameters
    ----------
    split_df    : output of splits.py — must have columns
                  [file_path, label, source_name, duration_sec, split]
    config      : config module
    output_path : if given, saves the chunk index CSV here

    Returns
    -------
    DataFrame with columns:
        chunk_id        unique integer id
        file_path       source wav file
        start_sec       chunk start position in seconds
        end_sec         chunk end position in seconds
        label           "human" | "ai"
        source_name     "FMA" | "Udio" | "Suno"
        split           "train" | "val" | "test"
        track_id        integer — unique per file_path (for GroupKFold)
    """
    rng = np.random.default_rng(config.SEED)

    required = ["file_path", "label", "source_name", "duration_sec", "split"]
    missing = [c for c in required if c not in split_df.columns]
    assert not missing, f"split_df missing columns: {missing}"

    log.info(f"Building chunk index for {len(split_df)} tracks...")

    # Assign integer track IDs (used later for GroupKFold & track-level eval)
    track_id_map = {fp: i for i, fp in enumerate(split_df["file_path"].unique())}

    rows = []
    skipped_tracks = 0

    for _, row in split_df.iterrows():
        if row["duration_sec"] < config.MIN_TRACK_DURATION:
            skipped_tracks += 1
            continue

        n_chunks = _n_chunks_for_source(row["source_name"], config)

        starts = _sample_start_positions(
            duration_sec=row["duration_sec"],
            n_chunks=n_chunks,
            chunk_sec=config.CHUNK_SEC,
            region_bounds=config.SAMPLING_REGION_BOUNDS,
            region_probs=config.SAMPLING_REGION_PROBS,
            rng=rng,
        )

        for start in starts:
            rows.append(
                {
                    "file_path": row["file_path"],
                    "start_sec": round(start, 4),
                    "end_sec": round(start + config.CHUNK_SEC, 4),
                    "label": row["label"],
                    "source_name": row["source_name"],
                    "split": row["split"],
                    "track_id": track_id_map[row["file_path"]],
                }
            )

    chunk_df = pd.DataFrame(rows).reset_index(drop=True)
    chunk_df.insert(0, "chunk_id", range(len(chunk_df)))

    if skipped_tracks:
        log.warning(
            f"Skipped {skipped_tracks} tracks shorter than {config.MIN_TRACK_DURATION}s"
        )

    _log_chunk_summary(chunk_df)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        chunk_df.to_csv(output_path, index=False)
        log.info(f"Saved chunk index → {output_path}")

    return chunk_df


def cap_ai_chunks(
    chunk_df: pd.DataFrame,
    config,
    split: str = "train",
) -> pd.DataFrame:
    """
    Cap AI chunks in a given split to match the human chunk count.
    Operates per-split so val/test counts are not affected by train capping.

    Capping is stratified by source — Suno and Udio are subsampled
    proportionally to their original counts, preserving source diversity.

    Parameters
    ----------
    chunk_df : full chunk index (all splits)
    config   : config module
    split    : which split to cap (typically "train")

    Returns
    -------
    Full chunk_df with AI chunks in `split` capped, all other rows unchanged.
    """
    rng = np.random.default_rng(config.SEED)

    mask_split = chunk_df["split"] == split
    human_mask = mask_split & (chunk_df["label"] == "human")
    ai_mask = mask_split & (chunk_df["label"] == "ai")

    n_human = human_mask.sum()
    n_ai = ai_mask.sum()

    log.info(f"[cap_ai_chunks] split='{split}' | human={n_human} | ai={n_ai}")

    if n_ai <= n_human:
        log.info("AI chunks already ≤ human — no capping needed.")
        return chunk_df

    # Stratified cap: preserve Suno/Udio proportions
    ai_chunks = chunk_df[ai_mask].copy()
    source_counts = ai_chunks["source_name"].value_counts()
    source_fracs = source_counts / source_counts.sum()
    target_per_source = (source_fracs * n_human).round().astype(int)

    # Adjust rounding error so total == n_human
    diff = n_human - target_per_source.sum()
    if diff != 0:
        largest = target_per_source.idxmax()
        target_per_source[largest] += diff

    kept_ai_idx = []
    for source, target in target_per_source.items():
        source_idx = ai_chunks[ai_chunks["source_name"] == source].index.tolist()
        n_keep = min(target, len(source_idx))
        kept = rng.choice(source_idx, size=n_keep, replace=False)
        kept_ai_idx.extend(kept.tolist())
        log.info(f"  {source}: {len(source_idx)} → {n_keep} chunks kept")

    # Rebuild: human rows + capped AI rows + all other splits untouched
    other_splits = chunk_df[~mask_split]
    human_rows = chunk_df[human_mask]
    ai_kept = chunk_df.loc[kept_ai_idx]

    result = (
        pd.concat([other_splits, human_rows, ai_kept])
        .sort_values("chunk_id")
        .reset_index(drop=True)
    )

    log.info(
        f"After capping: human={human_mask.sum()} | "
        f"ai={len(kept_ai_idx)} (split='{split}')"
    )
    return result


def _log_chunk_summary(chunk_df: pd.DataFrame) -> None:
    log.info(f"\n{'=' * 55}")
    log.info(f"Chunk index summary")
    log.info(f"{'=' * 55}")
    summary = (
        chunk_df.groupby(["split", "label", "source_name"])
        .size()
        .rename("chunks")
        .reset_index()
    )
    log.info(f"\n{summary.to_string(index=False)}")

    # Duration coverage stats
    chunk_df = chunk_df.copy()
    log.info(f"\nTotal chunks : {len(chunk_df)}")
    log.info(f"Unique tracks: {chunk_df['file_path'].nunique()}")

    per_track = chunk_df.groupby("file_path").size()
    log.info(
        f"Chunks/track : mean={per_track.mean():.1f} "
        f"min={per_track.min()} max={per_track.max()}"
    )
