# =============================================================================
# sampling.py
# =============================================================================
import numpy as np
import pandas as pd
from pathlib import Path
import logging

log = logging.getLogger(__name__)


def _n_chunks_for_source(source_name: str, is_train: bool, config) -> int:
    """Return configured chunk count for a source and split type."""
    if is_train:
        mapping = {
            "FMA": config.N_CHUNKS_FMA_TRAIN,
            "Udio": config.N_CHUNKS_UDIO_TRAIN,
            "Suno": config.N_CHUNKS_SUNO_TRAIN,
        }
    else:
        mapping = {
            "FMA": config.N_CHUNKS_FMA_EVAL,
            "Udio": config.N_CHUNKS_UDIO_EVAL,
            "Suno": config.N_CHUNKS_SUNO_EVAL,
        }
    return mapping.get(
        source_name,
        config.N_CHUNKS_UDIO_TRAIN if is_train else config.N_CHUNKS_UDIO_EVAL,
    )


def _sample_start_positions(
    duration_sec: float,
    n_chunks: int,
    chunk_sec: float,
    region_bounds: list,
    region_probs: list,
    rng: np.random.Generator,
) -> list[float]:
    """
    Sample n_chunks start positions (seconds) using weighted
    percentile region sampling with soft non-overlap constraint.
    """
    MAX_RETRIES = 20
    MIN_GAP_SEC = chunk_sec * 0.5

    usable_duration = duration_sec - chunk_sec
    if usable_duration <= 0:
        return []

    bounds_sec = [b * duration_sec for b in region_bounds]
    positions = []

    for _ in range(n_chunks):
        for _ in range(MAX_RETRIES):
            region_idx = rng.choice(len(region_probs), p=region_probs)
            lo = max(0.0, bounds_sec[region_idx])
            hi = min(usable_duration, bounds_sec[region_idx + 1])

            if hi <= lo:
                continue

            start = float(rng.uniform(lo, hi))
            too_close = any(abs(start - p) < MIN_GAP_SEC for p in positions)
            if not too_close or len(positions) == 0:
                positions.append(start)
                break

    return positions


def build_chunk_index(
    split_df: pd.DataFrame,
    config,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Build a chunk index from a track-level split DataFrame.

    Uses train chunk counts for split=="train" and eval chunk counts
    for split=="val" and split=="test".

    Returns DataFrame with columns:
        chunk_id, file_path, start_sec, end_sec,
        label, source_name, split, track_id
    """
    rng = np.random.default_rng(config.SEED)

    required = ["file_path", "label", "source_name", "duration_sec", "split"]
    missing = [c for c in required if c not in split_df.columns]
    assert not missing, f"split_df missing columns: {missing}"

    log.info(f"Building chunk index for {len(split_df)} tracks...")

    track_id_map = {fp: i for i, fp in enumerate(split_df["file_path"].unique())}

    rows = []
    skipped = 0

    for _, row in split_df.iterrows():
        if row["duration_sec"] < config.MIN_TRACK_DURATION:
            skipped += 1
            continue

        is_train = row["split"] == "train"
        n_chunks = _n_chunks_for_source(row["source_name"], is_train, config)

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

    if skipped:
        log.warning(
            f"Skipped {skipped} tracks shorter than {config.MIN_TRACK_DURATION}s"
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
    Cap AI chunks in training split to match human count.
    Stratified by source to preserve Suno/Udio proportions.
    Val/test splits are never capped.
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

    ai_chunks = chunk_df[ai_mask].copy()
    source_counts = ai_chunks["source_name"].value_counts()
    source_fracs = source_counts / source_counts.sum()
    target_per_src = (source_fracs * n_human).round().astype(int)

    diff = n_human - target_per_src.sum()
    if diff != 0:
        target_per_src[target_per_src.idxmax()] += diff

    kept_ai_idx = []
    for source, target in target_per_src.items():
        src_idx = ai_chunks[ai_chunks["source_name"] == source].index.tolist()
        n_keep = min(target, len(src_idx))
        kept = rng.choice(src_idx, size=n_keep, replace=False)
        kept_ai_idx.extend(kept.tolist())
        log.info(f"  {source}: {len(src_idx)} → {n_keep} kept")

    other = chunk_df[~mask_split]
    human = chunk_df[human_mask]
    ai = chunk_df.loc[kept_ai_idx]

    result = (
        pd.concat([other, human, ai]).sort_values("chunk_id").reset_index(drop=True)
    )

    log.info(
        f"After capping: human={human_mask.sum()} | "
        f"ai={len(kept_ai_idx)} (split='{split}')"
    )
    return result


def _log_chunk_summary(chunk_df: pd.DataFrame) -> None:
    log.info(f"\n{'=' * 55}")
    log.info("Chunk index summary")
    log.info(f"{'=' * 55}")
    summary = (
        chunk_df.groupby(["split", "label", "source_name"])
        .size()
        .rename("chunks")
        .reset_index()
    )
    log.info(f"\n{summary.to_string(index=False)}")
    log.info(f"\nTotal chunks : {len(chunk_df)}")
    log.info(f"Unique tracks: {chunk_df['file_path'].nunique()}")
    per_track = chunk_df.groupby(["split", "file_path"]).size()
    log.info(
        f"Chunks/track (train): "
        f"mean={per_track['train'].mean():.1f} "
        f"min={per_track['train'].min()} "
        f"max={per_track['train'].max()}"
        if "train" in per_track.index.get_level_values(0)
        else ""
    )
