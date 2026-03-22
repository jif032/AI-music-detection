# =============================================================================
# mert_mining.py
# =============================================================================
# MERT + KNN hard-negative mining for training chunks.
#
# Pipeline:
#   1. Extract MERT embeddings for all training chunks
#   2. Run KNN in embedding space to find "confused" chunks
#      (chunks whose K nearest neighbors contain >= threshold% opposite class)
#   3. Replace HARD_NEGATIVE_FRACTION of training chunks with hard negatives,
#      stratified by source to preserve Suno/Udio proportions
#   4. Log detailed statistics at every stage
#
# Only training chunks are processed. Val/test chunks are never touched.
# =============================================================================

import numpy as np
import pandas as pd
import torch
import librosa
import soundfile as sf
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import logging
import time

log = logging.getLogger(__name__)


# ── Audio utilities ───────────────────────────────────────────────────────────


def _rms_normalize(audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    rms = np.sqrt(np.mean(audio**2))
    if rms < 1e-8:
        return audio
    return audio * (target_rms / rms)


def _load_chunk_for_mert(
    file_path: str,
    start_sec: float,
    chunk_sec: float,
    mert_sr: int,
    rms_target: float,
) -> np.ndarray:
    """
    Load one chunk from a wav file, resample to MERT SR, RMS normalize.
    Pads with zeros if the loaded audio is shorter than expected.
    """
    expected_samples = int(chunk_sec * mert_sr)

    audio, _ = librosa.load(
        file_path,
        sr=mert_sr,
        mono=True,
        offset=start_sec,
        duration=chunk_sec,
    )
    audio = _rms_normalize(audio, rms_target)

    if len(audio) < expected_samples:
        audio = np.pad(audio, (0, expected_samples - len(audio)))

    return audio[:expected_samples]


# ── MERT embedding extraction ─────────────────────────────────────────────────


def _load_mert(config, device: torch.device):
    """Load MERT model and processor. Returns (model, processor)."""
    from transformers import AutoModel, AutoProcessor

    log.info(f"Loading MERT from {config.MERT_MODEL}...")
    processor = AutoProcessor.from_pretrained(config.MERT_MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(config.MERT_MODEL, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    log.info("MERT loaded successfully.")
    return model, processor


def _extract_embedding(
    audio: np.ndarray,
    model,
    processor,
    device: torch.device,
    layer: int,
    mert_sr: int,
) -> np.ndarray:
    """Extract mean-pooled embedding from a specific MERT layer."""
    inputs = processor(
        audio,
        sampling_rate=mert_sr,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states[layer]  # (1, time, 768)
    return hidden.mean(dim=1).squeeze(0).cpu().numpy()  # (768,)


def extract_mert_embeddings(
    chunk_df: pd.DataFrame,
    config,
    device: torch.device,
    log_every: int = 200,
) -> np.ndarray:
    """
    Extract MERT embeddings for every chunk in chunk_df.

    Parameters
    ----------
    chunk_df  : chunk index (any split — caller decides what to pass)
    config    : config module
    device    : torch device
    log_every : log progress every N chunks

    Returns
    -------
    embeddings : np.ndarray of shape (len(chunk_df), 768)
    """
    model, processor = _load_mert(config, device)

    n = len(chunk_df)
    emb_dim = 768
    embeddings = np.zeros((n, emb_dim), dtype=np.float32)

    log.info(
        f"Extracting MERT (layer {config.MERT_LAYER}) embeddings for {n} chunks..."
    )
    t0 = time.time()

    for i, (_, row) in enumerate(chunk_df.iterrows()):
        audio = _load_chunk_for_mert(
            file_path=row["file_path"],
            start_sec=row["start_sec"],
            chunk_sec=config.CHUNK_SEC,
            mert_sr=config.MERT_SR,
            rms_target=config.RMS_TARGET,
        )
        embeddings[i] = _extract_embedding(
            audio=audio,
            model=model,
            processor=processor,
            device=device,
            layer=config.MERT_LAYER,
            mert_sr=config.MERT_SR,
        )

        if (i + 1) % log_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (n - i - 1) / rate
            log.info(
                f"  {i + 1}/{n} chunks | "
                f"{elapsed:.0f}s elapsed | "
                f"~{remaining:.0f}s remaining"
            )

    log.info(f"Embedding extraction complete — {time.time() - t0:.1f}s total")
    return embeddings


# ── KNN confusion scoring ─────────────────────────────────────────────────────


def compute_confusion_scores(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    For each chunk, compute its confusion score:
      fraction of K nearest neighbors (excluding self) from the opposite class.

    Parameters
    ----------
    embeddings : (N, D) float32
    labels     : (N,) int — 0=human, 1=ai
    k          : number of neighbors

    Returns
    -------
    confusion_scores : (N,) float — values in [0, 1]
    """
    log.info(f"Running KNN (k={k}) on {len(embeddings)} embeddings...")
    t0 = time.time()

    # Normalize before KNN so cosine ~ euclidean distance
    scaler = StandardScaler()
    embs_normed = scaler.fit_transform(embeddings)

    # k+1 because the first neighbor is always the point itself
    knn = NearestNeighbors(
        n_neighbors=k + 1,
        metric="cosine",
        n_jobs=-1,
    )
    knn.fit(embs_normed)
    _, indices = knn.kneighbors(embs_normed)

    # Exclude self (index 0) and compute opposite-class fraction
    confusion_scores = np.zeros(len(embeddings), dtype=np.float32)
    for i, neighbor_idx in enumerate(indices):
        neighbor_idx = neighbor_idx[1:]  # exclude self
        neighbor_labels = labels[neighbor_idx]
        opposite = (neighbor_labels != labels[i]).sum()
        confusion_scores[i] = opposite / k

    log.info(f"KNN complete — {time.time() - t0:.1f}s")
    log.info(f"Confusion score stats:")
    log.info(f"  mean   = {confusion_scores.mean():.4f}")
    log.info(f"  median = {np.median(confusion_scores):.4f}")
    log.info(f"  >0.30  = {(confusion_scores >= 0.30).sum()} chunks")
    log.info(f"  >0.50  = {(confusion_scores >= 0.50).sum()} chunks")

    return confusion_scores


# ── Hard-negative selection ───────────────────────────────────────────────────


def _log_mining_stats(
    chunk_df: pd.DataFrame,
    confusion_scores: np.ndarray,
    hard_negative_mask: np.ndarray,
    config,
) -> None:
    """Log detailed per-source confusion statistics."""
    df = chunk_df.copy()
    df["confusion_score"] = confusion_scores
    df["is_hard_negative"] = hard_negative_mask

    log.info(f"\n{'=' * 55}")
    log.info("Hard-negative mining statistics")
    log.info(f"{'=' * 55}")

    for source in df["source_name"].unique():
        sub = df[df["source_name"] == source]
        hn = sub["is_hard_negative"].sum()
        log.info(
            f"  {source:8s} | total={len(sub):5d} | "
            f"confused (≥{config.KNN_CONFUSION_THRESHOLD:.0%})="
            f"{(sub['confusion_score'] >= config.KNN_CONFUSION_THRESHOLD).sum():5d} | "
            f"selected as HN={hn:5d} | "
            f"unique tracks={sub[sub['is_hard_negative']]['file_path'].nunique()}"
        )

    total_hn = hard_negative_mask.sum()
    log.info(f"\n  Total hard negatives selected : {total_hn}")
    log.info(f"  Fraction of training set      : {total_hn / len(chunk_df):.2%}")


def select_hard_negatives(
    chunk_df: pd.DataFrame,
    confusion_scores: np.ndarray,
    config,
) -> pd.DataFrame:
    """
    Replace HARD_NEGATIVE_FRACTION of training chunks with hard negatives.

    Strategy:
      - A chunk qualifies as a hard negative if its confusion score
        >= KNN_CONFUSION_THRESHOLD.
      - We replace ceil(HARD_NEGATIVE_FRACTION * N_train) random chunks
        with qualified hard negatives, stratified by source.
      - If fewer hard negatives exist than needed, we use all available
        and log a warning.
      - No chunk appears twice in the final index.

    Parameters
    ----------
    chunk_df         : training chunks only (split == "train")
    confusion_scores : (len(chunk_df),) — output of compute_confusion_scores
    config           : config module

    Returns
    -------
    Updated chunk DataFrame where ~HARD_NEGATIVE_FRACTION of rows
    are hard negatives and the rest are randomly sampled easy examples.
    """
    rng = np.random.default_rng(config.SEED)

    df = chunk_df.copy().reset_index(drop=True)
    df["confusion_score"] = confusion_scores

    n_total = len(df)
    n_hard = int(np.ceil(n_total * config.HARD_NEGATIVE_FRACTION))
    n_easy = n_total - n_hard

    log.info(f"\nSelecting hard negatives...")
    log.info(f"  Total train chunks : {n_total}")
    log.info(f"  Target hard        : {n_hard} ({config.HARD_NEGATIVE_FRACTION:.0%})")
    log.info(f"  Target easy        : {n_easy}")

    # Identify qualified hard negatives
    qualified_mask = confusion_scores >= config.KNN_CONFUSION_THRESHOLD
    qualified_df = df[qualified_mask].copy()

    log.info(
        f"  Qualified (score ≥ {config.KNN_CONFUSION_THRESHOLD:.0%}): "
        f"{len(qualified_df)}"
    )

    if len(qualified_df) < n_hard:
        log.warning(
            f"Only {len(qualified_df)} qualified hard negatives available "
            f"(need {n_hard}). Using all qualified chunks."
        )
        n_hard = len(qualified_df)
        n_easy = n_total - n_hard

    # Stratified hard-negative selection (preserve source proportions)
    source_counts = qualified_df["source_name"].value_counts()
    source_fracs = source_counts / source_counts.sum()
    target_hn_per_source = (source_fracs * n_hard).round().astype(int)

    # Fix rounding
    diff = n_hard - target_hn_per_source.sum()
    if diff != 0:
        target_hn_per_source[target_hn_per_source.idxmax()] += diff

    selected_hn_idx = []
    for source, target in target_hn_per_source.items():
        pool = qualified_df[qualified_df["source_name"] == source].index.tolist()
        n_pick = min(target, len(pool))
        chosen = rng.choice(pool, size=n_pick, replace=False).tolist()
        selected_hn_idx.extend(chosen)
        log.info(f"  Hard neg | {source:8s}: {n_pick} selected")

    # Easy chunks = everything not selected as hard negative
    all_idx = set(df.index.tolist())
    hn_idx_set = set(selected_hn_idx)
    easy_pool = list(all_idx - hn_idx_set)

    selected_easy_idx = rng.choice(
        easy_pool,
        size=min(n_easy, len(easy_pool)),
        replace=False,
    ).tolist()

    # Build hard_negative_mask for logging
    hard_negative_mask = np.zeros(len(df), dtype=bool)
    hard_negative_mask[selected_hn_idx] = True

    _log_mining_stats(df, confusion_scores, hard_negative_mask, config)

    # Combine and tag
    final_idx = selected_hn_idx + selected_easy_idx
    result = df.loc[final_idx].copy()
    result["is_hard_negative"] = result.index.isin(hn_idx_set)
    result = result.reset_index(drop=True)
    result["chunk_id"] = range(len(result))

    log.info(f"\nFinal training set:")
    log.info(
        result.groupby(["label", "source_name", "is_hard_negative"])
        .size()
        .rename("chunks")
        .to_string()
    )
    log.info(f"Unique tracks: {result['file_path'].nunique()}")

    return result


# ── Public entry point ────────────────────────────────────────────────────────


def run_hard_negative_mining(
    chunk_df: pd.DataFrame,
    config,
    device: torch.device,
    embeddings_cache_path: Path | None = None,
) -> pd.DataFrame:
    """
    Full hard-negative mining pipeline for training chunks.

    Parameters
    ----------
    chunk_df              : full chunk index (all splits)
    config                : config module
    device                : torch device
    embeddings_cache_path : if given, save/load embeddings from this .npy path
                            to avoid re-extracting on reruns

    Returns
    -------
    Full chunk_df with training chunks replaced by hard-negative-mined version.
    Val/test chunks are returned unchanged.
    """
    # Separate train from val/test — never touch val/test
    train_mask = chunk_df["split"] == "train"
    train_chunks = chunk_df[train_mask].copy().reset_index(drop=True)
    other_chunks = chunk_df[~train_mask].copy()

    log.info(
        f"Hard-negative mining on {len(train_chunks)} training chunks "
        f"({other_chunks['split'].value_counts().to_dict()} unchanged)"
    )

    # ── Step 1: Extract or load embeddings ───────────────────────────────
    if embeddings_cache_path is not None and embeddings_cache_path.exists():
        log.info(f"Loading cached embeddings from {embeddings_cache_path}")
        embeddings = np.load(embeddings_cache_path)
        assert len(embeddings) == len(train_chunks), (
            f"Cached embeddings length {len(embeddings)} != "
            f"train chunks {len(train_chunks)} — delete cache and rerun."
        )
    else:
        embeddings = extract_mert_embeddings(train_chunks, config, device)
        if embeddings_cache_path is not None:
            embeddings_cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(embeddings_cache_path, embeddings)
            log.info(f"Embeddings cached → {embeddings_cache_path}")

    # ── Step 2: KNN confusion scoring ─────────────────────────────────────
    labels = (train_chunks["label"] == "ai").astype(int).values
    confusion_scores = compute_confusion_scores(
        embeddings=embeddings,
        labels=labels,
        k=config.KNN_K,
    )

    # ── Step 3: Select hard negatives ─────────────────────────────────────
    mined_train = select_hard_negatives(
        chunk_df=train_chunks,
        confusion_scores=confusion_scores,
        config=config,
    )

    # ── Step 4: Recombine with val/test ───────────────────────────────────
    result = (
        pd.concat([mined_train, other_chunks])
        .sort_values(["split", "chunk_id"])
        .reset_index(drop=True)
    )

    log.info(f"\nFinal chunk index after mining:")
    log.info(
        result.groupby(["split", "label", "source_name"])
        .size()
        .rename("chunks")
        .to_string()
    )

    return result
