# =============================================================================
# mert_mining.py
# =============================================================================
# MERT-based hard negative mining — KNN neighbor fraction method.
#
# Hard negative identification:
#   For each chunk, find its K nearest neighbors from ALL chunks (human+AI).
#   A chunk is "hard" if the fraction of its neighbors from the opposite class
#   exceeds neighbor_confusion_threshold.
#   e.g. a human chunk whose 4 of 10 neighbors are AI → confusion = 0.40 → hard
#
# Subset construction:
#   1. Identify all hard human and hard AI chunks (sorted by confusion desc)
#   2. Cap hard chunks at hard_negative_fraction * target_n
#      removing least-hard chunks if needed
#   3. Fill remainder to target_n with random sampling
#      balancing the full subset to 50/50 human/AI overall
# =============================================================================

import numpy as np
import pandas as pd
import torch
import logging
from pathlib import Path

log = logging.getLogger(__name__)


# ── MERT embedding extraction ─────────────────────────────────────────────────


def extract_mert_embeddings(
    chunk_df: pd.DataFrame,
    config,
    device: torch.device,
    cache_path: Path | None = None,
) -> np.ndarray:
    """
    Extract MERT embeddings for all chunks in chunk_df.
    Loads from cache if it exists and chunk count matches.

    Returns
    -------
    embeddings : (N, 768) float32 array
    """
    if cache_path is not None and Path(cache_path).exists():
        cached = np.load(cache_path)
        if cached.shape[0] == len(chunk_df):
            log.info(
                f"Loaded cached embeddings: {cache_path} ({cached.shape[0]} chunks)"
            )
            return cached
        log.warning(
            f"Cache shape mismatch ({cached.shape[0]} vs "
            f"{len(chunk_df)}) — recomputing."
        )

    log.info(f"Extracting MERT embeddings for {len(chunk_df)} chunks...")
    log.info(f"  Model : {config.MERT_MODEL}")
    log.info(f"  Layer : {config.MERT_LAYER}")
    log.info(f"  SR    : {config.MERT_SR}")

    try:
        from transformers import AutoModel, AutoProcessor
    except ImportError:
        raise ImportError(
            "transformers not installed. "
            "Run: pip install transformers --break-system-packages"
        )

    import librosa
    import time

    model = AutoModel.from_pretrained(config.MERT_MODEL, trust_remote_code=True).to(
        device
    )
    processor = AutoProcessor.from_pretrained(config.MERT_MODEL, trust_remote_code=True)
    model.eval()

    embeddings = []
    n_chunks = len(chunk_df)
    start_time = time.time()

    with torch.no_grad():
        for i, (_, row) in enumerate(chunk_df.iterrows()):
            if i % 500 == 0:
                if i > 0:
                    elapsed = time.time() - start_time
                    per_chunk = elapsed / i
                    remaining = (n_chunks - i) * per_chunk
                    hrs = int(remaining // 3600)
                    mins = int((remaining % 3600) // 60)
                    log.info(
                        f"  [{i}/{n_chunks}] "
                        f"{per_chunk:.2f}s/chunk | "
                        f"ETA: {hrs}h {mins}m"
                    )
                else:
                    log.info(f"  [{i}/{n_chunks}] starting...")

            try:
                audio, _ = librosa.load(
                    row["file_path"],
                    sr=config.MERT_SR,
                    mono=True,
                    offset=float(row["start_sec"]),
                    duration=config.CHUNK_SEC,
                )
            except Exception as e:
                log.warning(f"  Failed: {row['file_path']} — {e}")
                embeddings.append(np.zeros(768, dtype=np.float32))
                continue

            inputs = processor(
                audio,
                sampling_rate=config.MERT_SR,
                return_tensors="pt",
            ).to(device)
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[config.MERT_LAYER]
            emb = hidden.mean(dim=1).squeeze().cpu().numpy()
            embeddings.append(emb.astype(np.float32))

    embeddings = np.stack(embeddings)

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
        log.info(f"Embeddings cached → {cache_path}")

    return embeddings


# ── Cosine similarity ─────────────────────────────────────────────────────────


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between each row of a and each row of b."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a_norm @ b_norm.T


# ── KNN neighbor fraction scoring ─────────────────────────────────────────────


def score_knn_confusion(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """
    For each chunk, find its K nearest neighbors from ALL chunks (human+AI
    combined, excluding itself), and compute the fraction of neighbors
    from the opposite class.

    A human chunk with many AI neighbors is hard (looks like AI).
    An AI chunk with many human neighbors is hard (looks like human).

    Parameters
    ----------
    embeddings : (N, 768)
    labels     : (N,) — 0=human, 1=ai
    k          : number of nearest neighbors

    Returns
    -------
    confusion_fracs : (N,) — fraction of K neighbors from opposite class
    """
    n = len(embeddings)
    log.info(f"Computing KNN confusion fractions (k={k}, N={n})...")

    # Full similarity matrix (N, N)
    sim_matrix = _cosine_sim(embeddings, embeddings)

    # Set self-similarity to -inf so each chunk doesn't neighbor itself
    np.fill_diagonal(sim_matrix, -np.inf)

    # For each chunk: indices of K nearest neighbors
    # argsort ascending → take last K for highest similarity
    top_k_indices = np.argsort(sim_matrix, axis=1)[:, -k:]  # (N, K)

    # For each chunk: labels of its K neighbors
    neighbor_labels = labels[top_k_indices]  # (N, K)

    # Fraction of neighbors from the opposite class
    opposite_class = 1 - labels  # flip: human→1, ai→0
    # For human (label=0): opposite=1=AI, count AI neighbors
    # For AI    (label=1): opposite=0=human, count human neighbors
    confusion_fracs = np.array(
        [(neighbor_labels[i] == opposite_class[i]).mean() for i in range(n)],
        dtype=np.float32,
    )

    # Log summary
    human_mask = labels == 0
    ai_mask = labels == 1
    log.info(
        f"  Human confusion fracs — "
        f"mean={confusion_fracs[human_mask].mean():.3f} "
        f"median={np.median(confusion_fracs[human_mask]):.3f} "
        f"max={confusion_fracs[human_mask].max():.3f}"
    )
    log.info(
        f"  AI    confusion fracs — "
        f"mean={confusion_fracs[ai_mask].mean():.3f} "
        f"median={np.median(confusion_fracs[ai_mask]):.3f} "
        f"max={confusion_fracs[ai_mask].max():.3f}"
    )

    return confusion_fracs


# ── Main function ─────────────────────────────────────────────────────────────


def build_mined_chunk_index(
    chunk_df: pd.DataFrame,
    config,
    device: torch.device,
    subset_fraction: float = 0.25,
    hard_negative_fraction: float = 0.30,
    neighbor_confusion_threshold: float = 0.30,
    knn_k: int = 10,
    embeddings_cache_path: Path | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Build a mined training subset using KNN neighbor fraction scoring.

    Parameters
    ----------
    chunk_df                     : full chunk DataFrame (all splits)
    config                       : config module
    device                       : torch device for MERT
    subset_fraction              : target train size as fraction of full train
                                   e.g. 0.25 → target_n = 25% of train chunks
    hard_negative_fraction       : max fraction of target_n that are hard chunks
                                   e.g. 0.30 → at most 30% hard, 70% random
    neighbor_confusion_threshold : min fraction of opposite-class neighbors
                                   to qualify as a hard chunk
                                   e.g. 0.30 → need ≥3 of 10 neighbors
                                   from opposite class
    knn_k                        : number of nearest neighbors
    embeddings_cache_path        : path to cache/load MERT embeddings
    random_seed                  : reproducibility

    Returns
    -------
    result_df : chunk DataFrame — train split replaced by mined subset
                val/test unchanged
                columns added:
                  hard_type        : "hard_human" | "hard_ai" | "random"
                  confusion_frac   : fraction of opposite-class neighbors
                  is_hard_negative : bool (backwards compat)
    """
    rng = np.random.default_rng(random_seed)

    # ── Separate train from val/test ──────────────────────────────────────
    train_mask = chunk_df["split"] == "train"
    train_df = chunk_df[train_mask].reset_index(drop=True)
    other_df = chunk_df[~train_mask].copy()

    n_train = len(train_df)
    target_n = int(n_train * subset_fraction)
    max_hard = int(target_n * hard_negative_fraction)
    n_random = target_n - max_hard  # at least this many random chunks

    labels = (train_df["label"].astype(str) == "ai").astype(int).values
    human_mask = labels == 0
    ai_mask = labels == 1

    log.info(f"\n{'=' * 60}")
    log.info(f"MERT KNN hard negative mining")
    log.info(f"  Full train set              : {n_train}")
    log.info(
        f"  Target subset               : {target_n} ({subset_fraction * 100:.0f}%)"
    )
    log.info(
        f"  Max hard chunks             : {max_hard} "
        f"({hard_negative_fraction * 100:.0f}% of target)"
    )
    log.info(
        f"  Min random fill             : {n_random} "
        f"({(1 - hard_negative_fraction) * 100:.0f}% of target)"
    )
    log.info(
        f"  Neighbor confusion threshold: "
        f"{neighbor_confusion_threshold:.2f} "
        f"(≥{int(neighbor_confusion_threshold * knn_k)} of {knn_k} "
        f"neighbors from opposite class)"
    )
    log.info(f"{'=' * 60}")

    # ── Load embeddings ───────────────────────────────────────────────────
    embeddings = extract_mert_embeddings(
        chunk_df=train_df,
        config=config,
        device=device,
        cache_path=embeddings_cache_path,
    )

    # ── Score KNN confusion for all training chunks ────────────────────────
    confusion_fracs = score_knn_confusion(
        embeddings=embeddings,
        labels=labels,
        k=knn_k,
    )

    # ── Identify hard chunks above threshold ──────────────────────────────
    # Hard human: human chunks with ≥ threshold fraction of AI neighbors
    hard_human_mask = human_mask & (confusion_fracs >= neighbor_confusion_threshold)
    # Hard AI: AI chunks with ≥ threshold fraction of human neighbors
    hard_ai_mask = ai_mask & (confusion_fracs >= neighbor_confusion_threshold)

    hard_human_idx = np.where(hard_human_mask)[0]
    hard_ai_idx = np.where(hard_ai_mask)[0]

    log.info(
        f"\nHard chunks identified (threshold={neighbor_confusion_threshold:.2f}):"
    )
    log.info(
        f"  Hard human: {len(hard_human_idx):,} / "
        f"{human_mask.sum():,} "
        f"({100 * len(hard_human_idx) / human_mask.sum():.1f}%)"
    )
    log.info(
        f"  Hard AI   : {len(hard_ai_idx):,} / "
        f"{ai_mask.sum():,} "
        f"({100 * len(hard_ai_idx) / ai_mask.sum():.1f}%)"
    )

    # ── Sort by confusion fraction descending (most confused first) ────────
    hard_human_idx = hard_human_idx[np.argsort(confusion_fracs[hard_human_idx])[::-1]]
    hard_ai_idx = hard_ai_idx[np.argsort(confusion_fracs[hard_ai_idx])[::-1]]

    # ── Cap hard chunks at max_hard total, split evenly between classes ────
    # Split budget evenly; if one class has fewer than budget, give
    # the remainder to the other class
    half_budget = max_hard // 2
    n_hard_human = min(len(hard_human_idx), half_budget)
    n_hard_ai = min(len(hard_ai_idx), half_budget)

    # If one side is short, give unused budget to the other
    shortfall_h = half_budget - n_hard_human
    shortfall_a = half_budget - n_hard_ai
    n_hard_ai = min(len(hard_ai_idx), half_budget + shortfall_h)
    n_hard_human = min(len(hard_human_idx), half_budget + shortfall_a)

    # Take only the top-N most confused chunks
    selected_hard_human = hard_human_idx[:n_hard_human]
    selected_hard_ai = hard_ai_idx[:n_hard_ai]
    n_hard_actual = n_hard_human + n_hard_ai
    n_random_needed = target_n - n_hard_actual

    log.info(f"\nAfter capping at {max_hard} hard chunks:")
    log.info(
        f"  Selected hard human : {n_hard_human} "
        f"(min confusion: "
        f"{confusion_fracs[selected_hard_human].min():.3f})"
    )
    log.info(
        f"  Selected hard AI    : {n_hard_ai} "
        f"(min confusion: "
        f"{confusion_fracs[selected_hard_ai].min():.3f})"
    )
    log.info(f"  Random fill needed  : {n_random_needed}")

    # ── Random fill — balance full subset to 50/50 human/AI ───────────────
    hard_set = set(selected_hard_human.tolist()) | set(selected_hard_ai.tolist())

    # Count how many human and AI are already in hard set
    n_hard_human_selected = n_hard_human
    n_hard_ai_selected = n_hard_ai

    # Target: 50/50 overall → target_n/2 of each class
    target_human = target_n // 2
    target_ai = target_n - target_human

    n_random_human = max(0, target_human - n_hard_human_selected)
    n_random_ai = max(0, target_ai - n_hard_ai_selected)

    # Pool: all training chunks excluding already-selected hard chunks
    remaining_human = np.array(
        [i for i in np.where(human_mask)[0] if i not in hard_set]
    )
    remaining_ai = np.array([i for i in np.where(ai_mask)[0] if i not in hard_set])

    # Clip if pool smaller than needed
    n_random_human = min(n_random_human, len(remaining_human))
    n_random_ai = min(n_random_ai, len(remaining_ai))

    random_human_idx = rng.choice(remaining_human, size=n_random_human, replace=False)
    random_ai_idx = rng.choice(remaining_ai, size=n_random_ai, replace=False)

    log.info(f"\nRandom fill (balancing to 50/50):")
    log.info(f"  Random human: {n_random_human} (from pool of {len(remaining_human)})")
    log.info(f"  Random AI   : {n_random_ai} (from pool of {len(remaining_ai)})")

    # ── Build DataFrames ──────────────────────────────────────────────────
    def _make_subset(indices, hard_type):
        df = train_df.iloc[indices].copy()
        df["hard_type"] = hard_type
        df["confusion_frac"] = confusion_fracs[indices]
        df["is_hard_negative"] = hard_type != "random"
        return df

    hard_human_df = _make_subset(selected_hard_human, "hard_human")
    hard_ai_df = _make_subset(selected_hard_ai, "hard_ai")
    random_h_df = _make_subset(random_human_idx, "random")
    random_a_df = _make_subset(random_ai_idx, "random")

    mined_train_df = (
        pd.concat(
            [hard_human_df, hard_ai_df, random_h_df, random_a_df],
            ignore_index=True,
        )
        .sample(frac=1, random_state=random_seed)
        .reset_index(drop=True)
    )

    # ── Summary ───────────────────────────────────────────────────────────
    final_human = (mined_train_df["label"].astype(str) == "human").sum()
    final_ai = (mined_train_df["label"].astype(str) == "ai").sum()

    log.info(f"\n{'=' * 60}")
    log.info(f"Final mined subset:")
    log.info(f"  Total chunks  : {len(mined_train_df)}")
    log.info(
        f"  Human         : {final_human} "
        f"({100 * final_human / len(mined_train_df):.1f}%)"
    )
    log.info(
        f"  AI            : {final_ai} ({100 * final_ai / len(mined_train_df):.1f}%)"
    )
    log.info(f"  Hard human    : {(mined_train_df['hard_type'] == 'hard_human').sum()}")
    log.info(f"  Hard AI       : {(mined_train_df['hard_type'] == 'hard_ai').sum()}")
    log.info(f"  Random        : {(mined_train_df['hard_type'] == 'random').sum()}")
    log.info(f"\nConfusion frac stats (hard chunks only):")
    hard_only = mined_train_df[mined_train_df["is_hard_negative"]]
    log.info(
        f"  min={hard_only['confusion_frac'].min():.3f} "
        f"mean={hard_only['confusion_frac'].mean():.3f} "
        f"max={hard_only['confusion_frac'].max():.3f}"
    )
    log.info(f"{'=' * 60}")

    print("\nLabel × source × hard_type breakdown:")
    print(
        mined_train_df.groupby(["label", "source_name", "hard_type"])
        .size()
        .rename("chunks")
        .to_string()
    )

    # ── Combine with val/test ─────────────────────────────────────────────
    other_df["hard_type"] = "random"
    other_df["confusion_frac"] = 0.0
    other_df["is_hard_negative"] = False

    return pd.concat([mined_train_df, other_df], ignore_index=True)
