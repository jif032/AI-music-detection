# =============================================================================
# splits.py
# =============================================================================
# Generates track-level train/val/test splits under three strategies.
# Each strategy saves a CSV with columns:
#   file_path, label, source_name, duration_sec, split
#
# Guarantees:
#   - No track appears in more than one split
#   - Suno tracks are distributed according to strategy
#   - Stratification preserves source proportions within splits
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)


def _save_split(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info(f"Saved split → {path}  ({len(df)} tracks)")


def _log_split_summary(df: pd.DataFrame, name: str) -> None:
    log.info(f"\n{'=' * 55}")
    log.info(f"Split summary: {name}")
    log.info(f"{'=' * 55}")
    summary = (
        df.groupby(["split", "label", "source_name"])
        .size()
        .rename("tracks")
        .reset_index()
    )
    log.info(f"\n{summary.to_string(index=False)}")
    totals = df.groupby("split").size().rename("total")
    log.info(f"\nTotals per split:\n{totals.to_string()}")


def make_random_split(
    meta: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    output_path: Path,
) -> pd.DataFrame:
    """
    Stratified random split.
    Suno tracks appear in all three splits proportionally.
    Stratification key: source_name (preserves FMA/Udio/Suno ratios).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Ratios must sum to 1.0"
    )

    log.info("Building random stratified split...")

    # Stratify by source_name so Suno is proportionally in all splits
    strat_col = meta["source_name"].values

    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        meta,
        test_size=(val_ratio + test_ratio),
        stratify=strat_col,
        random_state=seed,
    )

    # Second split: val vs test (from temp)
    val_size_adjusted = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_size_adjusted),
        stratify=temp_df["source_name"].values,
        random_state=seed,
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    result = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)

    # Sanity check — no track in multiple splits
    assert result["file_path"].nunique() == len(result), (
        "Duplicate tracks detected across splits!"
    )

    _log_split_summary(result, "random")
    _save_split(result, output_path)
    return result


def make_cross_source_split(
    meta: pd.DataFrame,
    train_source: str,  # "Udio" or "Suno"
    test_source: str,  # "Suno" or "Udio"
    val_ratio_of_train: float,  # fraction of train set held out for val
    train_ratio: float,  # fraction of FMA used in train
    seed: int,
    output_path: Path,
) -> pd.DataFrame:
    """
    Cross-source generalization split.

    Train : FMA (human subset) + train_source (AI)
    Val   : FMA (human subset) + train_source (AI) — held out from train
    Test  : FMA (human subset) + test_source  (AI) — completely unseen AI source

    The FMA tracks are split into three non-overlapping subsets:
      - train_fma  → paired with train_source AI tracks in train/val
      - test_fma   → paired with test_source AI tracks in test

    This ensures the model never sees test FMA tracks during training either.
    """
    log.info(f"Building cross-source split: train={train_source}, test={test_source}")

    fma = meta[meta["source_name"] == "FMA"].copy()
    train_ai = meta[meta["source_name"] == train_source].copy()
    test_ai = meta[meta["source_name"] == test_source].copy()

    log.info(f"FMA tracks        : {len(fma)}")
    log.info(f"Train AI ({train_source}): {len(train_ai)}")
    log.info(f"Test  AI ({test_source}) : {len(test_ai)}")

    # Split FMA into train/val portion and test portion
    # test portion of FMA is sized to roughly match test_ai count
    fma_test_ratio = min(
        len(test_ai) / len(fma),
        1.0 - val_ratio_of_train - 0.1,  # keep at least 10% for train
    )
    fma_test_ratio = max(fma_test_ratio, 0.15)  # at least 15% for test

    fma_trainval, fma_test = train_test_split(
        fma,
        test_size=fma_test_ratio,
        random_state=seed,
    )

    # Split train AI into train/val
    ai_train, ai_val = train_test_split(
        train_ai,
        test_size=val_ratio_of_train,
        random_state=seed,
    )

    # Split FMA trainval to match
    fma_train, fma_val = train_test_split(
        fma_trainval,
        test_size=val_ratio_of_train,
        random_state=seed,
    )

    # Assign splits
    fma_train["split"] = "train"
    fma_val["split"] = "val"
    fma_test["split"] = "test"
    ai_train["split"] = "train"
    ai_val["split"] = "val"
    test_ai["split"] = "test"  # all test_source goes to test

    result = pd.concat(
        [
            fma_train,
            fma_val,
            fma_test,
            ai_train,
            ai_val,
            test_ai,
        ]
    ).reset_index(drop=True)

    # Sanity checks
    assert result["file_path"].nunique() == len(result), (
        "Duplicate tracks detected across splits!"
    )

    # Verify test source only appears in test
    test_source_in_train = result[
        (result["source_name"] == test_source)
        & (result["split"].isin(["train", "val"]))
    ]
    assert len(test_source_in_train) == 0, (
        f"{test_source} tracks found in train/val — leakage detected!"
    )

    _log_split_summary(result, f"{train_source}_train_{test_source}_test")
    _save_split(result, output_path)
    return result


def generate_all_splits(config) -> dict:
    """
    Entry point called from the notebook.
    Generates all three split CSVs based on config.SPLIT_STRATEGY
    or generates all three unconditionally if called with strategy='all'.

    Returns dict of {strategy_name: DataFrame}.
    """
    log.info("Loading metadata...")
    meta = pd.read_csv(config.META_PATH)
    log.info(f"Loaded {len(meta)} tracks from {config.META_PATH}")

    # Verify required columns exist
    required_cols = ["file_path", "label", "source_name", "duration_sec"]
    missing = [c for c in required_cols if c not in meta.columns]
    assert not missing, f"Missing columns in metadata: {missing}"

    log.info(f"\nSource distribution:")
    log.info(meta.groupby(["label", "source_name"]).size().rename("tracks").to_string())

    splits = {}

    # ── Random split ──────────────────────────────────────────────────────
    random_path = config.SPLITS_DIR / "random_split.csv"
    splits["random"] = make_random_split(
        meta=meta,
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO,
        seed=config.SEED,
        output_path=random_path,
    )

    # ── Udio train → Suno test ────────────────────────────────────────────
    udio_path = config.SPLITS_DIR / "udio_train_split.csv"
    splits["udio_train"] = make_cross_source_split(
        meta=meta,
        train_source="Udio",
        test_source="Suno",
        val_ratio_of_train=config.VAL_RATIO,
        train_ratio=config.TRAIN_RATIO,
        seed=config.SEED,
        output_path=udio_path,
    )

    # ── Suno train → Udio test ────────────────────────────────────────────
    suno_path = config.SPLITS_DIR / "suno_train_split.csv"
    splits["suno_train"] = make_cross_source_split(
        meta=meta,
        train_source="Suno",
        test_source="Udio",
        val_ratio_of_train=config.VAL_RATIO,
        train_ratio=config.TRAIN_RATIO,
        seed=config.SEED,
        output_path=suno_path,
    )

    log.info("\nAll splits generated successfully.")
    return splits
