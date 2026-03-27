# =============================================================================
# results.py
# =============================================================================
# Accumulates experiment results into a comparison DataFrame.
# Saves/loads from JSON so results persist across notebook sessions.
# =============================================================================

import pandas as pd
import json
from pathlib import Path
import logging

log = logging.getLogger(__name__)

RESULTS_FILE = "results/all_results.json"


def load_results(path: str | Path | None = None) -> pd.DataFrame:
    """Load existing results from JSON. Returns empty DataFrame if not found."""
    p = Path(path or RESULTS_FILE)
    if not p.exists():
        log.info(f"No results file at {p} — starting fresh.")
        return pd.DataFrame()
    with open(p) as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)
    log.info(f"Loaded {len(df)} results from {p}")
    return df


def save_results(df: pd.DataFrame, path: str | Path | None = None) -> None:
    """Save results DataFrame to JSON."""
    p = Path(path or RESULTS_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)
    log.info(f"Saved {len(df)} results → {p}")


def add_result(
    results_df: pd.DataFrame,
    row: dict,
    path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Append one result row to the DataFrame and save.
    If a row with the same run_name exists, it is replaced.
    """
    new_row = pd.DataFrame([row])

    if len(results_df) > 0 and "run_name" in results_df.columns:
        # Remove existing row with same run_name if present
        results_df = results_df[results_df["run_name"] != row.get("run_name", "")]

    updated = pd.concat([results_df, new_row], ignore_index=True)
    save_results(updated, path)
    return updated


def show_results(
    results_df: pd.DataFrame,
    sort_by: str = "track_auroc",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Display a clean comparison table sorted by a metric.
    Returns the sorted DataFrame for further filtering.
    """
    if len(results_df) == 0:
        print("No results yet.")
        return results_df

    display_cols = [
        "run_name",
        "model_type",
        "split_strategy",
        "chunk_auroc",
        "chunk_auprc_ai",
        "chunk_auprc_hum",
        "track_auroc",
        "track_auprc_ai",
        "fma_chunk_auroc",
        "suno_chunk_auroc",
        "udio_chunk_auroc",
    ]
    # Only show columns that exist
    cols = [c for c in display_cols if c in results_df.columns]

    sorted_df = (
        results_df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
        if sort_by in results_df.columns
        else results_df
    )

    try:
        from IPython.display import display

        display(
            sorted_df[cols]
            .style.format(
                {
                    c: "{:.4f}"
                    for c in cols
                    if c not in ["run_name", "model_type", "split_strategy"]
                }
            )
            .background_gradient(
                subset=["track_auroc"] if "track_auroc" in cols else [],
                cmap="RdYlGn",
                vmin=0.5,
                vmax=1.0,
            )
        )
    except Exception:
        print(sorted_df[cols].to_string(index=False))

    return sorted_df


def compare_runs(
    results_df: pd.DataFrame,
    run_names: list[str],
) -> pd.DataFrame:
    """Filter results to specific run names for direct comparison."""
    mask = results_df["run_name"].isin(run_names)
    return show_results(results_df[mask])
