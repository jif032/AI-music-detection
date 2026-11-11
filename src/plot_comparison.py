"""Create notebook-style figures from saved model histories and test metrics.

The palette, line widths, subplot sizing, grid, and legend sizes follow the
existing comparison and analysis cells in 02_EDA.ipynb and
03_Models_Experimentations.ipynb. This script does not train or evaluate models.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ai_music_detection_mpl")
)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_ORDER = ["cnn_bigru", "cnn_bilstm", "cnn_bilstm_attention"]
MODEL_LABELS = {
    "cnn_bigru": "CNN-BiGRU",
    "cnn_bilstm": "CNN-BiLSTM",
    "cnn_bilstm_attention": "CNN-BiLSTM + Attention",
}
# Blue and orange are the notebook's Human/AI colors; green is its established
# source-comparison color. These provide stable model identities in the curves.
MODEL_COLORS = {
    "cnn_bigru": "#4C72B0",
    "cnn_bilstm": "#DD8452",
    "cnn_bilstm_attention": "#55A868",
}
RUN_NAMES = {
    "cnn_bigru": "cnn_bigru_mert_random",
    "cnn_bilstm": "cnn_bilstm_mert_random",
    "cnn_bilstm_attention": "cnn_bilstm_attention_mert_random",
}


def _load_histories(strategy: str) -> dict[str, pd.DataFrame]:
    histories = {}
    for model in MODEL_ORDER:
        path = ROOT / "logs" / model / strategy / f"{RUN_NAMES[model]}_history.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing saved training history: {path}")
        history = pd.read_csv(path)
        required = {"epoch", "split", "loss", "chunk_accuracy", "chunk_auroc", "chunk_auprc", "chunk_f1"}
        missing = required.difference(history.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        histories[model] = history
    return histories


def _plot_training_curves(histories: dict[str, pd.DataFrame], output: Path) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    metrics = [
        ("loss", "Loss", None),
        ("chunk_auroc", "Chunk AUROC", (0, 1)),
        ("chunk_auprc", "Chunk AUPRC (AI)", (0, 1)),
        ("chunk_accuracy", "Chunk Accuracy", (0, 1)),
        ("chunk_f1", "Chunk F1 (AI)", (0, 1)),
    ]

    for ax, (column, title, limits) in zip(axes, metrics):
        for model in MODEL_ORDER:
            history = histories[model]
            for split, line_style in (("train", "-"), ("val", "--")):
                subset = history[history["split"] == split]
                ax.plot(
                    subset["epoch"],
                    subset[column],
                    color=MODEL_COLORS[model],
                    linewidth=1.5,
                    linestyle=line_style,
                    label=f"{MODEL_LABELS[model]} ({split})",
                )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        if limits:
            ax.set_ylim(*limits)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)

    axes[-1].set_visible(False)
    fig.suptitle("Training and Validation Curves", fontsize=12)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_test_metrics(comparison: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    columns = [
        ("chunk_auroc", "track_auroc", "AUROC"),
        ("chunk_auprc_ai", "track_auprc_ai", "AUPRC (AI)"),
        ("chunk_accuracy", "track_accuracy", "Accuracy"),
        ("chunk_f1_ai", "track_f1_ai", "F1 (AI)"),
    ]
    x = range(len(MODEL_ORDER))
    labels = ["CNN-BiGRU", "CNN-BiLSTM", "BiLSTM + Attention"]

    for ax, (chunk_column, track_column, title) in zip(axes, columns):
        chunk_values = [comparison.loc[model, chunk_column] for model in MODEL_ORDER]
        track_values = [comparison.loc[model, track_column] for model in MODEL_ORDER]
        ax.plot(
            x,
            chunk_values,
            color="#4C72B0",
            linewidth=1.5,
            marker="o",
            label="Chunk-level",
        )
        ax.plot(
            x,
            track_values,
            color="#DD8452",
            linewidth=1.5,
            linestyle="--",
            marker="s",
            label="Track-level",
        )
        for position, value in enumerate(chunk_values):
            ax.annotate(f"{value:.3f}", (position, value), xytext=(0, -12),
                        textcoords="offset points", ha="center", fontsize=8)
        for position, value in enumerate(track_values):
            ax.annotate(f"{value:.3f}", (position, value), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=8)
        ax.set_xticks(list(x), labels)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Random-Split Test Metrics", fontsize=12)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="random")
    args = parser.parse_args()

    histories = _load_histories(args.strategy)
    comparison_path = ROOT / "results" / "architecture_comparison" / f"{args.strategy}.csv"
    comparison_raw = pd.read_csv(comparison_path)
    comparison_raw["model_type"] = comparison_raw["model_type"].astype(str)
    comparison = comparison_raw.set_index("model_type").loc[MODEL_ORDER]

    output_dir = ROOT / "results" / "architecture_comparison" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    curves_path = output_dir / f"{args.strategy}_training_validation_curves.png"
    test_path = output_dir / f"{args.strategy}_test_metrics.png"
    _plot_training_curves(histories, curves_path)
    _plot_test_metrics(comparison, test_path)
    print(f"Saved {curves_path}")
    print(f"Saved {test_path}")


if __name__ == "__main__":
    main()
