# =============================================================================
# config.py
# =============================================================================
# Fixed infrastructure configuration only.
#
# What lives here:
#   - All file paths
#   - Reproducibility seed
#   - Train/val/test split ratios
#   - Chunk sampling parameters (counts, probabilities, duration)
#   - Audio and mel spectrogram parameters
#   - Dataset balancing flag
#
# What does NOT live here (defined in the notebook):
#   - Training hyperparameters (LR, batch size, epochs, etc.)
#   - Data augmentation parameters
#   - MERT mining related parameters
# =============================================================================

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "music_data"
META_PATH = DATA_DIR / "metadata_usable.csv"
SPLITS_DIR = ROOT_DIR / "data" / "splits"
CHUNKS_DIR = ROOT_DIR / "data" / "chunks"
SPECTROGRAMS_DIR = ROOT_DIR / "data" / "spectrograms"
LOGS_DIR = ROOT_DIR / "logs"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
RESULTS_DIR = ROOT_DIR / "results"

for _dir in [
    SPLITS_DIR,
    CHUNKS_DIR,
    SPECTROGRAMS_DIR,
    LOGS_DIR,
    CHECKPOINTS_DIR,
    RESULTS_DIR,
]:
    _dir.mkdir(parents=True, exist_ok=True)

# Alias used throughout src/ modules
CHECKPOINT_DIR = CHECKPOINTS_DIR

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── Split strategy ────────────────────────────────────────────────────────────
# Options: "random" | "udio_train" | "suno_train"
# Set in the notebook before loading a manifest — not used by splits.py directly
SPLIT_STRATEGY = "random"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ── Chunk sampling — training set ─────────────────────────────────────────────
# Fewer chunks per track during training to reduce within-track redundancy
N_CHUNKS_FMA_TRAIN = 12
N_CHUNKS_UDIO_TRAIN = 5
N_CHUNKS_SUNO_TRAIN = 5

# ── Chunk sampling — val and test sets ───────────────────────────────────────
# More chunks per track for evaluation → better track-level aggregation
N_CHUNKS_FMA_EVAL = 20
N_CHUNKS_UDIO_EVAL = 8
N_CHUNKS_SUNO_EVAL = 8

# ── Chunk sampling — shared parameters ───────────────────────────────────────
CHUNK_SEC = 10  # seconds per chunk
SAMPLE_RATE = 22050  # Hz — stored wav files

# Weighted percentile sampling: [0-10%, 10-25%, 25-75%, 75-90%, 90-100%]
SAMPLING_REGION_BOUNDS = [0.0, 0.10, 0.25, 0.75, 0.90, 1.0]
SAMPLING_REGION_PROBS = [0.02, 0.13, 0.70, 0.13, 0.02]

# Minimum track duration to be eligible for any chunk sampling
MIN_TRACK_DURATION = CHUNK_SEC + 2.0  # 12 seconds

# ── Audio preprocessing ───────────────────────────────────────────────────────
RMS_TARGET = 0.1  # target RMS for normalization before spectrogram generation

# ── Mel spectrogram ───────────────────────────────────────────────────────────
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
F_MAX = 8000  # Hz — confirmed during EDA

# Expected time frames for a CHUNK_SEC chunk at SAMPLE_RATE with HOP_LENGTH
SPEC_TIME_FRAMES = 431
SPEC_SHAPE = (N_MELS, SPEC_TIME_FRAMES)  # (128, 431)

# ── MERT hard-negative mining ─────────────────────────────────────────────────
MERT_MODEL = "m-a-p/MERT-v1-95M"
MERT_LAYER = 6
MERT_SR = 24000

# ── Dataset balancing ─────────────────────────────────────────────────────────
# Cap AI chunks to match human count in training split
USE_WEIGHTED_SAMPLER = True
