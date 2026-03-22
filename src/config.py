# =============================================================================
# config.py
# =============================================================================
# Single source of truth for all pipeline parameters.
# Change values here — nowhere else.
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

for _dir in [SPLITS_DIR, CHUNKS_DIR, SPECTROGRAMS_DIR, LOGS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 28954

# ── Split strategy ────────────────────────────────────────────────────────────
# Options:
#   "random"      — stratified random split (Suno present in all sets)
#   "udio_train"  — train on FMA+Udio, val on FMA+Udio subset, test on FMA+Suno
#   "suno_train"  — train on FMA+Suno, val on FMA+Suno subset, test on FMA+Udio
SPLIT_STRATEGY = "random"

# Proportions — must sum to 1.0
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ── Chunk sampling ────────────────────────────────────────────────────────────
# Number of chunks to sample per track per source
N_CHUNKS_FMA = 12  # human — longer and more diverse tracks
N_CHUNKS_UDIO = 5  # ai
N_CHUNKS_SUNO = 5  # ai

CHUNK_SEC = 10  # seconds per chunk
SAMPLE_RATE = 22050  # Hz — our stored wav files

# Weighted percentile sampling probabilities
# Five regions: [0-10%, 10-25%, 25-75%, 75-90%, 90-100%]
# Must sum to 1.0
SAMPLING_REGION_BOUNDS = [0.0, 0.10, 0.25, 0.75, 0.90, 1.0]
SAMPLING_REGION_PROBS = [0.02, 0.13, 0.70, 0.13, 0.02]

# Minimum track duration to be eligible for sampling (seconds)
# Must accommodate at least one full chunk plus safety margin
MIN_TRACK_DURATION = CHUNK_SEC + 7.0  # 12 seconds

# ── Hard-negative mining ──────────────────────────────────────────────────────
# Set to False to skip MERT+KNN entirely and use random chunks only
USE_HARD_NEGATIVES = False

# Fraction of training chunks replaced by hard negatives
# e.g. 0.30 → 70% random + 30% hard negatives
HARD_NEGATIVE_FRACTION = 0.30

# KNN parameters
KNN_K = 10  # number of neighbors to consider
# Confusion threshold: fraction of K neighbors from opposite class
# e.g. 0.30 → at least 3/10 neighbors must be from opposite class
KNN_CONFUSION_THRESHOLD = 0.30

# MERT model for embedding extraction
MERT_MODEL = "m-a-p/MERT-v1-95M"
MERT_LAYER = 6  # layer to extract embeddings from
MERT_SR = 24000  # Hz — MERT's expected sample rate

# ── Mel spectrogram ───────────────────────────────────────────────────────────
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
F_MAX = 8000  # Hz — confirmed during EDA
# Target RMS for normalization before spectrogram generation
RMS_TARGET = 0.1

# Expected spectrogram shape — used for validation and model input size
# time_frames = ceil(CHUNK_SEC * SAMPLE_RATE / HOP_LENGTH)
SPEC_TIME_FRAMES = 431  # at 22050 Hz, 10s, hop=512
SPEC_SHAPE = (N_MELS, SPEC_TIME_FRAMES)  # (128, 431)

# ── Data augmentation (SpecAugment) ───────────────────────────────────────────
# Applied only during training, not val/test
USE_AUGMENTATION = True
FREQ_MASK_PARAM = 15  # max mel bins to mask
TIME_MASK_PARAM = 50  # max time frames to mask
N_FREQ_MASKS = 2
N_TIME_MASKS = 2

# ── Dataset balancing ─────────────────────────────────────────────────────────
# WeightedRandomSampler — handles remaining imbalance after chunk capping
USE_WEIGHTED_SAMPLER = True

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

# Scheduler
LR_PATIENCE = 7  # was 5 — give more time before reducing
LR_FACTOR = 0.5
LR_MIN = 1e-6

# Early stopping
EARLY_STOP_PATIENCE = 15  # was 10 — more patience


# Model checkpoint
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ── Model architecture ────────────────────────────────────────────────────────
# Options: "cnn", "cnn_lstm", "cnn_gru", "ast"
MODEL_TYPE = "cnn"

CNN_CHANNELS = [8, 16, 32, 64]  # ~26K params, ratio ≈ 6x
CNN_DROPOUT = 0.5

RNN_HIDDEN_SIZE = 64  # was 128
RNN_NUM_LAYERS = 1  # was 2
RNN_DROPOUT = 0.0  # must be 0 when num_layers=1


# RNN head (used when MODEL_TYPE is cnn_lstm or cnn_gru)
RNN_HIDDEN_SIZE = 128
RNN_NUM_LAYERS = 2
RNN_DROPOUT = 0.3

# ── Evaluation ────────────────────────────────────────────────────────────────
# Track-level aggregation: "mean" (soft voting on chunk probabilities)
TRACK_AGG_METHOD = "mean"
# Metrics computed at both chunk and track level
EVAL_METRICS = ["auroc", "auprc", "accuracy", "f1"]
