# 🎵 AI Music Forensics: Detecting Synthetic Audio with Temporal Intelligence

This repository contains a full end-to-end pipeline for detecting AI-generated music (Udio, Suno) versus human-generated audio (FMA). Unlike standard classifiers, this project focuses on **generalizability** by using **MERT-based hard-negative mining** to ignore confounding factors like genre or recording quality.

---

## 🚀 Project Overview

The goal is to distinguish between organic human compositions and high-fidelity AI generations. By treating audio as a sequence of **1-second segments within 10-second chunks**, I utilize a **Convolutional Recurrent Neural Network (CRNN)** to capture both spectral *"fingerprints"* and temporal *"narratives."*

### 🔑 Key Innovations

- **MERT-Based Mining**  
  Uses self-supervised music embeddings to identify *Hard Negatives* (human music that sounds like AI and vice versa), forcing the model to learn deeper structural differences.

- **30/70 Strategy**  
  A training mix of **30% hard-mined samples** and **70% random samples** to balance complex boundary learning with general robustness.

- **Track-Level Soft Voting**  
  Aggregates chunk-level predictions to produce a stable, high-confidence classification for full tracks.

---

## 📂 Repository Structure

### 📓 Notebooks (Workflow)

Run the notebooks in the following order:

- **01_Data_Collection.ipynb**  
  Downloads datasets from Hugging Face, filters (Udio/Suno), and organizes raw data.

- **02_EDA.ipynb**  
  Performs exploratory data analysis, included analysis of the MERT embeddings, and generates the usable metadata manifest.

- **03_Models_Experimentations.ipynb**  
  Core experimentation suite including:
  - MLP
  - CNN
  - CNN with Hard Negative Mining
  - CNN-GRU (CRNN)

  Includes training, evaluation, and interpretability (Integrated Gradients).

---

### 🛠 Source Code (`src/`)

- **config.py**  
  Centralized configuration (hyperparameters, paths, settings).

- **dataset.py & sampling.py**  
  Audio chunking logic and PyTorch Dataset/DataLoader definitions.

- **mert_mining.py**  
  Hard-negative mining using KNN on MERT embeddings.

- **spectrogram.py**  
  Efficient Mel-spectrogram generation (GPU/CPU).

- **train.py & evaluate.py**  
  Training loops and evaluation metrics.

- **results.py & splits.py**  
  Experiment tracking, JSON logging, and reproducible dataset splits.

---

## 🛠 Installation & Setup

### 1. Requirements

This project uses **uv** (recommended) or standard pip:

```bash
# Clone the repository
git clone https://github.com/NikolozDarsalia/AI_Music_Detection.git
cd AI_Music_Detection

# Install dependencies
uv sync

```

### 2. Authentication
To access datasets from Hugging Face:
- Create a folder named tokens/ in the root directory
- Inside it, create a file (e.g., token_hf.txt)
- Paste your Hugging Face Read Token into the file

## 🔄 Data Pipeline
Running the notebooks will automatically generate the following structure:

music_data/
  ├── *.wav
  └── metadata_usable.csv

data/
  ├── chunks/          # Chunk-level CSVs + MERT embeddings
  ├── spectrograms/    # Precomputed .npy files
  ├── splits/          # Train/val/test splits
  ├── checkpoints/     # Model weights
  ├── results/         # JSON metrics
  └── logs/            # Plots, confusion matrices, attribution maps


## 🧠 Experimental Results
I evaluated four architectures on a 6,000-chunk subset to quickly identify the most promising approach:

| Model     | Strategy            | Focus                                  |
| --------- | ------------------- | -------------------------------------- |
| MLP       | Baseline            | Raw spectral feature mapping           |
| CNN       | Random Sample       | Local 2D pattern recognition           |
| CNN-Mined | 30/70 Hard Negative | Reduced confounding bias               |
| CNN-GRU   | 30/70 + Temporal    | ✅ Best model (CNN + sequence modeling) |

## ⚖️ License
This project is licensed under the MIT License – see the LICENSE file for details.