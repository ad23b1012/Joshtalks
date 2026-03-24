"""
Central configuration for the Josh Talks ASR assignment.
All paths, URLs, model settings, and hyperparameters live here.
"""
import os
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_AUDIO_DIR = DATA_DIR / "raw_audio"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "whisper-small-hi-finetuned"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Create directories
for d in [DATA_DIR, RAW_AUDIO_DIR, PROCESSED_DIR, CACHE_DIR, OUTPUT_DIR, MODEL_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Dataset URLs ────────────────────────────────────────────────────
DATASET_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1bujiO2NgtHlgqPlNvYAQf5_7ZcXARlIfNX5HNb9f8cI/export?format=csv"
)
WER_REPORT_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1JItJnilmmSWjx9tAIr06cbTsyGjMMxEMhaebvn5qBHM/export?format=csv"
)
UNIQUE_WORDS_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "17DwCAx6Tym5Nt7eOni848np9meR-TIj7uULMtYcgQaw/export?format=csv"
)
LATTICE_DATA_CSV_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1J_I0raoRNbe29HiAPD5FROTr0jC93YtSkjOrIglKEjU/export?format=csv"
)

# ─── URL Rewriting ───────────────────────────────────────────────────
# Original: https://storage.googleapis.com/joshtalks-data-collection/hq_data/hi/{uid}/{rid}_*.{ext}
# Rewrite:  https://storage.googleapis.com/upload_goai/{uid}/{rid}_*.{ext}
ORIGINAL_BUCKET = "joshtalks-data-collection/hq_data/hi"
REWRITE_BUCKET = "upload_goai"

def rewrite_gcs_url(url: str) -> str:
    """Rewrite GCS URLs from original bucket to the accessible upload_goai bucket."""
    return url.replace(ORIGINAL_BUCKET, REWRITE_BUCKET)

# ─── Model Configuration ────────────────────────────────────────────
MODEL_NAME = "openai/whisper-small"
LANGUAGE = "hi"
TASK = "transcribe"
SAMPLE_RATE = 16000
MAX_INPUT_LENGTH = 30.0  # seconds — Whisper's max context window

# ─── Training Hyperparameters (optimized for RTX 4060 8GB) ──────────
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2  # effective batch = 16
LEARNING_RATE = 1e-5
WARMUP_STEPS = 500
MAX_STEPS = 5000
SAVE_STEPS = 500
EVAL_STEPS = 500
LOGGING_STEPS = 50
FP16 = True
GRADIENT_CHECKPOINTING = True

# ─── FLEURS Evaluation ──────────────────────────────────────────────
FLEURS_DATASET = "google/fleurs"
FLEURS_LANGUAGE = "hi_in"
FLEURS_SPLIT = "test"

# ─── Error Analysis ─────────────────────────────────────────────────
ERROR_SAMPLE_SIZE = 25
ERROR_SAMPLE_STRATEGY = "systematic"  # every Nth from top-100 worst
ERROR_TOP_N = 100
ERROR_STEP = 4  # 100 / 25 = every 4th
