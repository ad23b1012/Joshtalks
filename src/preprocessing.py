"""
Preprocessing — Audio segmentation, text cleaning, and HuggingFace Dataset creation.

Handles:
  1. Slicing long audio files into utterance-level segments using torchaudio.
  2. Text normalization (Unicode NFC, punctuation removal, etc.)
  3. Building a HuggingFace Dataset with Audio features for Whisper.
"""
import re
import unicodedata
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torchaudio
import numpy as np
import pandas as pd
from datasets import Dataset, Audio, Features, Value
from transformers import WhisperProcessor

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    SAMPLE_RATE, MAX_INPUT_LENGTH, MODEL_NAME, LANGUAGE, TASK,
    PROCESSED_DIR, CACHE_DIR
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ─── Text Cleaning ───────────────────────────────────────────────────────────

# Punctuation regex for Hindi text cleaning
# Keeps Devanagari characters, digits, and spaces
PUNCTUATION_REGEX = re.compile(r'[^\u0900-\u097F\u0964\u0965\sa-zA-Z0-9]')
MULTI_SPACE_REGEX = re.compile(r'\s+')

def clean_text(text: str) -> str:
    """
    Clean Hindi transcription text for ASR training.
    
    Steps:
        1. Unicode NFC normalization (canonical decomposition + recomposition)
        2. Remove non-Devanagari punctuation (keep ।, ॥)
        3. Collapse multiple spaces
        4. Strip and lowercase any English portions
        
    Args:
        text: Raw Hindi transcription text.
        
    Returns:
        Cleaned text suitable for Whisper tokenization.
    """
    # Unicode normalization
    text = unicodedata.normalize('NFC', text)
    
    # Remove special punctuation marks but keep Devanagari text
    text = PUNCTUATION_REGEX.sub(' ', text)
    
    # Collapse whitespace
    text = MULTI_SPACE_REGEX.sub(' ', text).strip()
    
    return text


# ─── Audio Segmentation ─────────────────────────────────────────────────────

def extract_audio_segment(
    audio_path: str,
    start_sec: float,
    end_sec: float,
    target_sr: int = SAMPLE_RATE
) -> Optional[np.ndarray]:
    """
    Extract a segment from a WAV file and resample to target sample rate.
    
    Args:
        audio_path: Path to the full recording WAV.
        start_sec: Start time in seconds.
        end_sec: End time in seconds.
        target_sr: Target sample rate (16kHz for Whisper).
        
    Returns:
        1-D numpy array of audio samples, or None on failure.
    """
    try:
        info = torchaudio.info(audio_path)
        original_sr = info.sample_rate
        
        # Calculate frame offsets
        start_frame = int(start_sec * original_sr)
        num_frames = int((end_sec - start_sec) * original_sr)
        
        # Load only the segment we need (memory efficient)
        waveform, sr = torchaudio.load(
            audio_path,
            frame_offset=start_frame,
            num_frames=num_frames
        )
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if necessary
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=target_sr
            )
            waveform = resampler(waveform)
        
        return waveform.squeeze().numpy()
    
    except Exception as e:
        logger.warning(f"Failed to extract segment from {audio_path} [{start_sec}-{end_sec}]: {e}")
        return None


# ─── Dataset Builder ─────────────────────────────────────────────────────────

def build_hf_dataset(manifest: pd.DataFrame) -> Dataset:
    """
    Convert the segment manifest into a HuggingFace Dataset
    with audio arrays and cleaned text labels.
    
    This function:
      1. Extracts audio segments from full recordings.
      2. Cleans transcription text.
      3. Saves as a HF Dataset with Audio feature.
      
    Args:
        manifest: DataFrame with columns:
            audio_path, start, end, text, recording_id, speaker_id
            
    Returns:
        HuggingFace Dataset ready for Whisper preprocessing.
    """
    dataset_path = PROCESSED_DIR / "hf_dataset"
    
    if dataset_path.exists():
        logger.info(f"Loading cached HF dataset from {dataset_path}")
        return Dataset.load_from_disk(str(dataset_path))
    
    logger.info("Building HuggingFace dataset from segment manifest...")
    
    records = []
    skipped = 0
    
    for idx, row in manifest.iterrows():
        audio = extract_audio_segment(row['audio_path'], row['start'], row['end'])
        
        if audio is None or len(audio) < SAMPLE_RATE * 0.5:  # Skip < 0.5s
            skipped += 1
            continue
        
        # Skip segments exceeding Whisper's 30s limit
        if len(audio) / SAMPLE_RATE > MAX_INPUT_LENGTH:
            skipped += 1
            continue
        
        text = clean_text(row['text'])
        if not text:
            skipped += 1
            continue
        
        records.append({
            'audio': {
                'array': audio,
                'sampling_rate': SAMPLE_RATE
            },
            'text': text,
            'recording_id': row['recording_id'],
            'duration': row['duration']
        })
    
    logger.info(f"Processed {len(records)} segments, skipped {skipped}")
    
    # Create HF Dataset
    dataset = Dataset.from_list(records)
    dataset = dataset.cast_column('audio', Audio(sampling_rate=SAMPLE_RATE))
    
    # Save to disk for caching
    dataset.save_to_disk(str(dataset_path))
    logger.info(f"Saved HF dataset to {dataset_path}")
    
    return dataset


def prepare_whisper_features(dataset: Dataset, processor: WhisperProcessor) -> Dataset:
    """
    Apply WhisperProcessor to convert raw audio + text into
    input_features (log-mel spectrogram) + labels (token IDs).
    
    Args:
        dataset: HF Dataset with 'audio' and 'text' columns.
        processor: WhisperProcessor instance.
        
    Returns:
        Dataset with 'input_features' and 'labels' columns.
    """
    def _prepare_batch(batch):
        # Extract log-mel spectrogram features
        audio = batch['audio']
        input_features = processor.feature_extractor(
            audio['array'],
            sampling_rate=audio['sampling_rate']
        ).input_features[0]
        
        # Tokenize the text
        labels = processor.tokenizer(batch['text']).input_ids
        
        return {
            'input_features': input_features,
            'labels': labels
        }
    
    logger.info("Computing Whisper features (log-mel spectrograms + token IDs)...")
    dataset = dataset.map(
        _prepare_batch,
        remove_columns=dataset.column_names,
        num_proc=1  # Audio processing isn't easily parallelizable
    )
    
    return dataset


# ─── Data Statistics ─────────────────────────────────────────────────────────

def compute_dataset_stats(manifest: pd.DataFrame) -> Dict:
    """Compute and log comprehensive dataset statistics."""
    stats = {
        'total_segments': len(manifest),
        'total_duration_hours': manifest['duration'].sum() / 3600,
        'unique_speakers': manifest['speaker_id'].nunique(),
        'unique_recordings': manifest['recording_id'].nunique(),
        'avg_segment_duration': manifest['duration'].mean(),
        'median_segment_duration': manifest['duration'].median(),
        'min_segment_duration': manifest['duration'].min(),
        'max_segment_duration': manifest['duration'].max(),
        'duration_std': manifest['duration'].std(),
    }
    
    logger.info("=" * 60)
    logger.info("DATASET STATISTICS")
    logger.info("=" * 60)
    for k, v in stats.items():
        logger.info(f"  {k:30s}: {v:.2f}" if isinstance(v, float) else f"  {k:30s}: {v}")
    logger.info("=" * 60)
    
    return stats


if __name__ == "__main__":
    from data_loader import prepare_raw_dataset
    
    manifest = prepare_raw_dataset()
    stats = compute_dataset_stats(manifest)
    dataset = build_hf_dataset(manifest)
    print(f"\nDataset: {dataset}")
    print(f"Sample: {dataset[0]}")
