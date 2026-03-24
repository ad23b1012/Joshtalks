"""
Data Loader — Download and parse the Josh Talks Hindi ASR dataset.

Handles:
  1. Fetching the metadata CSV from Google Sheets.
  2. Rewriting GCS URLs from the original bucket to the accessible one.
  3. Downloading audio files and transcription JSONs (parallel via asyncio).
  4. Parsing transcription JSONs into segment-level records.
"""
import os
import json
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import aiohttp
import pandas as pd
import requests
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATASET_CSV_URL, RAW_AUDIO_DIR, PROCESSED_DIR,
    ORIGINAL_BUCKET, REWRITE_BUCKET, DATA_DIR
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ─── URL Rewriting ───────────────────────────────────────────────────────────
def rewrite_url(url: str) -> str:
    """
    Rewrite GCS URLs from the original (access-restricted) bucket
    to the publicly accessible upload_goai bucket.
    
    Original:  .../joshtalks-data-collection/hq_data/hi/{uid}/{rid}_audio.wav
    Rewritten: .../upload_goai/{uid}/{rid}_audio.wav
    """
    return url.replace(ORIGINAL_BUCKET, REWRITE_BUCKET)


# ─── Metadata Loading ────────────────────────────────────────────────────────
def load_metadata(csv_url: str = DATASET_CSV_URL) -> pd.DataFrame:
    """
    Download the dataset metadata CSV and apply URL rewriting.
    
    Returns:
        DataFrame with columns:
        user_id, recording_id, language, duration,
        rec_url, transcription_url, metadata_url
        (all URLs rewritten to accessible bucket)
    """
    logger.info("Downloading dataset metadata CSV...")
    df = pd.read_csv(csv_url)
    
    # Apply URL rewriting to all URL columns
    url_cols = [c for c in df.columns if 'url' in c.lower()]
    for col in url_cols:
        df[col] = df[col].apply(rewrite_url)
    
    logger.info(f"Loaded {len(df)} recordings across {df['user_id'].nunique()} speakers")
    logger.info(f"Total duration: {df['duration'].sum() / 3600:.1f} hours")
    
    return df


# ─── Async Download Engine ───────────────────────────────────────────────────
async def _download_file(
    session: aiohttp.ClientSession,
    url: str,
    dest_path: Path,
    semaphore: asyncio.Semaphore,
    pbar: Optional[tqdm] = None
) -> bool:
    """Download a single file asynchronously."""
    if dest_path.exists():
        if pbar:
            pbar.update(1)
        return True
    
    async with semaphore:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status == 200:
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    content = await resp.read()
                    with open(dest_path, 'wb') as f:
                        f.write(content)
                    if pbar:
                        pbar.update(1)
                    return True
                else:
                    logger.warning(f"HTTP {resp.status} for {url}")
                    if pbar:
                        pbar.update(1)
                    return False
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            if pbar:
                pbar.update(1)
            return False


async def download_dataset(df: pd.DataFrame, max_concurrent: int = 10) -> Dict[str, List[Path]]:
    """
    Download all audio and transcription files in parallel.
    
    Args:
        df: Metadata DataFrame with rewritten URLs.
        max_concurrent: Max simultaneous downloads.
        
    Returns:
        Dict with 'audio_paths' and 'transcription_paths' lists.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = []
    audio_paths = []
    transcription_paths = []
    
    # Determine the URL column names
    rec_col = [c for c in df.columns if 'rec_url' in c.lower()][0]
    trans_col = [c for c in df.columns if 'transcription_url' in c.lower()][0]
    
    for _, row in df.iterrows():
        rid = row['recording_id']
        
        audio_dest = RAW_AUDIO_DIR / f"{rid}_audio.wav"
        trans_dest = RAW_AUDIO_DIR / f"{rid}_transcription.json"
        
        audio_paths.append(audio_dest)
        transcription_paths.append(trans_dest)
    
    total = len(audio_paths) + len(transcription_paths)
    pbar = tqdm(total=total, desc="Downloading dataset")
    
    async with aiohttp.ClientSession() as session:
        for i, (_, row) in enumerate(df.iterrows()):
            rid = row['recording_id']
            tasks.append(
                _download_file(session, row[rec_col], audio_paths[i], semaphore, pbar)
            )
            tasks.append(
                _download_file(session, row[trans_col], transcription_paths[i], semaphore, pbar)
            )
        
        results = await asyncio.gather(*tasks)
    
    pbar.close()
    
    success = sum(results)
    logger.info(f"Downloaded {success}/{total} files successfully")
    
    return {"audio_paths": audio_paths, "transcription_paths": transcription_paths}


# ─── Transcription Parsing ───────────────────────────────────────────────────
def parse_transcription_json(json_path: Path) -> List[Dict]:
    """
    Parse a transcription JSON file.
    
    Format: [{"start": float, "end": float, "speaker_id": int, "text": str}, ...]
    
    Returns:
        List of segment dictionaries.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            segments = json.load(f)
        return segments
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.warning(f"Failed to parse {json_path}: {e}")
        return []


def build_segment_manifest(
    df: pd.DataFrame,
    audio_paths: List[Path],
    transcription_paths: List[Path]
) -> pd.DataFrame:
    """
    Build a segment-level manifest by pairing each audio segment
    with its transcription text.
    
    Each original recording is split into utterance-level segments
    using the start/end timestamps from the transcription JSON.
    
    Returns:
        DataFrame with columns:
        recording_id, audio_path, start, end, duration, text, speaker_id
    """
    records = []
    
    for i, (_, row) in enumerate(df.iterrows()):
        rid = row['recording_id']
        audio_path = audio_paths[i]
        trans_path = transcription_paths[i]
        
        if not trans_path.exists() or not audio_path.exists():
            continue
        
        segments = parse_transcription_json(trans_path)
        
        for seg in segments:
            text = seg.get('text', '').strip()
            if not text:
                continue
                
            seg_duration = seg['end'] - seg['start']
            
            # Skip segments that are too short or too long
            if seg_duration < 0.5 or seg_duration > 30.0:
                continue
            
            records.append({
                'recording_id': rid,
                'audio_path': str(audio_path),
                'start': seg['start'],
                'end': seg['end'],
                'duration': seg_duration,
                'text': text,
                'speaker_id': seg.get('speaker_id', row['user_id'])
            })
    
    manifest = pd.DataFrame(records)
    logger.info(
        f"Built manifest: {len(manifest)} segments from "
        f"{manifest['recording_id'].nunique()} recordings, "
        f"total duration: {manifest['duration'].sum() / 3600:.2f}h"
    )
    
    return manifest


# ─── Main Entry Point ────────────────────────────────────────────────────────
def prepare_raw_dataset() -> pd.DataFrame:
    """
    End-to-end: load metadata → download files → build manifest.
    
    Returns:
        Segment-level manifest DataFrame.
    """
    manifest_path = PROCESSED_DIR / "segment_manifest.csv"
    
    if manifest_path.exists():
        logger.info(f"Loading cached manifest from {manifest_path}")
        return pd.read_csv(manifest_path)
    
    # Step 1: Load metadata
    df = load_metadata()
    
    # Step 2: Download all files
    paths = asyncio.run(download_dataset(df))
    
    # Step 3: Build segment manifest
    manifest = build_segment_manifest(df, paths['audio_paths'], paths['transcription_paths'])
    
    # Step 4: Save
    manifest.to_csv(manifest_path, index=False)
    logger.info(f"Saved manifest to {manifest_path}")
    
    return manifest


if __name__ == "__main__":
    manifest = prepare_raw_dataset()
    print(f"\nDataset Summary:")
    print(f"  Total segments:   {len(manifest):,}")
    print(f"  Total duration:   {manifest['duration'].sum() / 3600:.2f} hours")
    print(f"  Unique speakers:  {manifest['speaker_id'].nunique()}")
    print(f"\nDuration distribution (seconds):")
    print(manifest['duration'].describe())
