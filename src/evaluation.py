"""
Evaluation — WER computation on FLEURS Hindi test set.

Evaluates both the pretrained Whisper-small baseline and the
fine-tuned model, then generates a structured WER report.
"""
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import pandas as pd
import numpy as np
from datasets import load_dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    pipeline
)
import evaluate
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MODEL_NAME, MODEL_DIR, LANGUAGE, TASK, SAMPLE_RATE,
    FLEURS_DATASET, FLEURS_LANGUAGE, FLEURS_SPLIT,
    REPORTS_DIR
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ─── FLEURS Dataset Loader ───────────────────────────────────────────────────

def load_fleurs_test() -> "Dataset":
    """
    Load the FLEURS Hindi test split.
    
    Returns:
        HuggingFace Dataset with 'audio' and 'transcription' columns.
    """
    logger.info(f"Loading FLEURS {FLEURS_LANGUAGE} {FLEURS_SPLIT} split...")
    
    fleurs = load_dataset(FLEURS_DATASET, FLEURS_LANGUAGE, split=FLEURS_SPLIT, trust_remote_code=True)
    fleurs = fleurs.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    
    logger.info(f"Loaded {len(fleurs)} test samples from FLEURS")
    return fleurs


# ─── Inference Engine ────────────────────────────────────────────────────────

def run_inference(
    model_path: str,
    dataset,
    batch_size: int = 8,
    device: str = None
) -> List[str]:
    """
    Run batch inference on a dataset using a Whisper model.
    
    Args:
        model_path: Path to model (local dir or HuggingFace model ID).
        dataset: HF Dataset with 'audio' column.
        batch_size: Inference batch size.
        device: Device string ('cuda', 'cpu', etc.)
        
    Returns:
        List of predicted transcriptions.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Loading model from {model_path} on {device}...")
    
    # Use the pipeline API for cleaner inference
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model_path,
        device=device,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        chunk_length_s=30,
        batch_size=batch_size,
        generate_kwargs={
            "language": "hi",
            "task": "transcribe"
        }
    )
    
    predictions = []
    
    logger.info(f"Running inference on {len(dataset)} samples...")
    
    # Process in batches
    for i in tqdm(range(0, len(dataset), batch_size), desc="Inference"):
        batch = dataset[i:i + batch_size]
        audio_inputs = [
            {"raw": audio["array"], "sampling_rate": audio["sampling_rate"]}
            for audio in batch["audio"]
        ]
        
        results = asr_pipeline(
            audio_inputs,
            return_timestamps=False
        )
        
        for result in results:
            predictions.append(result["text"].strip())
    
    return predictions


# ─── WER Computation ─────────────────────────────────────────────────────────

def compute_wer(predictions: List[str], references: List[str]) -> float:
    """
    Compute Word Error Rate.
    
    Args:
        predictions: List of predicted transcripts.
        references: List of reference transcripts.
        
    Returns:
        WER as a ratio (0.0 to 1.0+).
    """
    wer_metric = evaluate.load("wer")
    wer = wer_metric.compute(predictions=predictions, references=references)
    return wer


def compute_per_sample_wer(
    predictions: List[str],
    references: List[str]
) -> List[float]:
    """
    Compute WER for each individual sample.
    
    Returns:
        List of per-sample WER values.
    """
    wer_metric = evaluate.load("wer")
    per_sample = []
    
    for pred, ref in zip(predictions, references):
        if not ref.strip():
            per_sample.append(0.0)
            continue
        sample_wer = wer_metric.compute(predictions=[pred], references=[ref])
        per_sample.append(sample_wer)
    
    return per_sample


# ─── Full Evaluation Pipeline ────────────────────────────────────────────────

def evaluate_on_fleurs(
    pretrained_model: str = MODEL_NAME,
    finetuned_model: str = None,
    batch_size: int = 8
) -> pd.DataFrame:
    """
    Evaluate both pretrained and fine-tuned models on FLEURS Hindi test set.
    
    Args:
        pretrained_model: HuggingFace model ID for baseline.
        finetuned_model: Path to fine-tuned model directory.
        batch_size: Inference batch size.
        
    Returns:
        DataFrame with evaluation results.
    """
    if finetuned_model is None:
        finetuned_model = str(MODEL_DIR)
    
    # Load test set
    fleurs = load_fleurs_test()
    references = fleurs["transcription"]
    
    results = {}
    
    # Evaluate pretrained baseline
    logger.info("=" * 60)
    logger.info("EVALUATING PRETRAINED WHISPER-SMALL BASELINE")
    logger.info("=" * 60)
    
    start_time = time.time()
    pretrained_preds = run_inference(pretrained_model, fleurs, batch_size)
    pretrained_time = time.time() - start_time
    
    pretrained_wer = compute_wer(pretrained_preds, references)
    results['pretrained'] = {
        'predictions': pretrained_preds,
        'wer': pretrained_wer,
        'time': pretrained_time,
        'per_sample_wer': compute_per_sample_wer(pretrained_preds, references)
    }
    logger.info(f"Pretrained WER: {pretrained_wer:.4f} ({pretrained_wer*100:.1f}%)")
    
    # Evaluate fine-tuned model
    logger.info("=" * 60)
    logger.info("EVALUATING FINE-TUNED WHISPER-SMALL")
    logger.info("=" * 60)
    
    start_time = time.time()
    finetuned_preds = run_inference(finetuned_model, fleurs, batch_size)
    finetuned_time = time.time() - start_time
    
    finetuned_wer = compute_wer(finetuned_preds, references)
    results['finetuned'] = {
        'predictions': finetuned_preds,
        'wer': finetuned_wer,
        'time': finetuned_time,
        'per_sample_wer': compute_per_sample_wer(finetuned_preds, references)
    }
    logger.info(f"Fine-tuned WER: {finetuned_wer:.4f} ({finetuned_wer*100:.1f}%)")
    
    # Generate structured report
    report_df = generate_wer_report(results, references)
    
    # Save detailed predictions for error analysis
    save_predictions(results, references, fleurs)
    
    return report_df


def generate_wer_report(results: Dict, references: List[str]) -> pd.DataFrame:
    """
    Generate the structured WER report table as required by the assignment.
    """
    report = pd.DataFrame({
        'Model': ['Whisper Small (Pretrained)', 'FT Whisper Small (ours)'],
        'Hindi WER': [
            f"{results['pretrained']['wer']:.4f}",
            f"{results['finetuned']['wer']:.4f}"
        ],
        'Inference Time (s)': [
            f"{results['pretrained']['time']:.1f}",
            f"{results['finetuned']['time']:.1f}"
        ]
    })
    
    # Save report
    report_path = REPORTS_DIR / "q1_wer_results.md"
    with open(report_path, 'w') as f:
        f.write("# Question 1c — WER Results on FLEURS Hindi Test Set\n\n")
        f.write("| Model | Hindi WER | Inference Time (s) |\n")
        f.write("|-------|-----------|--------------------|\n")
        for _, row in report.iterrows():
            f.write(f"| {row['Model']} | {row['Hindi WER']} | {row['Inference Time (s)']} |\n")
        f.write(f"\n*Evaluated on {len(references)} test samples from FLEURS hi_in.*\n")
        f.write(f"\nValues are raw WER ratios (e.g., 0.30 = 30%)\n")
    
    logger.info(f"WER report saved to {report_path}")
    return report


def save_predictions(results: Dict, references: List[str], dataset) -> Path:
    """
    Save all predictions to a CSV for downstream error analysis.
    """
    output_path = REPORTS_DIR / "q1_predictions.csv"
    
    records = []
    for i, ref in enumerate(references):
        records.append({
            'index': i,
            'reference': ref,
            'pretrained_prediction': results['pretrained']['predictions'][i],
            'finetuned_prediction': results['finetuned']['predictions'][i],
            'pretrained_wer': results['pretrained']['per_sample_wer'][i],
            'finetuned_wer': results['finetuned']['per_sample_wer'][i],
            'wer_improvement': (
                results['pretrained']['per_sample_wer'][i] -
                results['finetuned']['per_sample_wer'][i]
            )
        })
    
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")
    
    return output_path


if __name__ == "__main__":
    print("Evaluation module ready.")
    print(f"FLEURS: {FLEURS_DATASET} / {FLEURS_LANGUAGE} / {FLEURS_SPLIT}")
    print(f"Baseline: {MODEL_NAME}")
    print(f"Fine-tuned: {MODEL_DIR}")
    print("\nTo evaluate: evaluate_on_fleurs()")
