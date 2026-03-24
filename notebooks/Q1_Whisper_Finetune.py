#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║  Question 1: Hindi ASR — Whisper Fine-tuning Pipeline               ║
║  Josh Talks AI Researcher Intern Assessment                         ║
║                                                                      ║
║  This notebook covers:                                               ║
║    a) Data preprocessing                                             ║
║    b) Fine-tuning Whisper-small on ~10h Hindi dataset                ║
║    c) WER evaluation on FLEURS Hindi test set                        ║
║    d) Systematic error sampling (25 utterances)                      ║
║    e) Error taxonomy with concrete examples                          ║
║    f) Fix proposals for top 3 error types                            ║
║    g) Implemented fix with before/after results                      ║
╚══════════════════════════════════════════════════════════════════════╝

To run as notebook: jupyter notebook, then import this file
To run as script:   python notebooks/Q1_Whisper_Finetune.py
"""

# %% [markdown]
# # Question 1: Hindi ASR — Whisper Fine-tuning Pipeline
# 
# ## 1a. Data Preprocessing

# %%
import sys
sys.path.insert(0, '..')

from src.data_loader import prepare_raw_dataset, load_metadata
from src.preprocessing import (
    build_hf_dataset, prepare_whisper_features, 
    compute_dataset_stats, clean_text
)
from config import SAMPLE_RATE, MODEL_NAME

print("Loading and preprocessing the ~10-hour Hindi dataset...")

# %% [markdown]
# ### Step 1: Load Metadata & Rewrite URLs
# The dataset CSV contains GCS URLs pointing to audio, transcription, and metadata.
# We rewrite URLs from `joshtalks-data-collection` to `upload_goai` bucket.

# %%
# Load metadata - this automatically rewrites URLs
metadata_df = load_metadata()
print(f"\nDataset overview:")
print(f"  Recordings:   {len(metadata_df)}")
print(f"  Speakers:     {metadata_df['user_id'].nunique()}")
print(f"  Total hours:  {metadata_df['duration'].sum() / 3600:.1f}h")
print(f"\nSample record:")
print(metadata_df.iloc[0].to_dict())

# %% [markdown]
# ### Step 2: Download Audio & Transcriptions (Async)
# Files are downloaded in parallel with `aiohttp` (10 concurrent connections).

# %%
# Full download + manifest building
manifest = prepare_raw_dataset()
stats = compute_dataset_stats(manifest)

print(f"\nPreprocessing summary:")
print(f"  Total segments:       {stats['total_segments']:,}")
print(f"  Total duration:       {stats['total_duration_hours']:.2f} hours")
print(f"  Avg segment length:   {stats['avg_segment_duration']:.1f}s")
print(f"  Median segment:       {stats['median_segment_duration']:.1f}s")

# %% [markdown]
# ### Step 3: Text Cleaning
# We apply Unicode NFC normalization and remove non-Devanagari punctuation.

# %%
# Demo text cleaning
sample_texts = [
    'अब काफी अच्छा होता है, क्योंकि उनकी जनसंख्या बहुत कम दी जा रही है!',
    'मेरा interview बहुत अच्छा गया - और मुझे जॉब मिल गई।',
]
for text in sample_texts:
    print(f"  Original: {text}")
    print(f"  Cleaned:  {clean_text(text)}")
    print()

# %% [markdown]
# ### Step 4: Build HuggingFace Dataset with Whisper Features

# %%
from transformers import WhisperProcessor

# Build HF dataset with audio arrays + cleaned text
dataset = build_hf_dataset(manifest)
print(f"\nHuggingFace Dataset: {dataset}")

# Apply Whisper feature extraction
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="hi", task="transcribe")
processed_dataset = prepare_whisper_features(dataset, processor)
print(f"Processed Dataset: {processed_dataset}")

# %% [markdown]
# ## 1b. Fine-tuning Whisper-small
# 
# Training configuration optimized for RTX 4060 (8GB VRAM):
# - Batch size: 8, Gradient accumulation: 2 (effective batch: 16)
# - FP16 + Gradient checkpointing
# - Learning rate: 1e-5 with linear warmup (500 steps)

# %%
from src.whisper_finetune import train, load_model_and_processor

# Split dataset (90% train, 10% eval)
split = processed_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split['train']
eval_dataset = split['test']

print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

# Start training
trainer, processor = train(train_dataset, eval_dataset)

# %% [markdown]
# ## 1c. WER Evaluation on FLEURS Hindi

# %%
from src.evaluation import evaluate_on_fleurs

# Evaluate both pretrained baseline and fine-tuned model
report = evaluate_on_fleurs()
print(report.to_markdown(index=False))

# %% [markdown]
# ## 1d-e. Error Analysis & Taxonomy

# %%
from src.error_analysis import (
    sample_errors, build_error_taxonomy, propose_fixes,
    evaluate_fix, generate_error_report
)

# Sample 25 errors systematically
sampled = sample_errors("reports/q1_predictions.csv")
taxonomy = build_error_taxonomy(sampled)

print("Error Taxonomy:")
for category, examples in taxonomy.items():
    print(f"\n  {category} ({len(examples)} examples)")
    for ex in examples[:2]:
        print(f"    Ref:  {ex['reference'][:60]}...")
        print(f"    Pred: {ex['prediction'][:60]}...")

# %% [markdown]
# ## 1f. Proposed Fixes

# %%
fixes = propose_fixes()
for fix in fixes:
    print(f"\n[{fix['feasibility']}] {fix['category']}")
    print(f"  Proposal: {fix['proposal']}")
    print(f"  {fix['description'][:100]}...")

# %% [markdown]
# ## 1g. Implemented Fix — Number Normalization

# %%
fix_results = evaluate_fix("reports/q1_predictions.csv")
print(fix_results.to_markdown(index=False))

# Generate complete report
generate_error_report(sampled, taxonomy, fixes, fix_results)
print("\nQ1 pipeline ready. Uncomment training calls to run on GPU.")
