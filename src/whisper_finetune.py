"""
Whisper Fine-tuning — Seq2Seq training pipeline for Hindi ASR.

Uses HuggingFace Transformers Seq2SeqTrainer with:
  - Gradient checkpointing + FP16 for RTX 4060 (8GB VRAM)
  - SpecAugment (built into Whisper)
  - Custom data collator with label padding
"""
import torch
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from pathlib import Path

from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MODEL_NAME, LANGUAGE, TASK, MODEL_DIR,
    TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE, WARMUP_STEPS, MAX_STEPS, SAVE_STEPS, EVAL_STEPS,
    LOGGING_STEPS, FP16, GRADIENT_CHECKPOINTING
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ─── Data Collator ───────────────────────────────────────────────────────────

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Custom data collator for Whisper Seq2Seq training.
    
    Handles:
      - Padding input features (log-mel spectrograms) via the feature extractor
      - Padding label sequences (token IDs) via the tokenizer
      - Masking padded labels with -100 for loss computation
      - Stripping the BOS token if present at start of all labels
    """
    processor: WhisperProcessor
    decoder_start_token_id: int = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Pad input features (spectrograms)
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad labels (token sequences)
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        # Replace padding token IDs with -100 for loss masking
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Strip the BOS token from labels if it's at position 0 for all samples
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ─── Training Setup ──────────────────────────────────────────────────────────

def get_training_args(output_dir: str = None) -> Seq2SeqTrainingArguments:
    """
    Configure training arguments optimized for RTX 4060 (8GB VRAM).
    """
    if output_dir is None:
        output_dir = str(MODEL_DIR)
    
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        
        # Batch & memory optimization
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        fp16=FP16,
        
        # Learning rate schedule
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        lr_scheduler_type="linear",
        weight_decay=0.01,
        
        # Evaluation & saving
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        
        # Generation (for evaluation)
        predict_with_generate=True,
        generation_max_length=225,
        
        # Logging
        logging_steps=LOGGING_STEPS,
        report_to=["tensorboard"],
        logging_dir=str(Path(output_dir) / "logs"),
        
        # Optimization
        optim="adamw_torch",
        
        # Seed
        seed=42,
        data_seed=42,
        
        # Misc
        remove_unused_columns=False,
        label_names=["labels"],
        push_to_hub=False,
    )


def load_model_and_processor():
    """
    Load Whisper-small model and processor with Hindi language settings.
    
    Returns:
        Tuple of (model, processor)
    """
    logger.info(f"Loading {MODEL_NAME}...")
    
    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME,
        language=LANGUAGE,
        task=TASK
    )
    
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    
    # Configure the model for Hindi transcription
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model loaded: {total_params/1e6:.1f}M total, {trainable_params/1e6:.1f}M trainable")
    
    return model, processor


def create_trainer(
    model,
    processor,
    train_dataset,
    eval_dataset,
    output_dir: str = None
) -> Seq2SeqTrainer:
    """
    Create a Seq2SeqTrainer with all components configured.
    
    Args:
        model: WhisperForConditionalGeneration instance.
        processor: WhisperProcessor instance.
        train_dataset: Preprocessed training dataset.
        eval_dataset: Preprocessed evaluation dataset.
        output_dir: Output directory for checkpoints.
        
    Returns:
        Configured Seq2SeqTrainer.
    """
    # WER metric
    wer_metric = evaluate.load("wer")
    
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        # Replace -100 with pad token for decoding
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        
        # Decode predictions and references
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        # Compute WER
        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        
        return {"wer": wer}
    
    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    # Training arguments
    training_args = get_training_args(output_dir)
    
    # Create trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )
    
    return trainer


# ─── Main Training Loop ─────────────────────────────────────────────────────

def train(train_dataset, eval_dataset, resume_from_checkpoint: str = None):
    """
    Full training pipeline:
      1. Load model and processor
      2. Create trainer
      3. Train
      4. Save final model
      5. Return trainer for evaluation
    """
    model, processor = load_model_and_processor()
    
    trainer = create_trainer(model, processor, train_dataset, eval_dataset)
    
    logger.info("=" * 60)
    logger.info("STARTING TRAINING")
    logger.info(f"  Train samples: {len(train_dataset)}")
    logger.info(f"  Eval samples:  {len(eval_dataset)}")
    logger.info(f"  Max steps:     {MAX_STEPS}")
    logger.info(f"  Batch size:    {TRAIN_BATCH_SIZE} x {GRADIENT_ACCUMULATION_STEPS} = {TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    logger.info("=" * 60)
    
    # Train
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Save the final model
    trainer.save_model(str(MODEL_DIR))
    processor.save_pretrained(str(MODEL_DIR))
    
    # Log training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    logger.info(f"Training complete. Model saved to {MODEL_DIR}")
    
    return trainer, processor


if __name__ == "__main__":
    print("Whisper Fine-tuning module loaded successfully.")
    print(f"Model: {MODEL_NAME}")
    print(f"Output: {MODEL_DIR}")
    print(f"Effective batch size: {TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print("\nTo train, import and call train(train_dataset, eval_dataset)")
