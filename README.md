# Josh Talks — AI Researcher Intern: Speech & Audio Assessment

A production-grade solution for the Josh Talks AI Researcher Intern assessment, covering Hindi ASR fine-tuning, text cleanup pipelines, spelling classification, and lattice-based WER evaluation.

---

## 🏗️ Project Structure

```
josh/
├── config.py                          # Central configuration (paths, URLs, hyperparams)
├── requirements.txt                   # Python dependencies
│
├── src/
│   ├── data_loader.py                 # Async dataset download + URL rewriting
│   ├── preprocessing.py               # Audio segmentation + text cleaning
│   ├── whisper_finetune.py            # Seq2SeqTrainer training pipeline
│   ├── evaluation.py                  # FLEURS WER evaluation
│   ├── error_analysis.py              # Error taxonomy + fix implementation
│   ├── number_normalizer.py           # Hindi number word → digit conversion
│   ├── english_detector.py            # English loan word detection in Devanagari
│   ├── spell_checker.py              # 4-layer hybrid spelling classifier
│   └── lattice_wer.py                # Lattice construction + modified WER
│
├── notebooks/
│   ├── Q1_Whisper_Finetune.py         # Complete fine-tuning pipeline
│   ├── Q2_Cleanup_Pipeline.py         # Number normalization + English detection
│   ├── Q3_Spell_Checker.py            # Spelling classification analysis
│   └── Q4_Lattice_WER.py             # Lattice theory + implementation
│
├── reports/                           # Generated reports (after running)
│   ├── q1_wer_results.md
│   ├── q1_error_taxonomy.md
│   ├── q3_spelling_analysis.md
│   └── q4_lattice_theory.md
│
├── data/                              # Downloaded data (gitignored)
│   ├── raw_audio/
│   ├── processed/
│   └── cache/
│
└── outputs/                           # Model checkpoints (gitignored)
    └── whisper-small-hi-finetuned/
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Notebooks

Each question has a standalone notebook script:

```bash
# Q1: Whisper Fine-tuning (requires GPU)
python notebooks/Q1_Whisper_Finetune.py

# Q2: Cleanup Pipeline (runs on CPU)
python notebooks/Q2_Cleanup_Pipeline.py

# Q3: Spelling Classification (runs on CPU)
python notebooks/Q3_Spell_Checker.py

# Q4: Lattice WER Evaluation (runs on CPU)
python notebooks/Q4_Lattice_WER.py
```

> **Note for Windows Users**: Ensure your console supports UTF-8 for Hindi text by running `$env:PYTHONUTF8 = "1"` in PowerShell before executing the scripts.

### 3. Or Run Individual Modules

```bash
python -m src.number_normalizer      # Hindi number normalization examples
python -m src.english_detector       # English word detection examples
python -m src.spell_checker          # Process 1.77L words
python -m src.lattice_wer           # Lattice WER demo
```

---

## 📋 Question Summaries

### Q1: Whisper Fine-tuning

| Step | Description |
|------|-------------|
| **Preprocessing** | Async download (~10h audio + transcription JSONs), segment-level splitting using timestamps, text cleaning (Unicode NFC, punctuation removal) |
| **Fine-tuning** | `whisper-small` on Hindi data with FP16 + gradient checkpointing (RTX 4060 optimized) |
| **Evaluation** | Baseline vs fine-tuned WER on FLEURS Hindi test |
| **Error Analysis** | 25 systematically sampled errors, 7-category data-driven taxonomy, 3 fix proposals, 1 implemented fix (number normalization) with before/after WER |

### Q2: Cleanup Pipeline

| Component | Details |
|-----------|---------|
| **Number Normalization** | Complete Hindi number dictionary (0-99 + सौ/हज़ार/लाख/करोड़), compound number parsing, idiom exception handling |
| **English Detection** | 200+ curated English loan words, reverse transliteration engine, fuzzy matching |

### Q3: Spelling Classification

| Layer | Confidence | Description |
|-------|-----------|-------------|
| Dictionary | High | Core Hindi vocabulary lookup |
| English transliteration | High | Devanagari → Roman → English dictionary |
| Morphological | Medium | Valid suffix/prefix + known root |
| Character patterns | Varies | Valid Devanagari sequence validation |

### Q4: Lattice WER

- **Alignment unit**: Word-level (with justification vs character/subword/phrase)
- **Lattice construction**: Multiple Sequence Alignment across 6 models
- **Cost function**: Set-inclusion based (0 if predicted word ∈ bin, 1 otherwise)
- **Trust rule**: ≥ ⌈M/2⌉ model consensus → expand lattice bin

---

## 💻 System Requirements

| Component | Requirement |
|-----------|-------------|
| Python | 3.10+ |
| GPU | NVIDIA RTX 4050+ (6GB VRAM) for Q1 training |
| RAM | 16GB+ recommended |
| Storage | ~20GB for audio data + model checkpoints |

---

## 🪟 Windows & RTX 4050 Notes

If running this project on a Windows machine with a 6GB VRAM GPU (like the RTX 4050), the following adjustments are required and have been implemented in this environment:

1. **VRAM Optimization**: `config.py` batch sizes are reduced (`TRAIN_BATCH_SIZE=4`, `EVAL_BATCH_SIZE=2`) to fit training inside 6GB without OOM errors.
2. **Audio Processing Library Fixes**: The `torchcodec` library fails on Windows due to FFmpeg DLL issues. We downgraded the HF library to `datasets==2.21.0` and explicitly ran `pip uninstall torchcodec` to force the HuggingFace pipeline to safely fallback to `soundfile`.
3. **Hindi Encoding**: Set `$env:PYTHONUTF8="1"` when running the scripts in Windows PowerShell to prevent `UnicodeEncodeError` when the scripts print Hindi Devanagari characters to the terminal.
4. **Notebook Re-runs**: To rerun inference without automatically triggering the 4.5-hour Q1 training again, comment out the `train()` call in the Q1 notebook after the first successful run.

---

## 📊 Dataset

- **Source**: Josh Talks Hindi conversational dataset
- **Size**: ~10 hours, segmented into utterance-level clips  
- **Format**: WAV audio + JSON transcriptions with timestamps
- **URL Pattern**: `https://storage.googleapis.com/upload_goai/{uid}/{rid}_audio.wav`

---

## 📄 License

This project is submitted as part of the Josh Talks AI Researcher Intern assessment.
