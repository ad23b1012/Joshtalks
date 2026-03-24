#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║  Question 2: ASR Cleanup Pipeline                                    ║
║  Josh Talks AI Researcher Intern Assessment                         ║
║                                                                      ║
║  Operations:                                                         ║
║    a) Hindi Number Normalization (with idiom handling)                ║
║    b) English Word Detection in Devanagari text                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# %% [markdown]
# # Question 2: ASR Output Cleanup Pipeline
#
# Raw ASR output from Hindi conversations is messy:
# - Numbers come out as words
# - English words spoken in conversation are not always identified correctly 
# 
# This pipeline performs two operations and evaluates their impact.

# %%
import sys
sys.path.insert(0, '..')

from src.number_normalizer import normalize_hindi_numbers, run_examples as num_examples
from src.english_detector import tag_english_words, analyze_english_density, run_examples as eng_examples

# %% [markdown]
# ## 2a. Number Normalization
# 
# Convert spoken Hindi number words into digits while preserving idioms.

# %%
print("=" * 70)
print("PART A: NUMBER NORMALIZATION")
print("=" * 70)

# ── Correct Conversions (4-5 examples) ──
correct_examples = [
    "मेरी उम्र पच्चीस साल है",
    "तीन सौ चौवन लोग आए",
    "एक हज़ार रुपये दो",
    "सोलह दिन बाद मिलेंगे",
    "दो लाख तीन हज़ार चार सौ पच्चीस रुपये का बिल",
]

print("\n### Correct Conversions:\n")
for text in correct_examples:
    result = normalize_hindi_numbers(text)
    print(f"  Before: {text}")
    print(f"  After:  {result}")
    print()

# ── Edge Cases (2-3 examples with reasoning) ──
print("\n### Edge Cases:\n")

edge_cases = [
    (
        "दो-चार बातें करनी हैं",
        "PRESERVE — 'दो-चार बातें' is an idiom meaning 'a few things'. "
        "Converting to '2-4 बातें' would destroy the idiomatic meaning. "
        "We detect hyphenated number pairs as idiom markers."
    ),
    (
        "नौ दो ग्यारह हो गए",
        "PRESERVE — 'नौ दो ग्यारह' means 'to flee/disappear'. "
        "This is a well-known Hindi saying (9+2=11, metaphor for running away). "
        "We maintain a curated idiom exception list."
    ),
    (
        "हजारों लोग वहां थे",
        "PRESERVE — 'हजारों' (with ों suffix) is the approximate plural form, "
        "meaning 'thousands of'. Converting to '1000 लोग' would change the meaning "
        "from 'thousands' to exactly '1000'. We skip pluralized number forms."
    ),
]

for text, reasoning in edge_cases:
    result = normalize_hindi_numbers(text)
    print(f"  Before:    {text}")
    print(f"  After:     {result}")
    print(f"  Reasoning: {reasoning}")
    print()

# %% [markdown]
# ## 2b. English Word Detection
#
# Identify English words in Devanagari script using a multi-strategy approach:
# 1. Curated high-frequency dictionary (200+ common English loan words)
# 2. Reverse transliteration → English dictionary lookup  
# 3. Fuzzy matching for novel loan words

# %%
print("=" * 70)
print("PART B: ENGLISH WORD DETECTION")
print("=" * 70)

examples = [
    "मेरा इंटरव्यू बहुत अच्छा गया और मुझे जॉब मल गई",
    "ये प्रॉब्लम सॉल्व नहीं हो रहा",
    "कंप्यूटर और फ़ोन दोनों खराब हो गए",
    "मैं कॉलेज में टीचर हूं",
    "ट्रेन का टिकट ऑनलाइन बुक कर लो",
    "डॉक्टर ने मेडिसिन दी है",
    "बहुत प्योर हार्ट रहता है सबका",
    "जी फीडबैक मिलने पर सुधार करना",
]

print()
for text in examples:
    tagged = tag_english_words(text)
    print(f"  Input:  {text}")
    print(f"  Output: {tagged}")
    print()

# %% [markdown]
# ## Pipeline Impact Analysis
# 
# To evaluate where normalization helps vs. hurts, we'd run the pretrained 
# Whisper-small on all audio segments, then measure WER with and without
# each normalization step applied to the predictions.

# %%
# Pseudocode for impact analysis (requires GPU inference):
# 
# 1. Run pretrained Whisper on all audio → raw_predictions
# 2. Pair each raw prediction with human reference
# 3. Compute baseline WER
# 4. Apply number normalization to predictions → normalized_predictions
# 5. Compute WER again
# 6. If WER_normalized < WER_baseline → normalization helps
# 7. Repeat for English tagging (remove tags before WER comparison)

print("\nPipeline ready for evaluation on GPU machine.")
print("Steps: Generate ASR output → Apply normalizations → Measure WER impact")
