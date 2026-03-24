#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║  Question 4: Lattice-based WER Evaluation                            ║
║  Josh Talks AI Researcher Intern Assessment                         ║
║                                                                      ║
║  Theory + Implementation for fair ASR evaluation using lattices      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# %% [markdown]
# # Question 4: Lattice-based WER Evaluation
#
# ## Problem Statement
# Comparing ASR output against a single rigid Ground Truth string 
# unfairly penalizes valid transcriptions. This notebook designs and 
# implements a lattice-based approach that captures all valid alternatives.

# %%
import sys
sys.path.insert(0, '..')

from src.lattice_wer import (
    construct_lattice, construct_lattice_with_synonyms,
    compute_lattice_wer, compute_standard_wer,
    evaluate_all_models, generate_q4_report,
    _clean_for_alignment
)

# %% [markdown]
# ## 1. Alignment Unit Choice: WORD
# 
# | Unit | Lattice Complexity | Cross-model Consistency | Verdict |
# |------|--------------------|------------------------|---------|
# | Character | O(n×k) per char — explodes | Consistent | ✗ Too complex |
# | Subword (BPE) | Model-specific | Different tokenizers | ✗ Inconsistent |
# | **Word** | **O(n×k) per word — tractable** | **Natural unit** | **✓ Chosen** |
# | Phrase | O(n×k) per phrase | Chunking ambiguity | ✗ Too coarse |
# 
# Word-level alignment provides the optimal balance between granularity 
# and tractability for Hindi ASR evaluation.

# %% [markdown]
# ## 2. Lattice Construction — Demo

# %%
# Example from the assignment
reference = "उसने चौदह किताबें खरीदीं"

models = {
    'Model_A': "उसने 14 पुस्तकें खरीदी",
    'Model_B': "उसने चौदह किताबे खरीदीं",
    'Model_C': "उसने चौदह किताबें खरीदी",
}

print(f"Reference: {reference}")
print(f"\nModel outputs:")
for name, output in models.items():
    print(f"  {name}: {output}")

# Construct lattice
lattice = construct_lattice_with_synonyms(
    reference, models, number_equivalences=True
)

print(f"\nConstructed Lattice (bins per position):")
ref_words = _clean_for_alignment(reference)
for i, (word, bin_set) in enumerate(zip(ref_words, lattice)):
    print(f"  Position {i} [{word}]: {bin_set}")

# %% [markdown]
# ## 3. WER Comparison: Standard vs. Lattice

# %%
print(f"\n{'Model':15s} {'Standard WER':>15s} {'Lattice WER':>15s} {'Reduction':>12s}")
print("=" * 58)

for name, output in models.items():
    std = compute_standard_wer(output, reference)
    lat = compute_lattice_wer(output, lattice)
    delta = std - lat
    print(f"{name:15s} {std:15.2%} {lat:15.2%} {delta:12.2%}")

print(f"\nKey insight: Models outputting valid alternatives (14 instead of चौदह,")
print(f"पुस्तकें instead of किताबें) see reduced WER with lattice evaluation.")

# %% [markdown]
# ## 4. Consensus Trust Rule
# 
# **Rule**: If ≥ ⌈M/2⌉ models agree on a word W at position i, and W 
# differs from the human reference, add W to the lattice bin.
# 
# **Rationale**: Human annotations contain errors. If most models 
# independently produce the same output, it's statistically likely 
# to be correct. But the human reference is NEVER removed — only expanded.
# 
# **Safeguard**: Lattice bins are additive. We can never make WER worse 
# by expanding bins — we can only make it better or keep it the same.

# %% [markdown]
# ## 5. Full Evaluation on Q4 Dataset

# %%
# Run on the provided dataset with Human + 6 model transcriptions
try:
    summary = evaluate_all_models()
    print(summary.to_markdown(index=False))
    
    # Generate theory report
    generate_q4_report(summary)
except Exception as e:
    print(f"Full evaluation: {e}")
    print("(Expected if running without internet)")

# %% [markdown]
# ## 6. Pseudocode Summary
# 
# ```
# function CONSTRUCT_LATTICE(reference, model_outputs, threshold):
#     ref_words = TOKENIZE(reference)
#     lattice = [{word} for word in ref_words]
#     
#     for each model in model_outputs:
#         alignment = ALIGN(ref_words, model.words)
#         for (ref_pos, model_word) in alignment:
#             position_votes[ref_pos][model_word] += 1
#     
#     for each position:
#         for candidate, votes in position_votes[position]:
#             if votes >= ceil(n_models * threshold):
#                 lattice[position].add(candidate)
#         EXPAND_WITH_LINGUISTIC_KNOWLEDGE(lattice[position])
#     
#     return lattice
# 
# function LATTICE_WER(prediction, lattice):
#     dp[i][j] = min(
#         dp[i-1][j] + 1,                                  // deletion
#         dp[i][j-1] + 1,                                  // insertion
#         dp[i-1][j-1] + (0 if pred[j] IN lattice[i] else 1)  // sub/match
#     )
#     return dp[n][m] / n
# ```
