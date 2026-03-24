"""
Lattice-based WER Evaluation (Q4).

Theory + Implementation for constructing transcription lattices
and computing fair WER scores.

Key concepts:
  - Alignment unit: WORD level (justified below)
  - Lattice construction from multiple ASR model outputs
  - Modified Levenshtein distance with set-inclusion cost
  - Consensus-based reference correction (majority vote)
  - Per-model WER computation using lattice vs. rigid reference
"""
import re
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import LATTICE_DATA_CSV_URL, REPORTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# THEORETICAL FRAMEWORK
# ═══════════════════════════════════════════════════════════════════════════
"""
1. ALIGNMENT UNIT CHOICE: WORD

   Why word-level and not subword/character/phrase?

   a) Character-level: Too granular — lattice explodes in complexity.
      A single word "खरीदीं" generates 6+ nodes. For ~10 words per 
      utterance and 6 models, the lattice becomes intractable.
      
   b) Subword-level (BPE/SentencePiece): Model-dependent tokenization 
      makes cross-model alignment inconsistent. Different models use 
      different subword vocabularies.
      
   c) Phrase-level: Too coarse — loses the ability to identify precisely 
      which words differ between models. Multi-word phrase alignment 
      requires complex chunking heuristics.
      
   d) Word-level (chosen): Natural unit for Hindi text. Each position 
      in the lattice corresponds to one spoken word, allowing direct 
      comparison across models. The lattice has O(n × k) bins where 
      n = utterance length and k = average alternatives per position.

2. LATTICE STRUCTURE

   A lattice L = [B₁, B₂, ..., Bₙ] where each Bᵢ is a "bin" containing 
   all valid text representations for position i. For example:
   
   L = [{"उसने"}, {"चौदह", "14"}, {"किताबें", "किताबे", "पुस्तकें"}, {"खरीदीं", "खरीदी"}]
   
   This captures:
   - Number format variants (चौदह ↔ 14)
   - Spelling variations (किताबें ↔ किताबे)  
   - Lexical synonyms (किताबें ↔ पुस्तकें)
   - Morphological variants (खरीदीं ↔ खरीदी)

3. HANDLING INSERTIONS/DELETIONS/SUBSTITUTIONS

   Standard WER uses Levenshtein distance with uniform cost (1) for 
   insertions, deletions, and substitutions. In lattice WER:
   
   - Substitution cost = 0 if the predicted word ∈ Bᵢ (any valid variant)
   - Substitution cost = 1 if the predicted word ∉ Bᵢ
   - Insertion cost = 1 (extra word in prediction)
   - Deletion cost = 1 (missing word from reference)
   
   This ensures models aren't penalized for valid alternative representations.

4. CONSENSUS-BASED TRUST

   When to trust model agreement over the reference:
   
   Rule: If ≥ ceil(M/2) models agree on a word that differs from the 
         human reference (where M = number of models), add the agreed 
         word to the lattice bin.
   
   Rationale: If most models independently produce the same output, it's 
   likely correct even if the human reference is wrong (human annotation 
   errors exist). However, we never REMOVE the human reference from the 
   bin — we only EXPAND it.
"""


# ═══════════════════════════════════════════════════════════════════════════
# ALIGNMENT UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def _clean_for_alignment(text: str) -> List[str]:
    """
    Tokenize text into words for alignment.
    Removes punctuation, normalizes whitespace.
    """
    text = re.sub(r'[।,\.\!\?\;\:\"\'\-\(\)\[\]]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.split() if text else []


def _word_level_alignment(ref_words: List[str], hyp_words: List[str]) -> List[Tuple]:
    """
    Align two word sequences using minimum edit distance with backtrace.
    
    Returns:
        List of (operation, ref_pos, hyp_pos) tuples.
        Operations: 'match', 'sub', 'ins', 'del'
    """
    n = len(ref_words)
    m = len(hyp_words)
    
    # DP table
    dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    
    # Backtrace table
    bt = [['' for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        bt[i][0] = 'D'
    for j in range(1, m + 1):
        bt[0][j] = 'I'
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                cost_sub = 0
                bt_sub = 'M'
            else:
                cost_sub = 1
                bt_sub = 'S'
            
            candidates = [
                (dp[i-1][j-1] + cost_sub, bt_sub),   # match/substitute
                (dp[i-1][j] + 1, 'D'),                # deletion
                (dp[i][j-1] + 1, 'I'),                # insertion
            ]
            
            best = min(candidates, key=lambda x: x[0])
            dp[i][j] = best[0]
            bt[i][j] = best[1]
    
    # Backtrace to get alignment
    alignment = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and bt[i][j] in ('M', 'S'):
            alignment.append((bt[i][j], i-1, j-1))
            i -= 1
            j -= 1
        elif i > 0 and bt[i][j] == 'D':
            alignment.append(('D', i-1, None))
            i -= 1
        elif j > 0 and bt[i][j] == 'I':
            alignment.append(('I', None, j-1))
            j -= 1
        else:
            break
    
    alignment.reverse()
    return alignment


# ═══════════════════════════════════════════════════════════════════════════
# LATTICE CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════

def construct_lattice(
    reference: str,
    model_outputs: Dict[str, str],
    consensus_threshold: float = 0.5
) -> List[Set[str]]:
    """
    Construct a transcription lattice from multiple model outputs.
    
    Algorithm:
      1. Start with the human reference as the base sequence.
      2. Align each model output to the reference using word-level MED.
      3. At each reference position, collect all model alternatives.
      4. If a model word at position i differs from the reference but 
         is produced by ≥ threshold fraction of models, add it to bin i.
      5. Always keep the human reference in the bin.
    
    Args:
        reference: Human reference transcription.
        model_outputs: Dict of model_name -> model transcription.
        consensus_threshold: Fraction of models needed to add an alternative.
        
    Returns:
        List of sets (bins), one per reference word position.
    """
    ref_words = _clean_for_alignment(reference)
    n_models = len(model_outputs)
    min_agreement = max(1, int(np.ceil(n_models * consensus_threshold)))
    
    # Initialize lattice with reference words
    lattice = [set() for _ in ref_words]
    for i, word in enumerate(ref_words):
        lattice[i].add(word)
    
    # Collect alternatives from each model at each position
    position_candidates = defaultdict(Counter)
    
    for model_name, model_text in model_outputs.items():
        model_words = _clean_for_alignment(model_text)
        alignment = _word_level_alignment(ref_words, model_words)
        
        for op, ref_pos, hyp_pos in alignment:
            if op in ('M', 'S') and ref_pos is not None and hyp_pos is not None:
                candidate = model_words[hyp_pos]
                position_candidates[ref_pos][candidate] += 1
    
    # Apply consensus rule to expand bins
    for pos, candidates in position_candidates.items():
        if pos < len(lattice):
            for word, count in candidates.items():
                if count >= min_agreement:
                    lattice[pos].add(word)
    
    return lattice


def construct_lattice_with_synonyms(
    reference: str,
    model_outputs: Dict[str, str],
    synonym_map: Optional[Dict[str, Set[str]]] = None,
    number_equivalences: bool = True
) -> List[Set[str]]:
    """
    Enhanced lattice construction with linguistic knowledge.
    
    Adds:
      - Synonym expansion (if provided)
      - Number word ↔ digit equivalences
      - Common spelling variant pairs
    """
    lattice = construct_lattice(reference, model_outputs)
    
    # Number equivalences
    if number_equivalences:
        number_map = {
            '1': {'एक'}, '2': {'दो'}, '3': {'तीन'}, '4': {'चार'},
            '5': {'पांच', 'पाँच'}, '6': {'छह', 'छे'}, '7': {'सात'},
            '8': {'आठ'}, '9': {'नौ'}, '10': {'दस'}, '11': {'ग्यारह'},
            '12': {'बारह'}, '13': {'तेरह'}, '14': {'चौदह'}, '15': {'पंद्रह'},
            '20': {'बीस'}, '25': {'पच्चीस'}, '50': {'पचास'}, '100': {'सौ'},
        }
        # Reverse mapping
        word_to_digit = {}
        for digit, words in number_map.items():
            for w in words:
                word_to_digit[w] = digit
        
        for i, bin_set in enumerate(lattice):
            expanded = set(bin_set)
            for word in bin_set:
                # If it's a digit, add Hindi words
                if word in number_map:
                    expanded.update(number_map[word])
                # If it's a Hindi number word, add the digit
                if word in word_to_digit:
                    expanded.add(word_to_digit[word])
            lattice[i] = expanded
    
    # Synonym expansion
    if synonym_map:
        for i, bin_set in enumerate(lattice):
            expanded = set(bin_set)
            for word in bin_set:
                if word in synonym_map:
                    expanded.update(synonym_map[word])
            lattice[i] = expanded
    
    # Common spelling variants
    spelling_variants = {
        'किताबें': {'किताबे', 'कताबें', 'कताबे'},
        'खरीदीं': {'खरीदी', 'ख़रीदीं'},
        'बिल्कुल': {'बिलकुल'},
        'ज़्यादा': {'ज्यादा', 'जियादा'},
        'रक्षाबंधन': {'रक्षा बंधन'},
        'खेतीबाड़ी': {'खेती बाड़ी'},
        'सियाराम': {'सिया राम', 'शिय राम'},
    }
    
    for i, bin_set in enumerate(lattice):
        expanded = set(bin_set)
        for word in bin_set:
            if word in spelling_variants:
                expanded.update(spelling_variants[word])
        lattice[i] = expanded
    
    return lattice


# ═══════════════════════════════════════════════════════════════════════════
# LATTICE-BASED WER COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_lattice_wer(
    model_output: str,
    lattice: List[Set[str]]
) -> float:
    """
    Compute WER using a lattice reference instead of a rigid string.
    
    Uses modified Levenshtein distance where substitution cost is 0
    if the predicted word appears in the corresponding lattice bin.
    
    Args:
        model_output: Model's transcription string.
        lattice: List of sets representing valid alternatives per position.
        
    Returns:
        WER as a ratio (0.0 to 1.0+).
    """
    pred_words = _clean_for_alignment(model_output)
    n = len(lattice)  # reference length
    m = len(pred_words)  # prediction length
    
    if n == 0:
        return 0.0 if m == 0 else float('inf')
    
    # DP table
    dp = np.zeros((n + 1, m + 1), dtype=float)
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            ref_bin = lattice[i - 1]
            pred_word = pred_words[j - 1]
            
            # Substitution cost: 0 if match any variant, 1 otherwise
            sub_cost = 0.0 if pred_word in ref_bin else 1.0
            
            dp[i][j] = min(
                dp[i-1][j] + 1.0,          # Deletion
                dp[i][j-1] + 1.0,          # Insertion  
                dp[i-1][j-1] + sub_cost    # Substitution (or free match)
            )
    
    return dp[n][m] / n


def compute_standard_wer(prediction: str, reference: str) -> float:
    """
    Compute standard (rigid) WER for comparison.
    """
    ref_words = _clean_for_alignment(reference)
    pred_words = _clean_for_alignment(prediction)
    
    n = len(ref_words)
    m = len(pred_words)
    
    if n == 0:
        return 0.0 if m == 0 else float('inf')
    
    dp = np.zeros((n + 1, m + 1), dtype=float)
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0.0 if ref_words[i-1] == pred_words[j-1] else 1.0
            dp[i][j] = min(
                dp[i-1][j] + 1.0,
                dp[i][j-1] + 1.0,
                dp[i-1][j-1] + cost
            )
    
    return dp[n][m] / n


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_all_models(csv_url: str = LATTICE_DATA_CSV_URL) -> pd.DataFrame:
    """
    Full evaluation pipeline:
      1. Load the Q4 dataset (Human + 6 model transcriptions)
      2. For each utterance, construct a lattice
      3. Compute both standard WER and lattice WER for each model
      4. Generate comparative report
    """
    logger.info("Loading Q4 lattice evaluation data...")
    df = pd.read_csv(csv_url)
    
    # Identify columns
    segment_col = df.columns[0]
    human_col = 'Human'
    model_cols = [c for c in df.columns if c not in [segment_col, human_col, '']]
    
    logger.info(f"Loaded {len(df)} utterances, {len(model_cols)} models")
    
    # Results storage
    results = {model: {'standard_wer': [], 'lattice_wer': []} for model in model_cols}
    
    for idx, row in df.iterrows():
        reference = str(row[human_col]).strip()
        if not reference or reference == 'nan':
            continue
        
        # Collect all model outputs
        model_outputs = {}
        for model in model_cols:
            output = str(row[model]).strip()
            if output and output != 'nan':
                model_outputs[model] = output
        
        if not model_outputs:
            continue
        
        # Construct lattice using all models + linguistic knowledge
        lattice = construct_lattice_with_synonyms(
            reference=reference,
            model_outputs=model_outputs,
            number_equivalences=True
        )
        
        # Compute WER for each model
        for model, output in model_outputs.items():
            std_wer = compute_standard_wer(output, reference)
            lat_wer = compute_lattice_wer(output, lattice)
            
            results[model]['standard_wer'].append(std_wer)
            results[model]['lattice_wer'].append(lat_wer)
    
    # Build summary table
    summary_rows = []
    for model in model_cols:
        if results[model]['standard_wer']:
            std_avg = np.mean(results[model]['standard_wer'])
            lat_avg = np.mean(results[model]['lattice_wer'])
            delta = std_avg - lat_avg
            
            summary_rows.append({
                'Model': model,
                'Standard WER': f"{std_avg:.4f}",
                'Lattice WER': f"{lat_avg:.4f}",
                'WER Reduction': f"{delta:.4f}",
                'Fairly Penalized?': '✓ Reduced' if delta > 0.001 else '— Unchanged'
            })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Log results
    logger.info("=" * 70)
    logger.info("LATTICE WER EVALUATION RESULTS")
    logger.info("=" * 70)
    for _, row in summary_df.iterrows():
        logger.info(
            f"  {row['Model']:12s}: Standard={row['Standard WER']}, "
            f"Lattice={row['Lattice WER']}, Δ={row['WER Reduction']} "
            f"({row['Fairly Penalized?']})"
        )
    logger.info("=" * 70)
    
    return summary_df


# ─── Report Generation ──────────────────────────────────────────────────────

def generate_q4_report(summary_df: pd.DataFrame) -> Path:
    """Generate the complete Q4 theory + results report."""
    report_path = REPORTS_DIR / "q4_lattice_theory.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Question 4: Lattice-based WER Evaluation\n\n")
        
        f.write("## 1. Alignment Unit Choice: Word\n\n")
        f.write("We choose **word-level** alignment for the following reasons:\n\n")
        f.write("| Unit | Pros | Cons | Verdict |\n")
        f.write("|------|------|------|---------|\n")
        f.write("| Character | Fine-grained | Lattice explodes (O(n×k) per char) | ✗ |\n")
        f.write("| Subword | Model-native | Tokenization inconsistent across models | ✗ |\n")
        f.write("| Word | Natural unit, tractable | May miss sub-word variations | ✓ |\n")
        f.write("| Phrase | Captures MWEs | Chunking ambiguity | ✗ |\n\n")
        
        f.write("## 2. Lattice Construction Algorithm\n\n")
        f.write("```\n")
        f.write("function CONSTRUCT_LATTICE(reference, model_outputs, threshold):\n")
        f.write("    ref_words = TOKENIZE(reference)\n")
        f.write("    lattice = [{word} for word in ref_words]  // Initialize with reference\n")
        f.write("    \n")
        f.write("    for each model in model_outputs:\n")
        f.write("        alignment = ALIGN(ref_words, model.words)  // MED alignment\n")
        f.write("        for each (ref_pos, model_word) in alignment:\n")
        f.write("            position_votes[ref_pos][model_word] += 1\n")
        f.write("    \n")
        f.write("    for each position in lattice:\n")
        f.write("        for each candidate, vote_count in position_votes[position]:\n")
        f.write("            if vote_count >= ceil(n_models * threshold):\n")
        f.write("                lattice[position].add(candidate)\n")
        f.write("        \n")
        f.write("        // Also add: number equivalences, spelling variants, synonyms\n")
        f.write("        EXPAND_WITH_LINGUISTIC_KNOWLEDGE(lattice[position])\n")
        f.write("    \n")
        f.write("    return lattice\n")
        f.write("```\n\n")
        
        f.write("## 3. Modified WER Computation\n\n")
        f.write("```\n")
        f.write("function LATTICE_WER(prediction, lattice):\n")
        f.write("    pred_words = TOKENIZE(prediction)\n")
        f.write("    n = len(lattice), m = len(pred_words)\n")
        f.write("    dp[0..n][0..m] = standard MED initialization\n")
        f.write("    \n")
        f.write("    for i in 1..n:\n")
        f.write("        for j in 1..m:\n")
        f.write("            sub_cost = 0 if pred_words[j] IN lattice[i] else 1\n")
        f.write("            dp[i][j] = min(\n")
        f.write("                dp[i-1][j] + 1,       // deletion\n")
        f.write("                dp[i][j-1] + 1,       // insertion\n")
        f.write("                dp[i-1][j-1] + sub_cost  // sub/match\n")
        f.write("            )\n")
        f.write("    \n")
        f.write("    return dp[n][m] / n\n")
        f.write("```\n\n")
        
        f.write("## 4. Consensus Trust Rule\n\n")
        f.write("If ≥ ⌈M/2⌉ models agree on a word W at position i, and W differs ")
        f.write("from the human reference, then W is added to the lattice bin at i. ")
        f.write("The human reference is **never removed** — only expanded. This allows ")
        f.write("the system to tolerate human annotation errors without overriding ")
        f.write("the reference entirely.\n\n")
        
        f.write("## 5. Results\n\n")
        if len(summary_df) > 0:
            f.write("| Model | Standard WER | Lattice WER | WER Reduction | Status |\n")
            f.write("|-------|-------------|-------------|---------------|--------|\n")
            for _, row in summary_df.iterrows():
                f.write(
                    f"| {row['Model']} | {row['Standard WER']} | "
                    f"{row['Lattice WER']} | {row['WER Reduction']} | "
                    f"{row['Fairly Penalized?']} |\n"
                )
            f.write("\n")
            f.write("Models that were unfairly penalized show reduced WER under "
                    "lattice evaluation. Models that made genuine errors show "
                    "unchanged or minimally changed WER.\n")
    
    logger.info(f"Q4 report saved to {report_path}")
    return report_path


# ─── Demo ────────────────────────────────────────────────────────────────────

def run_demo():
    """Run the example from the assignment."""
    print("=" * 70)
    print("LATTICE WER DEMO")
    print("=" * 70)
    
    # Example from the assignment
    reference = "उसने चौदह किताबें खरीदीं"
    
    models = {
        'Model_A': "उसने 14 पुस्तकें खरीदी",
        'Model_B': "उसने चौदह किताबे खरीदीं",
        'Model_C': "उसने चौदह किताबें खरीदी",
    }
    
    print(f"\nReference: {reference}")
    print(f"Models:")
    for name, output in models.items():
        print(f"  {name}: {output}")
    
    # Construct lattice
    lattice = construct_lattice_with_synonyms(
        reference, models, number_equivalences=True
    )
    
    print(f"\nConstructed Lattice:")
    ref_words = _clean_for_alignment(reference)
    for i, (word, bin_set) in enumerate(zip(ref_words, lattice)):
        print(f"  Position {i} ({word}): {bin_set}")
    
    # Compute WER
    print(f"\nWER Comparison:")
    print(f"{'Model':15s} {'Standard WER':>15s} {'Lattice WER':>15s} {'Reduction':>12s}")
    print("-" * 58)
    
    for name, output in models.items():
        std = compute_standard_wer(output, reference)
        lat = compute_lattice_wer(output, lattice)
        delta = std - lat
        print(f"{name:15s} {std:15.2%} {lat:15.2%} {delta:12.2%}")


if __name__ == "__main__":
    run_demo()
    
    print("\n\n")
    print("=" * 70)
    print("Running full evaluation on Q4 dataset...")
    print("=" * 70)
    
    try:
        summary = evaluate_all_models()
        generate_q4_report(summary)
    except Exception as e:
        logger.error(f"Full evaluation failed: {e}")
        logger.info("This is expected if running without internet access.")
