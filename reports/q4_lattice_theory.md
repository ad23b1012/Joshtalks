# Question 4: Lattice-based WER Evaluation

## 1. Alignment Unit Choice: Word

We choose **word-level** alignment for the following reasons:

| Unit | Pros | Cons | Verdict |
|------|------|------|---------|
| Character | Fine-grained | Lattice explodes (O(n×k) per char) | ✗ |
| Subword | Model-native | Tokenization inconsistent across models | ✗ |
| Word | Natural unit, tractable | May miss sub-word variations | ✓ |
| Phrase | Captures MWEs | Chunking ambiguity | ✗ |

## 2. Lattice Construction Algorithm

```
function CONSTRUCT_LATTICE(reference, model_outputs, threshold):
    ref_words = TOKENIZE(reference)
    lattice = [{word} for word in ref_words]  // Initialize with reference
    
    for each model in model_outputs:
        alignment = ALIGN(ref_words, model.words)  // MED alignment
        for each (ref_pos, model_word) in alignment:
            position_votes[ref_pos][model_word] += 1
    
    for each position in lattice:
        for each candidate, vote_count in position_votes[position]:
            if vote_count >= ceil(n_models * threshold):
                lattice[position].add(candidate)
        
        // Also add: number equivalences, spelling variants, synonyms
        EXPAND_WITH_LINGUISTIC_KNOWLEDGE(lattice[position])
    
    return lattice
```

## 3. Modified WER Computation

```
function LATTICE_WER(prediction, lattice):
    pred_words = TOKENIZE(prediction)
    n = len(lattice), m = len(pred_words)
    dp[0..n][0..m] = standard MED initialization
    
    for i in 1..n:
        for j in 1..m:
            sub_cost = 0 if pred_words[j] IN lattice[i] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,       // deletion
                dp[i][j-1] + 1,       // insertion
                dp[i-1][j-1] + sub_cost  // sub/match
            )
    
    return dp[n][m] / n
```

## 4. Consensus Trust Rule

If ≥ ⌈M/2⌉ models agree on a word W at position i, and W differs from the human reference, then W is added to the lattice bin at i. The human reference is **never removed** — only expanded. This allows the system to tolerate human annotation errors without overriding the reference entirely.

## 5. Results

| Model | Standard WER | Lattice WER | WER Reduction | Status |
|-------|-------------|-------------|---------------|--------|
| Model H | 0.0331 | 0.0252 | 0.0079 | ✓ Reduced |
| Model i | 0.0061 | 0.0061 | 0.0000 | — Unchanged |
| Model k | 0.1060 | 0.0931 | 0.0129 | ✓ Reduced |
| Model l | 0.1066 | 0.1002 | 0.0064 | ✓ Reduced |
| Model m | 0.2012 | 0.1810 | 0.0202 | ✓ Reduced |
| Model n | 0.1073 | 0.0884 | 0.0189 | ✓ Reduced |

Models that were unfairly penalized show reduced WER under lattice evaluation. Models that made genuine errors show unchanged or minimally changed WER.
