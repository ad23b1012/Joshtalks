# Question 3: Spelling Classification Report

## 3a. Approach

We used a **multi-layer hybrid approach** combining:

1. **Hindi Dictionary Lookup** — Core vocabulary of common Hindi words
2. **English Loan Word Detection** — Transliteration-based check
3. **Morphological Analysis** — Valid suffix/prefix derivation
4. **Character Pattern Validation** — Devanagari sequence rules

### Results Summary

| Metric | Count |
|--------|-------|
| Total unique words | 177,421 |
| Correctly spelled | 36,352 |
| Incorrectly spelled | 141,069 |

## 3b. Confidence Scoring

| Confidence | Count | % |
|------------|-------|---|
| high | 4,406 | 2.5% |
| medium | 856 | 0.5% |
| low | 172,159 | 97.0% |

## 3c. Low Confidence Review

Reviewed 50 words from the low-confidence bucket.

| Word | Classification | Reason |
|------|---------------|--------|
| कहलो | incorrect | Not found in any dictionary or pattern |
| कैराम | incorrect | Not found in any dictionary or pattern |
| माध्यम। | incorrect | Not found in any dictionary or pattern |
| वोकेलबरी | incorrect | Not found in any dictionary or pattern |
| सॉल्टी | incorrect | Not found in any dictionary or pattern |
| मैलापन | correct | Has valid Hindi suffix but root not in dictionary |
| मेरी-मेरी | incorrect | Not found in any dictionary or pattern |
| सब्र | incorrect | Not found in any dictionary or pattern |
| ओटीटी | incorrect | Not found in any dictionary or pattern |
| वोहा | incorrect | Not found in any dictionary or pattern |
| व्यक्तिकत | incorrect | Not found in any dictionary or pattern |
| लगाई। | incorrect | Not found in any dictionary or pattern |
| मेंलेवर | incorrect | Not found in any dictionary or pattern |
| अनुभुवो | incorrect | Not found in any dictionary or pattern |
| हैयह | incorrect | Not found in any dictionary or pattern |
| वैकेनिक | incorrect | Not found in any dictionary or pattern |
| अक्वार्ड | incorrect | Not found in any dictionary or pattern |
| बटकाता | correct | Has valid Hindi suffix but root not in dictionary |
| एक्सटिमेंट | incorrect | Not found in any dictionary or pattern |
| दयत | incorrect | Not found in any dictionary or pattern |

## 3d. Unreliable Word Categories

### Proper Nouns (Names, Places)

Proper nouns like person names (सुभाष, अरुणिमा), place names (लखनऊ, चंडीगढ़), and organization names are not in standard dictionaries. The system incorrectly flags many valid proper nouns as misspelled. These require a separate Named Entity Recognition pass or gazetteer.

**Examples**: सुभाष, अरुणिमा, लखनऊ, चंडीगढ़

### Regional/Dialectal Variations

Hindi has significant dialectal variation (Bhojpuri, Braj, Awadhi, Rajasthani influences). Words like "बोल्" instead of "बोलो" or "रोटी" spelled as "रोटि" are dialectally correct but flagged as errors. The system is trained on standard Hindi and cannot distinguish valid dialectal forms from actual spelling mistakes.

**Examples**: बोल्यो, करवे, जावे, खावे

