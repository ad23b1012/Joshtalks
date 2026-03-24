"""
Hindi Number Normalizer — Convert spoken Hindi number words to digits (Q2a).

Handles:
  - Simple cases: दो → 2, दस → 10, सौ → 100
  - Compound numbers: तीन सौ चौवन → 354, पच्चीस → 25, एक हज़ार → 1000
  - Edge cases: Idioms/phrases where conversion would be semantically wrong
  - Complex compounds: दो लाख तीन हज़ार चार सौ पच्चीस → 203425
"""
import re
import logging
from typing import List, Tuple, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ─── Complete Hindi Number Dictionary ────────────────────────────────────────

# Individual numbers 0-99
UNITS = {
    'शून्य': 0, 'एक': 1, 'दो': 2, 'तीन': 3, 'चार': 4,
    'पांच': 5, 'पाँच': 5, 'छह': 6, 'छे': 6, 'छै': 6,
    'सात': 7, 'आठ': 8, 'नौ': 9, 'नो': 9,
}

TEENS_AND_TENS = {
    'दस': 10, 'ग्यारह': 11, 'बारह': 12, 'तेरह': 13, 'चौदह': 14,
    'पंद्रह': 15, 'सोलह': 16, 'सत्रह': 17, 'अठारह': 18, 'उन्नीस': 19,
    'बीस': 20, 'इक्कीस': 21, 'बाईस': 22, 'तेईस': 23, 'चौबीस': 24,
    'पच्चीस': 25, 'छब्बीस': 26, 'सत्ताईस': 27, 'अट्ठाईस': 28,
    'उनतीस': 29, 'तीस': 30, 'इकतीस': 31, 'बत्तीस': 32, 'तैंतीस': 33,
    'चौंतीस': 34, 'पैंतीस': 35, 'छत्तीस': 36, 'सैंतीस': 37,
    'अड़तीस': 38, 'उनतालीस': 39, 'चालीस': 40, 'इकतालीस': 41,
    'बयालीस': 42, 'तैंतालीस': 43, 'चवालीस': 44, 'पैंतालीस': 45,
    'छियालीस': 46, 'सैंतालीस': 47, 'अड़तालीस': 48, 'उनचास': 49,
    'पचास': 50, 'इक्यावन': 51, 'बावन': 52, 'तिरपन': 53, 'चौवन': 54,
    'पचपन': 55, 'छप्पन': 56, 'सत्तावन': 57, 'अट्ठावन': 58,
    'उनसठ': 59, 'साठ': 60, 'इकसठ': 61, 'बासठ': 62, 'तिरसठ': 63,
    'चौंसठ': 64, 'पैंसठ': 65, 'छियासठ': 66, 'सड़सठ': 67,
    'अड़सठ': 68, 'उनहत्तर': 69, 'सत्तर': 70, 'इकहत्तर': 71,
    'बहत्तर': 72, 'तिहत्तर': 73, 'चौहत्तर': 74, 'पचहत्तर': 75,
    'छिहत्तर': 76, 'सतहत्तर': 77, 'अठहत्तर': 78, 'उन्यासी': 79,
    'अस्सी': 80, 'इक्यासी': 81, 'बयासी': 82, 'तिरासी': 83,
    'चौरासी': 84, 'पचासी': 85, 'छियासी': 86, 'सत्तासी': 87,
    'अट्ठासी': 88, 'नवासी': 89, 'नब्बे': 90, 'इक्यानवे': 91,
    'बानवे': 92, 'तिरानवे': 93, 'चौरानवे': 94, 'पचानवे': 95,
    'छियानवे': 96, 'सत्तानवे': 97, 'अट्ठानवे': 98, 'निन्यानवे': 99,
}

# Multipliers (Indian number system)
MULTIPLIERS = {
    'सौ': 100,
    'हज़ार': 1000, 'हजार': 1000, 'हज़ारों': 1000,
    'लाख': 100000, 'लाखों': 100000,
    'करोड़': 10000000, 'करोड़ों': 10000000,
    'अरब': 1000000000,
}

# Merged lookup for all valid number words
ALL_NUMBER_WORDS = {}
ALL_NUMBER_WORDS.update(UNITS)
ALL_NUMBER_WORDS.update(TEENS_AND_TENS)
ALL_NUMBER_WORDS.update(MULTIPLIERS)


# ─── Idiom / Phrase Exception List ───────────────────────────────────────────

IDIOM_PATTERNS = [
    # Exact idiom phrases to skip
    r'दो-चार\s+बातें',
    r'दो\s+चार\s+बातें',
    r'नौ\s+दो\s+ग्यारह',
    r'एक\s+ना\s+एक\s+दिन',
    r'एक\s+न\s+एक',
    r'चार\s+चांद',
    r'दो\s+टूक',
    r'तीन\s+तेरह',
    r'सात\s+समंदर',
    r'चार\s+धाम',
    r'नौ\s+रस',
    r'छत्तीस\s+का\s+आंकड़ा',
    r'एक\s+दूसरे',
    r'एक-दूसरे',
    r'एक\s+साथ',
    r'एक\s+बार',
    r'एक\s+तरह',
    r'एक\s+तरफ',
    r'दो\s+तरफा',
]

# Pluralized/approximate forms that shouldn't be converted
SKIP_SUFFIXED = {
    'हजारों', 'लाखों', 'करोड़ों', 'सैकड़ों', 'बीसियों', 'दर्जनों',
}


# ─── Core Number Parser ─────────────────────────────────────────────────────

def _evaluate_number_sequence(words: List[str]) -> int:
    """
    Evaluate a sequence of Hindi number words into a single integer.
    
    Follows Indian number system place-value logic:
      - Small numbers (0-99) are added to a running accumulator
      - Multipliers (सौ, हज़ार, लाख, करोड़) multiply the accumulator
      - Higher multipliers flush the total
      
    Examples:
      ['तीन', 'सौ', 'चौवन'] → 354
      ['दो', 'लाख', 'तीन', 'हज़ार', 'चार', 'सौ', 'पच्चीस'] → 203425
      ['एक', 'हज़ार'] → 1000
    """
    total = 0
    current = 0
    
    for word in words:
        value = ALL_NUMBER_WORDS.get(word)
        
        if value is None:
            continue
        
        if value in (100, 1000, 100000, 10000000, 1000000000):
            # Multiplier
            if current == 0:
                current = 1  # "सौ" alone = 100
            current *= value
            
            # If this is a "higher tier" multiplier, flush to total
            if value >= 1000:
                total += current
                current = 0
        else:
            # Unit or compound number (0-99)
            current += value
    
    total += current
    return total


def _is_number_word(word: str) -> bool:
    """Check if a word is a recognized Hindi number word."""
    return word in ALL_NUMBER_WORDS


def _find_idiom_spans(text: str) -> List[Tuple[int, int]]:
    """
    Find character spans in the text that match known idiom patterns.
    These spans should NOT be number-normalized.
    """
    protected_spans = []
    for pattern in IDIOM_PATTERNS:
        for match in re.finditer(pattern, text):
            protected_spans.append((match.start(), match.end()))
    return protected_spans


def _is_position_protected(pos: int, protected_spans: List[Tuple[int, int]]) -> bool:
    """Check if a character position falls within a protected idiom span."""
    for start, end in protected_spans:
        if start <= pos < end:
            return True
    return False


# ─── Main Normalizer ────────────────────────────────────────────────────────

def normalize_hindi_numbers(text: str) -> str:
    """
    Convert spoken Hindi number words into digits.
    
    Handles:
      - Simple: दो → 2, दस → 10, सौ → 100
      - Compound: तीन सौ चौवन → 354, पच्चीस → 25
      - Complex: दो लाख तीन हज़ार → 203000
      - Edge cases: Idioms preserved (दो-चार बातें stays as-is)
      
    Args:
        text: Hindi text potentially containing number words.
        
    Returns:
        Text with number words replaced by digits.
    """
    if not text:
        return text
    
    # Step 1: Find and protect idiom spans
    protected_spans = _find_idiom_spans(text)
    
    # Step 2: Build the regex pattern for all number words
    # Sort by length (longest first) to avoid partial matches
    all_words = sorted(ALL_NUMBER_WORDS.keys(), key=len, reverse=True)
    
    # Pattern matches one or more consecutive number words
    word_pattern = '|'.join(re.escape(w) for w in all_words)
    full_pattern = rf'(?:(?:{word_pattern})(?:\s+(?:{word_pattern}))*)'
    
    def replace_match(match):
        # Check if this match falls within a protected span
        if _is_position_protected(match.start(), protected_spans):
            return match.group()
        
        matched_text = match.group()
        words = matched_text.split()
        
        # Filter to only actual number words
        num_words = [w for w in words if _is_number_word(w)]
        
        if not num_words:
            return matched_text
        
        # Skip pluralized/approximate forms
        if any(w in SKIP_SUFFIXED for w in num_words):
            return matched_text
        
        # Evaluate the number
        value = _evaluate_number_sequence(num_words)
        
        return str(value)
    
    result = re.sub(full_pattern, replace_match, text)
    
    return result


# ─── Examples & Tests ────────────────────────────────────────────────────────

def run_examples():
    """Run the examples requested in the assignment."""
    print("=" * 70)
    print("HINDI NUMBER NORMALIZATION — EXAMPLES")
    print("=" * 70)
    
    # 4-5 correct conversion examples
    correct_examples = [
        ("मेरी उम्र पच्चीस साल है", "Simple: पच्चीस → 25"),
        ("तीन सौ चौवन लोग आए", "Compound: तीन सौ चौवन → 354"),
        ("एक हज़ार रुपये दो", "Multiplier: एक हज़ार → 1000"),
        ("सोलह दिन बाद मिलेंगे", "Teen: सोलह → 16"),
        ("दो लाख तीन हज़ार चार सौ पच्चीस", "Complex: दो लाख तीन हज़ार चार सौ पच्चीस → 203425"),
    ]
    
    print("\n### Correct Conversions:\n")
    for text, desc in correct_examples:
        result = normalize_hindi_numbers(text)
        print(f"  [{desc}]")
        print(f"    Before: {text}")
        print(f"    After:  {result}")
        print()
    
    # 2-3 edge cases
    edge_cases = [
        (
            "दो-चार बातें करनी हैं",
            "IDIOM — 'दो-चार बातें' means 'a few things', NOT '2-4 things'. "
            "Converting would destroy the idiomatic meaning."
        ),
        (
            "नौ दो ग्यारह हो गए",
            "IDIOM — 'नौ दो ग्यारह' means 'to flee/disappear', NOT '9 2 11'. "
            "It's a Hindi saying with no numerical meaning."
        ),
        (
            "हजारों लोग वहां थे",
            "APPROXIMATE — 'हजारों' (thousands) is an indefinite plural, "
            "not the number 1000. Converting would lose the semantic nuance."
        ),
    ]
    
    print("### Edge Cases (preserved correctly):\n")
    for text, reasoning in edge_cases:
        result = normalize_hindi_numbers(text)
        print(f"  Before: {text}")
        print(f"  After:  {result}")
        print(f"  Reasoning: {reasoning}")
        print()


if __name__ == "__main__":
    run_examples()
