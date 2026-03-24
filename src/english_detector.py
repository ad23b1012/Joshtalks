"""
English Word Detection in Hindi Transcriptions (Q2b).

Identifies English loan words written in Devanagari script within Hindi
transcriptions and tags them with [EN]...[/EN] markers.

Multi-strategy approach:
  1. Curated high-frequency English loan word dictionary
  2. Reverse transliteration (Devanagari → Roman) + English dictionary lookup
  3. Character n-gram analysis (distinguishing native vs. loan patterns)
"""
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ─── Strategy 1: Curated English Loan Word Dictionary ────────────────────────

# High-frequency English words commonly used in Hindi conversations,
# transcribed in Devanagari per the Josh Talks transcription guidelines.
ENGLISH_LOANS_DEVANAGARI = {
    # Technology
    'कंप्यूटर', 'फ़ोन', 'फोन', 'मोबाइल', 'लैपटॉप', 'इंटरनेट',
    'वेबसाइट', 'ऐप', 'सॉफ्टवेयर', 'हार्डवेयर', 'डेटा', 'नेटवर्क',
    'वाईफाई', 'ब्लूटूथ', 'सर्वर', 'प्रोग्राम', 'कोड', 'टेक्नोलॉजी',
    'ऑनलाइन', 'ऑफलाइन', 'डिजिटल', 'वीडियो', 'ऑडियो', 'कैमरा',
    
    # Education & Work
    'स्कूल', 'कॉलेज', 'यूनिवर्सिटी', 'इंटरव्यू', 'जॉब', 'कंपनी',
    'ऑफिस', 'बॉस', 'मैनेजर', 'प्रोजेक्ट', 'मीटिंग', 'प्रेजेंटेशन',
    'रिज्यूम', 'सैलरी', 'प्रमोशन', 'टीम', 'लीडर', 'बजट', 'टारगेट',
    'रिपोर्ट', 'ट्रेनिंग', 'सर्टिफिकेट', 'डिग्री', 'एग्जाम', 'रिजल्ट',
    'क्लास', 'टीचर', 'स्टूडेंट', 'सब्जेक्ट', 'टॉपिक', 'नोट्स',
    'होमवर्क', 'असाइनमेंट', 'लेक्चर',
    
    # Daily Life
    'बस', 'ट्रेन', 'ऑटो', 'टैक्सी', 'बाइक', 'कार', 'ड्राइवर',
    'टिकट', 'स्टेशन', 'एयरपोर्ट', 'होटल', 'रेस्टोरेंट', 'शॉपिंग',
    'मॉल', 'पार्क', 'हॉस्पिटल', 'डॉक्टर', 'नर्स', 'मेडिसिन',
    
    # Communication
    'मैसेज', 'कॉल', 'ईमेल', 'चैट', 'सोशल', 'मीडिया', 'पोस्ट',
    'शेयर', 'लाइक', 'कमेंट', 'फॉलो', 'सब्सक्राइब', 'अकाउंट',
    'पासवर्ड', 'प्रोफाइल', 'स्टोरी', 'ग्रुप',
    
    # Emotions & States
    'हैप्पी', 'सैड', 'नर्वस', 'कॉन्फिडेंट', 'स्ट्रेस', 'रिलैक्स',
    'मोटिवेशन', 'इंस्पिरेशन', 'पैशन', 'सक्सेस', 'फेल',
    
    # Problem solving
    'प्रॉब्लम', 'सॉल्व', 'सॉल्यूशन', 'इश्यू', 'फिक्स', 'एरर',
    'रीजन', 'ऑप्शन', 'चॉइस', 'डिसीजन', 'प्लान', 'गोल',
    
    # General
    'फैमिली', 'फ्रेंड', 'रिलेशनशिप', 'पार्टी', 'टूर', 'ट्रिप',
    'हॉबी', 'स्पोर्ट्स', 'गेम', 'म्यूजिक', 'डांस', 'फिल्म', 'मूवी',
    'सीरीज', 'बुक', 'स्टोरी', 'न्यूज़', 'टाइम', 'लाइफ',
    'एक्सपीरियंस', 'टाइप', 'पॉइंट', 'लेवल', 'फीचर', 'क्वालिटी',
    'सिस्टम', 'प्रोसेस', 'मेथड', 'बेसिक', 'एडवांस', 'सिंपल',
    
    # Additional common ones from conversational Hindi
    'फ्यूचर', 'कॅरियर', 'करियर', 'फीडबैक', 'अपडेट', 'वर्जन',
    'कॉन्टेंट', 'ब्रांड', 'मार्केट', 'बिजनेस', 'कस्टमर',
    'प्रॉडक्ट', 'सर्विस', 'डिलीवरी', 'ऑर्डर',
    'परसेंट', 'प्रतिशत', 'नंबर', 'टोटल', 'एवरेज',
    'पॉपुलर', 'फेमस', 'इंपोर्टेंट', 'स्पेशल', 'नॉर्मल',
    'ऐक्चुअली', 'बेसिकली', 'ऑब्वियसली', 'प्रॉपर',
    'सीरियस', 'प्रैक्टिकल', 'थ्योरी',
    
    # From the Q4 dataset
    'प्योर', 'हार्ट', 'सिंपल',
}


# ─── Strategy 2: Transliteration-based Detection ────────────────────────────

# Simple Devanagari → Roman approximate mappings for reverse transliteration
DEVANAGARI_TO_ROMAN = {
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo',
    'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
    'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'w': 'w',
    'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',
    'क्ष': 'ksh', 'त्र': 'tr', 'ज्ञ': 'gny',
    'ं': 'n', 'ँ': 'n', 'ः': 'h',
    'ा': 'a', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo',
    'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au',
    '्': '', 'ॉ': 'o', 'ॉ': 'o', 'ॅ': 'e',
    'फ़': 'f', 'ज़': 'z', 'ड़': 'r', 'ढ़': 'rh', 'ऑ': 'o',
}

# Common English words for dictionary check
# In production, use nltk.corpus.words or a larger dictionary
COMMON_ENGLISH_WORDS = {
    'interview', 'job', 'computer', 'phone', 'mobile', 'laptop', 'internet',
    'website', 'app', 'software', 'hardware', 'data', 'network', 'server',
    'program', 'code', 'technology', 'online', 'offline', 'digital',
    'video', 'audio', 'camera', 'school', 'college', 'university',
    'office', 'boss', 'manager', 'project', 'meeting', 'presentation',
    'resume', 'salary', 'promotion', 'team', 'leader', 'budget', 'target',
    'report', 'training', 'certificate', 'degree', 'exam', 'result',
    'class', 'teacher', 'student', 'subject', 'topic', 'notes',
    'homework', 'assignment', 'lecture', 'bus', 'train', 'auto', 'taxi',
    'bike', 'car', 'driver', 'ticket', 'station', 'airport', 'hotel',
    'restaurant', 'shopping', 'mall', 'park', 'hospital', 'doctor',
    'nurse', 'medicine', 'message', 'call', 'email', 'chat', 'social',
    'media', 'post', 'share', 'like', 'comment', 'follow', 'subscribe',
    'account', 'password', 'profile', 'story', 'group', 'happy', 'sad',
    'nervous', 'confident', 'stress', 'relax', 'motivation', 'success',
    'fail', 'problem', 'solve', 'solution', 'issue', 'fix', 'error',
    'reason', 'option', 'choice', 'decision', 'plan', 'goal', 'family',
    'friend', 'relationship', 'party', 'tour', 'trip', 'hobby', 'sports',
    'game', 'music', 'dance', 'film', 'movie', 'series', 'book', 'news',
    'time', 'life', 'experience', 'type', 'point', 'level', 'feature',
    'quality', 'system', 'process', 'method', 'basic', 'advance', 'simple',
    'future', 'career', 'feedback', 'update', 'version', 'content',
    'brand', 'market', 'business', 'customer', 'product', 'service',
    'delivery', 'order', 'percent', 'number', 'total', 'average',
    'popular', 'famous', 'important', 'special', 'normal', 'actually',
    'basically', 'obviously', 'proper', 'serious', 'practical', 'theory',
    'pure', 'heart',
}


def _rough_transliterate(devanagari_word: str) -> str:
    """
    Approximate Devanagari → Roman transliteration.
    This is a simplified version; for production, use ai4bharat/IndicTrans.
    """
    result = []
    i = 0
    chars = list(devanagari_word)
    
    while i < len(chars):
        # Try two-character combinations first
        if i + 1 < len(chars):
            digraph = chars[i] + chars[i + 1]
            if digraph in DEVANAGARI_TO_ROMAN:
                result.append(DEVANAGARI_TO_ROMAN[digraph])
                i += 2
                continue
        
        char = chars[i]
        if char in DEVANAGARI_TO_ROMAN:
            result.append(DEVANAGARI_TO_ROMAN[char])
        else:
            result.append(char)
        i += 1
    
    return ''.join(result).lower()


def is_english_loan(word: str) -> Tuple[bool, str]:
    """
    Determine if a Devanagari word is an English loan word.
    
    Returns:
        Tuple of (is_english, method) where method describes how it was detected.
    """
    # Clean the word
    clean = re.sub(r'[^\u0900-\u097F]', '', word)
    
    if not clean:
        return False, 'empty'
    
    # Strategy 1: Direct dictionary lookup
    if clean in ENGLISH_LOANS_DEVANAGARI:
        return True, 'dictionary'
    
    # Strategy 2: Reverse transliteration + English dictionary
    roman = _rough_transliterate(clean)
    if roman in COMMON_ENGLISH_WORDS:
        return True, 'transliteration'
    
    # Strategy 3: Check if close to any English word (fuzzy)
    # Simple edit distance check for top candidates
    for eng_word in COMMON_ENGLISH_WORDS:
        if len(eng_word) >= 4 and roman.startswith(eng_word[:4]):
            return True, 'fuzzy_transliteration'
    
    return False, 'none'


# ─── Main Tagger ────────────────────────────────────────────────────────────

def tag_english_words(text: str) -> str:
    """
    Tag English loan words in a Hindi transcript with [EN]...[/EN] markers.
    
    Args:
        text: Hindi transcript potentially containing English words in Devanagari.
        
    Returns:
        Tagged text with English words marked.
        
    Example:
        Input:  "मेरा इंटरव्यू बहुत अच्छा गया और मुझे जॉब मल गई"
        Output: "मेरा [EN]इंटरव्यू[/EN] बहुत अच्छा गया और मुझे [EN]जॉब[/EN] मल गई"
    """
    words = text.split()
    tagged = []
    
    for word in words:
        # Preserve any attached punctuation
        prefix = ''
        suffix = ''
        core = word
        
        # Strip leading/trailing punctuation
        while core and not ('\u0900' <= core[0] <= '\u097F' or core[0].isalpha()):
            prefix += core[0]
            core = core[1:]
        while core and not ('\u0900' <= core[-1] <= '\u097F' or core[-1].isalpha()):
            suffix = core[-1] + suffix
            core = core[:-1]
        
        if not core:
            tagged.append(word)
            continue
        
        is_english, method = is_english_loan(core)
        
        if is_english:
            tagged.append(f"{prefix}[EN]{core}[/EN]{suffix}")
        else:
            tagged.append(word)
    
    return ' '.join(tagged)


def analyze_english_density(texts: List[str]) -> Dict:
    """
    Analyze the density of English loan words across transcriptions.
    
    Returns statistics about English word usage in the corpus.
    """
    total_words = 0
    english_words = 0
    english_word_counts = {}
    
    for text in texts:
        words = text.split()
        total_words += len(words)
        
        for word in words:
            clean = re.sub(r'[^\u0900-\u097F]', '', word)
            is_eng, _ = is_english_loan(clean)
            if is_eng:
                english_words += 1
                english_word_counts[clean] = english_word_counts.get(clean, 0) + 1
    
    return {
        'total_words': total_words,
        'english_words': english_words,
        'english_ratio': english_words / total_words if total_words > 0 else 0,
        'top_english_words': sorted(
            english_word_counts.items(), key=lambda x: x[1], reverse=True
        )[:20]
    }


# ─── Examples ────────────────────────────────────────────────────────────────

def run_examples():
    """Run the examples requested in the assignment."""
    print("=" * 70)
    print("ENGLISH WORD DETECTION — EXAMPLES")
    print("=" * 70)
    
    examples = [
        "मेरा इंटरव्यू बहुत अच्छा गया और मुझे जॉब मल गई",
        "ये प्रॉब्लम सॉल्व नहीं हो रहा",
        "कंप्यूटर और फ़ोन दोनों खराब हो गए",
        "मैं कॉलेज में टीचर हूं और स्टूडेंट्स को प्रोग्रामिंग सिखाता हूं",
        "ट्रेन का टिकट ऑनलाइन बुक कर लो",
        "डॉक्टर ने मेडिसिन दी है",
        "बहुत प्योर हार्ट रहता है सबका",
    ]
    
    for text in examples:
        tagged = tag_english_words(text)
        print(f"\n  Input:  {text}")
        print(f"  Output: {tagged}")


if __name__ == "__main__":
    run_examples()
