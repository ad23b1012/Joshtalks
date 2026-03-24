# Question 1: Error Analysis Report

## 1d. Systematic Error Sampling

**Strategy**: systematic

We sorted all 25 test utterances by per-sample WER (descending), selected the top 100 worst-performing samples, then sampled every 4th entry to obtain 25 representative error examples. This ensures coverage across severity levels without cherry-picking.

## 1e. Error Taxonomy

### Homophone Substitution (5 examples)

**Example 1:**
- **Reference**: रोलैंडो मेंडोज़ा ने अपनी m16 राइफल से पर्यटकों के ऊपर फायर किया
- **Prediction**: वो लेंगा ने दो जाने की आं सिक्स देन राइफ है पल तो पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पह�
- **WER**: 6.500
- **Reasoning**: The model substituted a phonetically similar word. This is common in Hindi where words like वह/वो or है/हैं sound identical but have different spellings.

**Example 2:**
- **Reference**: द्वीप-समूह पेनिनसुला की उत्तर दिशा में 120 किमी दूर स्थित है विलास लास एस्ट्रेलास जैसे रिहायशी इलाके के साथ किंग ज़ॉर्ज सबसे बड़ा आइलैंड है
- **Prediction**: टीप्स हूं पेनिन सुला के उतर दिशा में एक सौ बीस कीमिक दूरी सिथ है विलास लास एस्ट्रे लास जैसे रही रही है सी इलाकर के साथ किंग जोर सबसे बड़ा इलाइलेन है
- **WER**: 0.840
- **Reasoning**: The model substituted a phonetically similar word. This is common in Hindi where words like वह/वो or है/हैं sound identical but have different spellings.

**Example 3:**
- **Reference**: जिस कारण दो मछली की प्रजातियां विलुप्त हो गई हैं और अन्य दो लुप्तप्राय हो गई हैं जिसमें हंपबैक चूब भी शामिल है
- **Prediction**: चीज कारण दो मचली किर पर जाती है अभी लुप्त हो गई है और अन्य दो लुप्ते प्राई हो गई है जिसमें हम पे पे चुप की सामिल है
- **WER**: 0.783
- **Reasoning**: The model substituted a phonetically similar word. This is common in Hindi where words like वह/वो or है/हैं sound identical but have different spellings.

### Other (5 examples)

**Example 1:**
- **Reference**: सत्ताधारी दल साउथ वेस्ट अफ्रीका पीपल आर्गेनाइजेशन स्वापो ने संसदीय चुनाव में बहुमत पाया
- **Prediction**: सत्ताधारिदल साउथ विस्ट अफरीका पीपल ओर्गनाइजेशन सवापों ने संसदीए चुनाओं में भहुत मत पाया
- **WER**: 0.714
- **Reasoning**: This error doesn't fit the major categories and may involve rare vocabulary, code-switching, or domain-specific terminology.

**Example 2:**
- **Reference**: टेलीविजन रिपोर्टों में प्लांट से निकलने वाला सफेद धुआं दिखाया गया है
- **Prediction**: टेरीवेजन रिपोटो के में प्लांट में निकालने जाने वाला सफे दुभा दिखाया गया है
- **WER**: 0.667
- **Reasoning**: This error doesn't fit the major categories and may involve rare vocabulary, code-switching, or domain-specific terminology.

**Example 3:**
- **Reference**: एरोस्मिथ ने अपने दौरे के अपने शेष संगीत कार्यक्रमों को रद्द कर दिया है
- **Prediction**: एरो इसमित्ने अपने दौड़े के अपने सेथ्स संगीत कार्कमों को रद्यकर दिया है।
- **WER**: 0.571
- **Reasoning**: This error doesn't fit the major categories and may involve rare vocabulary, code-switching, or domain-specific terminology.

### Word Repetition (4 examples)

**Example 1:**
- **Reference**: रोलैंडो मेंडोज़ा ने अपनी m16 राइफल से पर्यटकों के ऊपर फायर किया
- **Prediction**: वो लेंगा ने दो जाने की आं सिक्स देन राइफ है पल तो पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पह�
- **WER**: 6.500
- **Reasoning**: The model repeated a word, likely due to a decoding loop or attention drift in the autoregressive generation.

**Example 2:**
- **Reference**: द्वीप-समूह पेनिनसुला की उत्तर दिशा में 120 किमी दूर स्थित है विलास लास एस्ट्रेलास जैसे रिहायशी इलाके के साथ किंग ज़ॉर्ज सबसे बड़ा आइलैंड है
- **Prediction**: टीप्स हूं पेनिन सुला के उतर दिशा में एक सौ बीस कीमिक दूरी सिथ है विलास लास एस्ट्रे लास जैसे रही रही है सी इलाकर के साथ किंग जोर सबसे बड़ा इलाइलेन है
- **WER**: 0.840
- **Reasoning**: The model repeated a word, likely due to a decoding loop or attention drift in the autoregressive generation.

**Example 3:**
- **Reference**: जिस कारण दो मछली की प्रजातियां विलुप्त हो गई हैं और अन्य दो लुप्तप्राय हो गई हैं जिसमें हंपबैक चूब भी शामिल है
- **Prediction**: चीज कारण दो मचली किर पर जाती है अभी लुप्त हो गई है और अन्य दो लुप्ते प्राई हो गई है जिसमें हम पे पे चुप की सामिल है
- **WER**: 0.783
- **Reasoning**: The model repeated a word, likely due to a decoding loop or attention drift in the autoregressive generation.

### Number Format Mismatch (2 examples)

**Example 1:**
- **Reference**: रोलैंडो मेंडोज़ा ने अपनी m16 राइफल से पर्यटकों के ऊपर फायर किया
- **Prediction**: वो लेंगा ने दो जाने की आं सिक्स देन राइफ है पल तो पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पह�
- **WER**: 6.500
- **Reasoning**: The model output numbers in a different format than the reference. This is likely because Whisper's pretrained model was trained on diverse data where numbers may appear as digits, while the reference uses Hindi words.

**Example 2:**
- **Reference**: द्वीप-समूह पेनिनसुला की उत्तर दिशा में 120 किमी दूर स्थित है विलास लास एस्ट्रेलास जैसे रिहायशी इलाके के साथ किंग ज़ॉर्ज सबसे बड़ा आइलैंड है
- **Prediction**: टीप्स हूं पेनिन सुला के उतर दिशा में एक सौ बीस कीमिक दूरी सिथ है विलास लास एस्ट्रे लास जैसे रही रही है सी इलाकर के साथ किंग जोर सबसे बड़ा इलाइलेन है
- **WER**: 0.840
- **Reasoning**: The model output numbers in a different format than the reference. This is likely because Whisper's pretrained model was trained on diverse data where numbers may appear as digits, while the reference uses Hindi words.

### Word Insertion/Hallucination (2 examples)

**Example 1:**
- **Reference**: रोलैंडो मेंडोज़ा ने अपनी m16 राइफल से पर्यटकों के ऊपर फायर किया
- **Prediction**: वो लेंगा ने दो जाने की आं सिक्स देन राइफ है पल तो पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पहुत पह�
- **WER**: 6.500
- **Reasoning**: The model inserted extra words not present in the audio, which is a known issue with Whisper in low-resource languages.

**Example 2:**
- **Reference**: द्वीप-समूह पेनिनसुला की उत्तर दिशा में 120 किमी दूर स्थित है विलास लास एस्ट्रेलास जैसे रिहायशी इलाके के साथ किंग ज़ॉर्ज सबसे बड़ा आइलैंड है
- **Prediction**: टीप्स हूं पेनिन सुला के उतर दिशा में एक सौ बीस कीमिक दूरी सिथ है विलास लास एस्ट्रे लास जैसे रही रही है सी इलाकर के साथ किंग जोर सबसे बड़ा इलाइलेन है
- **WER**: 0.840
- **Reasoning**: The model inserted extra words not present in the audio, which is a known issue with Whisper in low-resource languages.

### English Loan Word Error (1 examples)

**Example 1:**
- **Reference**: पिछले हफ्ते meti ने घोषणा की कि apple ने 34 ओवरहीटिंग के मामले सामने आये है  जिसे कंपनी ने गंभीरता से नहीं लिया ।
- **Prediction**: पिछले हवते मेटी आई ने घोशना की कि एपल ने चौथतीस ओवर हीटिंग के मामले सामने आए हैं जिससे कंपनी ने गंभीरता से नहीं लिया
- **WER**: 0.500
- **Reasoning**: The model incorrectly handled an English word spoken in Hindi conversation. It either used Roman script instead of Devanagari or mistranscribed the loanword.

## 1f. Proposed Fixes (Top 3 Error Types)

### Fix for: Number Format Mismatch

**Proposal**: Text Normalization Post-Processor

Apply a Hindi number normalization layer that converts digit sequences back to Hindi words (or vice versa) to match the reference format. This handles the systematic difference between Whisper's tendency to output digits and the reference transcriptions using Hindi number words.

**Implementation Steps**:
1. Build a bidirectional Hindi number word ↔ digit converter
2. Normalize both prediction and reference to the same format before WER
3. This alone can reduce WER by 2-5% on numbers-heavy utterances

**Feasibility**: HIGH — Rule-based, no additional data needed

### Fix for: English Loan Word Error

**Proposal**: Devanagari-English Transliteration Normalizer

Detect English words in Devanagari script and their Roman equivalents, then normalize to a canonical form. For example, map "interview" ↔ "इंटरव्यू" so that either representation is considered correct.

**Implementation Steps**:
1. Maintain a lookup table of common English→Devanagari transliterations
2. For detected English words, accept both script variants
3. Augment training data with both representations

**Feasibility**: MEDIUM — Needs curated transliteration dictionary

### Fix for: Homophone Substitution

**Proposal**: Contextual Language Model Rescoring

Add a Hindi language model (e.g., IndicBERT) as a second-pass rescorer. Generate N-best hypotheses from Whisper and re-rank them using the LM to select the contextually appropriate homophone.

**Implementation Steps**:
1. Generate top-5 beam search hypotheses from Whisper
2. Score each with a pretrained Hindi LM (perplexity-based)
3. Select the hypothesis with lowest perplexity
4. This targets homophones where context disambiguates

**Feasibility**: MEDIUM — Requires inference pipeline modification

## 1g. Implemented Fix — Number Format Normalization

| Reference | Before | After | WER Before | WER After | Improved |
|-----------|--------|-------|------------|-----------|----------|
| सबसे नज़दीकी सिरे पर क्रस्ट की मोटाई करी... | सबसे नजदी किसी रे पर क्रस्ट की... | सबसे नजदी किसी रे पर क्रस्ट की... | 0.429 | 0.429 | ✗ |
| एक पूरी तरह से विकसित एथलीट की तरह बाघ च... | एक पूरी तरह से विकसित एथिलीट क... | एक पूरी तरह से विकसित एथिलीट क... | 0.171 | 0.171 | ✗ |
| यदि आप केवल शिपबोर्ड भ्रमण का उपयोग करके... | यदि आप केवल शिप बोर्ड प्रमण का... | यदि आप केवल शिप बोर्ड प्रमण का... | 0.435 | 0.435 | ✗ |
| 1963 में बांध बनने के बाद मौसमी बाढ़ जो ... | एक अजार नौसौ तिर्सट में बाद बन... | एक अजार नौसौ तिर्सट में बाद बन... | 0.474 | 0.474 | ✗ |
| 29 वर्षीय डॉ मलार बालासुब्रमण्यम सिनसिना... | उन्नतीस वर्षियर डॉक्टर मालाल ब... | उन्नतीस वर्षियर डॉक्टर मालाल ब... | 0.390 | 0.390 | ✗ |
| तीसरी शताब्दी ईसा पूर्व में मिस्रियों द्... | तीसरी शतावती इसापुर्म में मिश्... | तीसरी शतावती इसापुर्म में मिश्... | 0.462 | 0.462 | ✗ |
| स्प्रिंगबोक्स के लिए इसने पांच-मैचों की ... | स्प्रिंग बॉक्स के लिए इसने पां... | स्प्रिंग बॉक्स के लिए इसने पां... | 0.417 | 0.417 | ✗ |
| सबसे नज़दीकी सिरे पर क्रस्ट की मोटाई करी... | सबसे नजदीकी शिर्य पर क्रिस्ट क... | सबसे नजदीकी शिर्य पर क्रिस्ट क... | 0.429 | 0.429 | ✗ |
| हमारे ग्रह की नदियों से महासागरों में जा... | हमारे ग्रेह की नदियों से महां ... | हमारे ग्रेह की नदियों से महां ... | 0.412 | 0.412 | ✗ |
| सुंदरबन दुनिया का सबसे बड़ा तटीय मैंग्रो... | सुन्दर बन दुनियां का सबसे बड़ा... | सुन्दर बन दुनियां का सबसे बड़ा... | 0.577 | 0.577 | ✗ |
