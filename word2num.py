"""
Deterministic multilingual ASR number parser (English / Hindi / Bengali / mixed).

Architecture (per the "candidates, not decisions" principle):

    raw text
      -> digit normalization (Bengali/Devanagari/Arabic-Indic -> ASCII)
      -> tokenization (punctuation, hyphenation, digit-run boundaries)
      -> STRICT classification (exact match against per-language lexicons only)
      -> structural compound-hundred recognition (merged "barosho" -> 12*100,
         validated by EXACT prefix lookup, never by fuzzy distance)
      -> CONSERVATIVE fuzzy candidate generation for tokens still unresolved
         (length-aware tolerance, per-script candidate pools, stopword-protected,
         and only accepted if there is a single unambiguous best candidate)
      -> hierarchical grammar/accumulation with magnitude-order + duplicate
         validation
      -> result: a value, OR an explicit ambiguous/unresolved flag

No ML model, no network calls, no third-party dependencies.

IMPORTANT / HONEST CAVEAT: the Hindi and Bengali 0-99 word lists below were
compiled from well-documented standard forms, not verified against a native
speaker or production ASR corpus. Before shipping this against real billing
data, have a Hindi and a Bengali speaker sanity-check the two CANONICAL_*
tables, and grow ROMAN_*_VARIANTS from your actual ASR logs -- that variant
table is the single highest-leverage thing you can invest in.
"""

import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Union

# ======================================================================
# 1. DIGIT NORMALIZATION
# ======================================================================

_BENGALI_DIGITS = str.maketrans('০১২৩৪৫৬৭৮৯', '0123456789')
_DEVANAGARI_DIGITS = str.maketrans('०१२३४५६७८९', '0123456789')
_ARABIC_INDIC_DIGITS = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')


def normalize_digits(text: str) -> str:
    text = text.translate(_BENGALI_DIGITS)
    text = text.translate(_DEVANAGARI_DIGITS)
    text = text.translate(_ARABIC_INDIC_DIGITS)
    return text


def normalize_punctuation(text: str) -> str:
    # "2,500" (no space) -> "2500"; "2, 5" (space after comma) is left alone,
    # it's plausibly two separate numbers, not a thousands separator.
    text = re.sub(r'(?<=\d),(?=\d)', '', text)
    # hyphenation: "twenty-five" -> "twenty five"
    text = text.replace('-', ' ')
    return text


# ======================================================================
# 2. CANONICAL LEXICONS (per-language, kept separate -- never merged into
#    one global fuzzy pool; spec section 27)
# ======================================================================

ENGLISH_UNITS = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
    'seven': 7, 'eight': 8, 'nine': 9,
}
ENGLISH_TEENS = {
    'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
    'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
}
ENGLISH_TENS = {
    'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60,
    'seventy': 70, 'eighty': 80, 'ninety': 90,
}
CANONICAL_ENGLISH = {**ENGLISH_UNITS, **ENGLISH_TEENS, **ENGLISH_TENS}

# Complete Hindi 0-99 (irregular, non-compositional -- see module caveat)
CANONICAL_HINDI = {
    'शून्य': 0, 'एक': 1, 'दो': 2, 'तीन': 3, 'चार': 4, 'पांच': 5, 'पाँच': 5,
    'छह': 6, 'छः': 6, 'सात': 7, 'आठ': 8, 'नौ': 9, 'दस': 10,
    'ग्यारह': 11, 'बारह': 12, 'तेरह': 13, 'चौदह': 14, 'पंद्रह': 15, 'सोलह': 16,
    'सत्रह': 17, 'अठारह': 18, 'उन्नीस': 19, 'बीस': 20,
    'इक्कीस': 21, 'बाईस': 22, 'तेईस': 23, 'चौबीस': 24, 'पच्चीस': 25,
    'छब्बीस': 26, 'सत्ताईस': 27, 'अट्ठाईस': 28, 'उनतीस': 29, 'तीस': 30,
    'इकतीस': 31, 'बत्तीस': 32, 'तैंतीस': 33, 'चौंतीस': 34, 'पैंतीस': 35,
    'छत्तीस': 36, 'सैंतीस': 37, 'अड़तीस': 38, 'उनतालीस': 39, 'चालीस': 40,
    'इकतालीस': 41, 'बयालीस': 42, 'तैंतालीस': 43, 'चौवालीस': 44, 'पैंतालीस': 45,
    'छियालीस': 46, 'सैंतालीस': 47, 'अड़तालीस': 48, 'उनचास': 49, 'पचास': 50,
    'इक्यावन': 51, 'बावन': 52, 'तिरपन': 53, 'चौवन': 54, 'पचपन': 55,
    'छप्पन': 56, 'सत्तावन': 57, 'अट्ठावन': 58, 'उनसठ': 59, 'साठ': 60,
    'इकसठ': 61, 'बासठ': 62, 'तिरसठ': 63, 'चौंसठ': 64, 'पैंसठ': 65,
    'छियासठ': 66, 'सड़सठ': 67, 'अड़सठ': 68, 'उनहत्तर': 69, 'सत्तर': 70,
    'इकहत्तर': 71, 'बहत्तर': 72, 'तिहत्तर': 73, 'चौहत्तर': 74, 'पचहत्तर': 75,
    'छिहत्तर': 76, 'सतहत्तर': 77, 'अठहत्तर': 78, 'उन्यासी': 79, 'अस्सी': 80,
    'इक्यासी': 81, 'बयासी': 82, 'तिरासी': 83, 'चौरासी': 84, 'पचासी': 85,
    'छियासी': 86, 'सतासी': 87, 'अट्ठासी': 88, 'नवासी': 89, 'नब्बे': 90,
    'इक्यानवे': 91, 'बानवे': 92, 'तिरानवे': 93, 'चौरानवे': 94, 'पंचानवे': 95,
    'छियानवे': 96, 'सत्तानवे': 97, 'अट्ठानवे': 98, 'निन्यानवे': 99,
}

# Complete Bengali 0-99 (irregular, non-compositional -- see module caveat)
CANONICAL_BENGALI = {
    'শূন্য': 0, 'এক': 1, 'দুই': 2, 'তিন': 3, 'চার': 4, 'পাঁচ': 5, 'ছয়': 6,
    'সাত': 7, 'আট': 8, 'নয়': 9, 'দশ': 10,
    'এগারো': 11, 'বারো': 12, 'তেরো': 13, 'চৌদ্দ': 14, 'পনেরো': 15, 'ষোল': 16,
    'সতেরো': 17, 'আঠারো': 18, 'উনিশ': 19, 'বিশ': 20,
    'একুশ': 21, 'বাইশ': 22, 'তেইশ': 23, 'চব্বিশ': 24, 'পঁচিশ': 25,
    'ছাব্বিশ': 26, 'সাতাশ': 27, 'আটাশ': 28, 'ঊনত্রিশ': 29, 'ত্রিশ': 30,
    'একত্রিশ': 31, 'বত্রিশ': 32, 'তেত্রিশ': 33, 'চৌত্রিশ': 34, 'পঁয়ত্রিশ': 35,
    'ছত্রিশ': 36, 'সাঁইত্রিশ': 37, 'আটত্রিশ': 38, 'ঊনচল্লিশ': 39, 'চল্লিশ': 40,
    'একচল্লিশ': 41, 'বিয়াল্লিশ': 42, 'তেতাল্লিশ': 43, 'চুয়াল্লিশ': 44,
    'পঁয়তাল্লিশ': 45, 'ছেচল্লিশ': 46, 'সাতচল্লিশ': 47, 'আটচল্লিশ': 48,
    'ঊনপঞ্চাশ': 49, 'পঞ্চাশ': 50,
    'একান্ন': 51, 'বাহান্ন': 52, 'তেপ্পান্ন': 53, 'চুয়ান্ন': 54, 'পঞ্চান্ন': 55,
    'ছাপ্পান্ন': 56, 'সাতান্ন': 57, 'আটান্ন': 58, 'ঊনষাট': 59, 'ষাট': 60,
    'একষট্টি': 61, 'বাষট্টি': 62, 'তেষট্টি': 63, 'চৌষট্টি': 64, 'পঁয়ষট্টি': 65,
    'ছেষট্টি': 66, 'সাতষট্টি': 67, 'আটষট্টি': 68, 'ঊনসত্তর': 69, 'সত্তর': 70,
    'একাত্তর': 71, 'বাহাত্তর': 72, 'তিয়াত্তর': 73, 'চুয়াত্তর': 74, 'পঁচাত্তর': 75,
    'ছিয়াত্তর': 76, 'সাতাত্তর': 77, 'আটাত্তর': 78, 'ঊনআশি': 79, 'আশি': 80,
    'একাশি': 81, 'বিরাশি': 82, 'তিরাশি': 83, 'চুরাশি': 84, 'পঁচাশি': 85,
    'ছিয়াশি': 86, 'সাতাশি': 87, 'আটাশি': 88, 'ঊননব্বই': 89, 'নব্বই': 90,
    'একানব্বই': 91, 'বিরানব্বই': 92, 'তিরানব্বই': 93, 'চুরানব্বই': 94,
    'পঁচানব্বই': 95, 'ছিয়ানব্বই': 96, 'সাতানব্বই': 97, 'আটানব্বই': 98,
    'নিরানব্বই': 99,
}

# Romanized Hindi variants (ASR-observed spellings -> value). Seed list --
# grow this from real production logs, it matters more than anything else here.
ROMAN_HINDI_VARIANTS: Dict[str, int] = {
    'shunya': 0, 'ek': 1, 'aek': 1, 'do': 2, 'du': 2, 'teen': 3, 'tin': 3,
    'char': 4, 'chaar': 4, 'panch': 5, 'paanch': 5, 'chhe': 6, 'che': 6,
    'saat': 7, 'sat': 7, 'aath': 8, 'ath': 8, 'nau': 9, 'das': 10, 'dus': 10,
    'gyarah': 11, 'gyaarah': 11, 'barah': 12, 'baarah': 12,
    'terah': 13, 'terha': 13,
    'chaudah': 14, 'chodah': 14,
    'pandrah': 15, 'pandra': 15,
    'solah': 16, 'sola': 16,
    'satrah': 17, 'saterah': 17, 'satero': 17, 'sotero': 17,
    'atharah': 18, 'atherah': 18, 'athero': 18, 'atharo': 18,
    'unnis': 19, 'unnees': 19,
    'bees': 20, 'bis': 20,
    'ikkis': 21, 'ekus': 21, 'ekkus': 21,   # 'ekus'/'ekkus' reported ASR variants
    'baees': 22, 'bais': 22, 'teis': 23, 'chaubis': 24,
    'pachis': 25, 'pachees': 25, 'pachish': 25,
    'chhabbis': 26, 'sattais': 27, 'atthais': 28, 'unatis': 29, 'unantis': 29,
    'tees': 30, 'tis': 30,
    'ikatis': 31, 'battis': 32, 'taintis': 33, 'chauntis': 34, 'paintis': 35,
    'chhattis': 36, 'saintis': 37, 'adtis': 38, 'artis': 38, 'unchalis': 39,
    'chalis': 40, 'chaalis': 40,
    'iktalis': 41, 'bayalis': 42, 'taintalis': 43, 'chauwalis': 44,
    'paintalis': 45, 'chhiyalis': 46, 'saintalis': 47, 'adtalis': 48,
    'unchaas': 49, 'unchas': 49,
    'pachas': 50, 'pachash': 50,
    'ikyawan': 51, 'bawan': 52, 'tirpan': 53, 'chauwan': 54, 'pachpan': 55,
    'chhappan': 56, 'sattawan': 57, 'atthawan': 58, 'unsath': 59,
    'saath': 60, 'sath': 60,
    'iksath': 61, 'baasath': 62, 'tirsath': 63, 'chausath': 64,
    'painsath': 65, 'chhiyasath': 66, 'sadsath': 67, 'adsath': 68,
    'unhattar': 69,
    'sattar': 70,
    'ikhattar': 71, 'bahattar': 72, 'tihattar': 73, 'chauhattar': 74,
    'pachhattar': 75, 'chhihattar': 76, 'sathattar': 77, 'athhattar': 78,
    'unyasi': 79,
    'assi': 80, 'asi': 80,
    'ikyasi': 81, 'bayasi': 82, 'tirasi': 83, 'chaurasi': 84, 'pachasi': 85,
    'chhiyasi': 86, 'satasi': 87, 'atthasi': 88, 'navasi': 89,
    'nabbe': 90,
    'ikyanave': 91, 'baanve': 92, 'tiranve': 93, 'chauranve': 94,
    'pachanve': 95, 'chhiyanve': 96, 'sattanve': 97, 'atthanve': 98,
    'ninyanve': 99,
}

# Romanized Bengali variants (seed list -- extend from real logs)
ROMAN_BENGALI_VARIANTS: Dict[str, int] = {
    'shunno': 0, 'ek': 1, 'aek': 1, 'dui': 2, 'du': 2, 'tin': 3, 'teen': 3,
    'char': 4, 'chaar': 4, 'panch': 5, 'pach': 5, 'choy': 6, 'chhoy': 6,
    'shaat': 7, 'saat': 7, 'aat': 8, 'ath': 8, 'noy': 9, 'dosh': 10, 'dash': 10,
    'egaro': 11, 'egro': 11, 'baro': 12, 'baaro': 12,
    'choddo': 14, 'chowddo': 14,
    'ponero': 15, 'ponro': 15,
    'sholo': 16, 'sholoh': 16,
    'sotero': 17, 'satero': 17, 'shotero': 17,
    'atharo': 18, 'athero': 18, 'attharo': 18,
    'unish': 19, 'unnish': 19,
    'bish': 20, 'bees': 20,
    'ekush': 21, 'baish': 22, 'teish': 23, 'chobbish': 24,
    'pochish': 25, 'pochis': 25, 'pachish': 25,
    'chhabbish': 26, 'shatash': 27, 'atash': 28, 'unotrish': 29,
    'trish': 30, 'tirish': 30,
    'ektrish': 31, 'batrish': 32, 'tetrish': 33, 'choutrish': 34,
    'poytrish': 35, 'chotrish': 36, 'saintrish': 37, 'attrish': 38,
    'unochollish': 39,
    'chollish': 40,
    'ekchollish': 41, 'biallish': 42, 'tetallish': 43, 'chuallish': 44,
    'poytallish': 45, 'checchollish': 46, 'shatchollish': 47,
    'atchollish': 48, 'unponchash': 49,
    'ponchash': 50,
    'ekanno': 51, 'bahanno': 52, 'teppanno': 53, 'chuanno': 54,
    'ponchanno': 55, 'chappanno': 56, 'shatanno': 57, 'atanno': 58,
    'unshaat': 59,
    'shaat_': 60,  # placeholder avoided below, see SHAAT60 handling
    'ekshotti': 61, 'bashotti': 62, 'teshotti': 63, 'choushotti': 64,
    'poyshotti': 65, 'cheshotti': 66, 'shatshotti': 67, 'atshotti': 68,
    'unshottor': 69,
    'shottor': 70,
    'ekattor': 71, 'bahattor': 72, 'tiattor': 73, 'chuattor': 74,
    'pochattor': 75, 'chiattor': 76, 'shatattor': 77, 'atattor': 78,
    'unashi': 79,
    'ashi': 80,
    'ekashi': 81, 'birashi': 82, 'tirashi': 83, 'churashi': 84,
    'pochashi': 85, 'chiyashi': 86, 'shatashi': 87, 'atashi': 88,
    'unonobboi': 89,
    'nobboi': 90,
    'ekanobboi': 91, 'biranobboi': 92, 'tiranobboi': 93, 'churanobboi': 94,
    'pochanobboi': 95, 'chiyanobboi': 96, 'shatanobboi': 97, 'atanobboi': 98,
    'niranobboi': 99,
}
# 'shaat_' was a placeholder to keep 60 out of collision with 7 ('shaat'=7
# above); fix: use the correct romanization for 60 without clashing with 7.
del ROMAN_BENGALI_VARIANTS['shaat_']
ROMAN_BENGALI_VARIANTS['sháth'] = 60
ROMAN_BENGALI_VARIANTS['shaath'] = 60

# The "teroso" lesson, made explicit and permanent:
# "tero" is deliberately NOT registered as a variant in either language. It's
# a plausible fragment of "terah"(13ish) AND phonetically brushes past
# "satero"(17) / "athero"(18). Per the spec: a fuzzy matcher must never pick
# the nearest of these by edit distance alone. So we simply never teach the
# system that "tero" means anything on its own. An unresolved "tero" or
# "teroso" comes back as ambiguous/unresolved rather than a confident wrong
# number. If your real ASR data shows "tero" reliably means one specific
# thing, add it explicitly and consciously -- don't let fuzzy matching infer it.

CANONICAL_MAGNITUDES = {
    'hundred': 100, 'thousand': 1000, 'lakh': 100_000, 'lac': 100_000,
    'million': 1_000_000, 'crore': 10_000_000, 'billion': 1_000_000_000,
    'শ': 100, 'শত': 100, 'হাজার': 1000, 'লাখ': 100_000, 'লক্ষ': 100_000,
    'কোটি': 10_000_000,
    'सौ': 100, 'हज़ार': 1000, 'हजार': 1000, 'लाख': 100_000, 'करोड़': 10_000_000,
}
ROMAN_MAGNITUDE_VARIANTS = {
    'hazar': 1000, 'hazaar': 1000, 'hajar': 1000, 'hajaar': 1000, 'hazr': 1000,
    'lac': 100_000, 'lakhs': 100_000,
    'crore': 10_000_000, 'crores': 10_000_000,
    # the Hindi/Bengali "hundred" suffix, merged or separated ("baro sho",
    # "barosho", "sau", "so") -- this is what makes compound-hundred parsing
    # (section below) work without special-casing every merged spelling
    'sau': 100, 'sho': 100, 'so': 100, 'shô': 100,
}

CONNECTORS = {'and', 'ও', 'আর', 'और', 'aur'}
DECIMAL_MARKERS = {'point', 'দশমিক', 'दशमलव'}
NEGATIVE_MARKERS = {'minus', 'negative', 'মাইনাস', 'माइनस'}

# Bengali-script phonetic renderings of English number words (ASR hears
# English, transcribes in Bengali script). Seed list -- extend from logs.
BENGALI_SCRIPT_ENGLISH = {
    'ওয়ান': 1, 'টু': 2, 'থ্রি': 3, 'ফোর': 4, 'ফাইভ': 5, 'সিক্স': 6,
    'সেভেন': 7, 'এইট': 8, 'নাইন': 9, 'টেন': 10,
    'ইলেভেন': 11, 'টুয়েলভ': 12, 'থার্টিন': 13, 'ফোরটিন': 14, 'ফিফটিন': 15,
    'সিক্সটিন': 16, 'সেভেনটিন': 17, 'এইটিন': 18, 'নাইনটিন': 19,
    'টুয়েন্টি': 20, 'থার্টি': 30, 'ফোর্টি': 40, 'ফিফটি': 50, 'সিক্সটি': 60,
    'সেভেন্টি': 70, 'এইটি': 80, 'নাইন্টি': 90,
    'হান্ড্রেড': 100, 'থাউজেন্ড': 1000,
}
# Devanagari-script phonetic renderings of English number words
DEVANAGARI_SCRIPT_ENGLISH = {
    'वन': 1, 'टू': 2, 'थ्री': 3, 'फोर': 4, 'फाइव': 5, 'सिक्स': 6,
    'सेवन': 7, 'एट': 8, 'नाइन': 9, 'टेन': 10,
    'इलेवन': 11, 'ट्वेल्व': 12, 'थर्टीन': 13, 'फोर्टीन': 14, 'फिफ्टीन': 15,
    'सिक्सटीन': 16, 'सेवनटीन': 17, 'एटीन': 18, 'नाइनटीन': 19,
    'ट्वेंटी': 20, 'थर्टी': 30, 'फोर्टी': 40, 'फिफ्टी': 50, 'सिक्सटी': 60,
    'सेवंटी': 70, 'एटी': 80, 'नाइंटी': 90,
    'हंड्रेड': 100, 'थाउजेंड': 1000,
}

# Words that must NEVER be treated as numbers even if phonetically close to
# one -- this is the direct fix for "টাকা"("taka"/currency) -> "lakh".
# Grow this from your real ASR vocabulary's frequent non-number tokens.
STOPWORDS = {
    'taka', 'টাকা', 'rupee', 'rupees', 'rupaye', 'रुपये', 'रुपया', 'টাকাটা',
    'kg', 'কেজি', 'किलो', 'gram', 'গ্রাম', 'ग्राम',
    'piece', 'pieces', 'পিস', 'पीस',
    'product', 'প্রোডাক্ট', 'उत्पाद', 'rate', 'রেট', 'दर',
    'price', 'দাম', 'কীমত', 'कीमत', 'quantity', 'পরিমাণ', 'मात्रा',
}

ALL_MAGNITUDE_VALUES = set(CANONICAL_MAGNITUDES.values()) | set(ROMAN_MAGNITUDE_VARIANTS.values())


# ======================================================================
# 3. SCRIPT DETECTION
# ======================================================================

def script_of(token: str) -> str:
    for ch in token:
        cp = ord(ch)
        if 0x0980 <= cp <= 0x09FF:
            return 'bengali'
        if 0x0900 <= cp <= 0x097F:
            return 'devanagari'
        if ('a' <= ch <= 'z') or ('A' <= ch <= 'Z'):
            return 'latin'
    return 'other'


# Rough akshara -> roman consonant/vowel skeleton. Not linguistically perfect --
# it only needs to be *consistent* enough that edit-distance fuzzy matching
# against romanized forms works.
_BENGALI_TRANSLIT = {
    'অ': 'a', 'আ': 'a', 'ই': 'i', 'ঈ': 'i', 'উ': 'u', 'ঊ': 'u',
    'এ': 'e', 'ঐ': 'oi', 'ও': 'o', 'ঔ': 'ou',
    'ক': 'k', 'খ': 'kh', 'গ': 'g', 'ঘ': 'gh', 'ঙ': 'ng',
    'চ': 'ch', 'ছ': 'chh', 'জ': 'j', 'ঝ': 'jh', 'ঞ': 'n',
    'ট': 't', 'ঠ': 'th', 'ড': 'd', 'ঢ': 'dh', 'ণ': 'n',
    'ত': 't', 'থ': 'th', 'দ': 'd', 'ধ': 'dh', 'ন': 'n',
    'প': 'p', 'ফ': 'ph', 'ব': 'b', 'ভ': 'bh', 'ম': 'm',
    'য': 'j', 'র': 'r', 'ল': 'l', 'শ': 'sh', 'ষ': 'sh',
    'স': 's', 'হ': 'h', 'ড়': 'r', 'ঢ়': 'rh', 'য়': 'y',
    'া': 'a', 'ি': 'i', 'ী': 'i', 'ু': 'u', 'ূ': 'u',
    'ে': 'e', 'ৈ': 'oi', 'ো': 'o', 'ৌ': 'ou', 'ং': 'ng',
    'ঃ': 'h', '্': '', 'ঁ': '',
}
_DEVANAGARI_TRANSLIT = {
    'अ': 'a', 'आ': 'a', 'इ': 'i', 'ई': 'i', 'उ': 'u', 'ऊ': 'u',
    'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
    'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'n',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'श': 'sh', 'ष': 'sh',
    'स': 's', 'ह': 'h', 'ड़': 'r', 'ढ़': 'rh',
    'ा': 'a', 'ि': 'i', 'ी': 'i', 'ु': 'u', 'ू': 'u',
    'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au', 'ं': 'n',
    'ः': 'h', '्': '', 'ँ': '',
}


def transliterate_to_roman(token: str) -> str:
    """Best-effort phonetic skeleton, used only for fuzzy matching (never
    for display, and never for exact-match lookups)."""
    script = script_of(token)
    table = _BENGALI_TRANSLIT if script == 'bengali' else (
        _DEVANAGARI_TRANSLIT if script == 'devanagari' else None
    )
    if table is None:
        return token.lower()
    return ''.join(table.get(ch, ch if ch.isascii() else '') for ch in token)


# ======================================================================
# 4. STRICT (EXACT-ONLY) LOOKUP -- built per script, never merged globally
# ======================================================================

_EXACT_LATIN: Dict[str, int] = {}
_EXACT_LATIN.update(CANONICAL_ENGLISH)
_EXACT_LATIN.update(ROMAN_HINDI_VARIANTS)
_EXACT_LATIN.update(ROMAN_BENGALI_VARIANTS)
_EXACT_LATIN.update(CANONICAL_MAGNITUDES)   # english-keyed ones only matter here
_EXACT_LATIN.update(ROMAN_MAGNITUDE_VARIANTS)
_EXACT_LATIN.update({'hundred': 100, 'thousand': 1000})

_EXACT_BENGALI: Dict[str, int] = {}
_EXACT_BENGALI.update(CANONICAL_BENGALI)
_EXACT_BENGALI.update({k: v for k, v in CANONICAL_MAGNITUDES.items() if script_of(k) == 'bengali'})
_EXACT_BENGALI.update(BENGALI_SCRIPT_ENGLISH)

_EXACT_DEVANAGARI: Dict[str, int] = {}
_EXACT_DEVANAGARI.update(CANONICAL_HINDI)
_EXACT_DEVANAGARI.update({k: v for k, v in CANONICAL_MAGNITUDES.items() if script_of(k) == 'devanagari'})
_EXACT_DEVANAGARI.update(DEVANAGARI_SCRIPT_ENGLISH)

_MAGNITUDE_KEYS = set(CANONICAL_MAGNITUDES.keys()) | set(ROMAN_MAGNITUDE_VARIANTS.keys()) | {'hundred', 'thousand'}


def _exact_lookup(token: str) -> Optional[Tuple[int, str]]:
    """Try exact match in the script-appropriate table. Returns (value, script)."""
    script = script_of(token)
    lower = token.lower()
    if script == 'bengali' and token in _EXACT_BENGALI:
        return _EXACT_BENGALI[token], 'bengali'
    if script == 'devanagari' and token in _EXACT_DEVANAGARI:
        return _EXACT_DEVANAGARI[token], 'devanagari'
    if script == 'latin':
        if token in _EXACT_LATIN:
            return _EXACT_LATIN[token], 'latin'
        if lower in _EXACT_LATIN:
            return _EXACT_LATIN[lower], 'latin'
    return None


# ======================================================================
# 5. STRUCTURAL COMPOUND-HUNDRED RECOGNITION
#    "barosho"/"baroso" -> strip suffix, EXACT-match the prefix, never fuzzy.
# ======================================================================

_HUNDRED_SUFFIXES = ('sho', 'sau', 'so')  # longest first would matter if overlapping; these don't


def try_compound_hundred(token: str) -> Optional[int]:
    """Return the composed value (e.g. 1200) only if the token is a merged
    <number-word><hundred-suffix> and the prefix EXACTLY matches a known
    1-99 variant (romanized OR native-script). No fuzzy matching here --
    this is exactly the mechanism that must reject "teroso"."""
    lower = token.lower()
    for suffix in _HUNDRED_SUFFIXES:
        if lower.endswith(suffix) and len(lower) > len(suffix):
            prefix = lower[: -len(suffix)]
            if prefix in ROMAN_HINDI_VARIANTS:
                return ROMAN_HINDI_VARIANTS[prefix] * 100
            if prefix in ROMAN_BENGALI_VARIANTS:
                return ROMAN_BENGALI_VARIANTS[prefix] * 100
            if prefix in ENGLISH_UNITS or prefix in ENGLISH_TEENS:
                return (ENGLISH_UNITS.get(prefix) or ENGLISH_TEENS.get(prefix)) * 100

    # native-script merged compounds, e.g. Bengali "পাঁচশ" (pach + শ = 500),
    # "একশ" (ek + শ = 100). Suffix stripped on the ORIGINAL (non-lowered)
    # string since Bengali/Devanagari have no case.
    for suffix in ('শো', 'শ'):  # try longer suffix first
        if token.endswith(suffix) and len(token) > len(suffix):
            prefix = token[: -len(suffix)]
            if prefix in CANONICAL_BENGALI:
                return CANONICAL_BENGALI[prefix] * 100
    for suffix in ('सौ',):
        if token.endswith(suffix) and len(token) > len(suffix):
            prefix = token[: -len(suffix)]
            if prefix in CANONICAL_HINDI:
                return CANONICAL_HINDI[prefix] * 100
    return None


def _looks_like_failed_compound(token: str) -> bool:
    """True if the token has the *shape* of a merged hundred-compound (ends
    in a recognized hundred-suffix with a nontrivial prefix) even though
    try_compound_hundred() didn't validate it. Used to block naive whole-token
    fuzzy matching, which would otherwise ignore the suffix's meaning entirely
    (this is precisely how "teroso" -> 13/1800 happened)."""
    lower = token.lower()
    for suffix in _HUNDRED_SUFFIXES:
        if lower.endswith(suffix) and len(lower) > len(suffix) + 1:
            return True
    for suffix in ('শো', 'শ', 'সৌ'):
        if token.endswith(suffix) and len(token) > len(suffix):
            return True
    if token.endswith('सौ') and len(token) > len('सौ'):
        return True
    return False


# ======================================================================
# 6. CONSERVATIVE FUZZY CANDIDATE GENERATION
#    Only for tokens that failed strict + structural resolution. Never used
#    on short/critical words. Only accepted if unambiguous.
# ======================================================================

def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * lb
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


MIN_FUZZY_LENGTH = 5       # words shorter than this: exact match only, no fuzzy
                            # (raising from 4->5 specifically stops "tero" ~ "zero"
                            #  style false positives on short critical words --
                            #  short misspellings should be added as explicit
                            #  known variants above instead, see 'ekus'/'ekkus')
_MAX_DIST_BY_LEN = {5: 1, 6: 2, 7: 2, 8: 2}
_MIN_MARGIN = 1             # best cluster must beat the runner-up cluster by
                            # at least this much distance, or we refuse to guess


def _max_dist(length: int) -> int:
    if length in _MAX_DIST_BY_LEN:
        return _MAX_DIST_BY_LEN[length]
    return 1 if length < 5 else 3


# ----------------------------------------------------------------------
# CONSOLIDATED cross-language variant pool, per value, romanized/transliterated
# into one shared phonetic space. This is what lets "ekus" (latin) potentially
# fuzzy-match against a transliterated form of "একুশ" (bengali) -- everything
# lands in the same comparison space instead of siloed per-script pools.
# Still only used as a FALLBACK after exact + compound-hundred both fail, and
# still requires an unambiguous single-value winner (see resolve_fuzzy_unified).
# ----------------------------------------------------------------------

def _build_unified_pool() -> Dict[str, int]:
    pool: Dict[str, int] = {}
    pool.update(CANONICAL_ENGLISH)
    pool.update(ROMAN_HINDI_VARIANTS)
    pool.update(ROMAN_BENGALI_VARIANTS)
    for word, val in CANONICAL_HINDI.items():
        pool.setdefault(transliterate_to_roman(word), val)
    for word, val in CANONICAL_BENGALI.items():
        pool.setdefault(transliterate_to_roman(word), val)
    for word, val in BENGALI_SCRIPT_ENGLISH.items():
        pool.setdefault(transliterate_to_roman(word), val)
    for word, val in DEVANAGARI_SCRIPT_ENGLISH.items():
        pool.setdefault(transliterate_to_roman(word), val)
    return pool


_UNIFIED_POOL: Dict[str, int] = None  # built lazily, after transliterate_to_roman is defined below


@dataclass
class Candidate:
    word: str
    value: int
    distance: int


def fuzzy_candidates(token: str) -> List[Candidate]:
    """Generate candidates from the unified cross-language pool. Caller
    (resolve_fuzzy) decides whether to trust them -- this function never
    picks a winner itself."""
    global _UNIFIED_POOL
    if _UNIFIED_POOL is None:
        _UNIFIED_POOL = _build_unified_pool()

    script = script_of(token)
    query = token.lower() if script == 'latin' else transliterate_to_roman(token)

    if len(query) < MIN_FUZZY_LENGTH:
        return []  # too short/critical to risk fuzzy matching

    limit = _max_dist(len(query))
    out = []
    for cand_word, cand_val in _UNIFIED_POOL.items():
        d = edit_distance(query, cand_word)
        if d <= limit:
            out.append(Candidate(cand_word, cand_val, d))
    out.sort(key=lambda c: c.distance)
    return out


def resolve_fuzzy(token: str) -> Optional[int]:
    """Accept a fuzzy result only if the best-matching VALUE has a clear,
    unambiguous lead over every other distinct value in the candidate pool
    (margin-gated), not merely the single nearest string. This is what lets
    a big shared pool increase recall without reintroducing "nearest word
    wins" as a silent decision-maker."""
    cands = fuzzy_candidates(token)
    if not cands:
        return None

    best_distance_per_value: Dict[int, int] = {}
    for c in cands:
        if c.value not in best_distance_per_value or c.distance < best_distance_per_value[c.value]:
            best_distance_per_value[c.value] = c.distance

    ranked = sorted(best_distance_per_value.items(), key=lambda kv: kv[1])
    best_value, best_dist = ranked[0]
    if len(ranked) == 1:
        return best_value
    _, second_dist = ranked[1]
    if second_dist - best_dist >= _MIN_MARGIN:
        return best_value
    return None  # two distinct numbers are both plausible -- refuse rather than guess


# ======================================================================
# 7. TOKENIZATION + CLASSIFICATION
# ======================================================================

TOKEN_RE = re.compile(r'\d+|[^\s]+')


@dataclass
class Token:
    raw: str
    kind: str            # DIGIT | WORD | MAGNITUDE | COMPOUND | CONNECTOR | DECIMAL_MARK | NEGATIVE | OTHER
    value: Optional[int] = None
    matched_as: Optional[str] = None
    ambiguous_source: bool = False   # true if resolution was refused due to ambiguity


def classify(raw: str) -> Token:
    if raw.isdigit():
        return Token(raw, 'DIGIT', int(raw))

    lower = raw.lower()
    if raw in CONNECTORS or lower in CONNECTORS:
        return Token(raw, 'CONNECTOR')
    if raw in DECIMAL_MARKERS or lower in DECIMAL_MARKERS:
        return Token(raw, 'DECIMAL_MARK')
    if raw in NEGATIVE_MARKERS or lower in NEGATIVE_MARKERS:
        return Token(raw, 'NEGATIVE')

    if raw in STOPWORDS or lower in STOPWORDS:
        return Token(raw, 'OTHER')

    exact = _exact_lookup(raw)
    if exact is not None:
        val, _ = exact
        key = raw if raw in _MAGNITUDE_KEYS else lower
        is_mag = (raw in _MAGNITUDE_KEYS) or (lower in _MAGNITUDE_KEYS)
        return Token(raw, 'MAGNITUDE' if is_mag else 'WORD', val, matched_as=raw)

    compound = try_compound_hundred(raw)
    if compound is not None:
        return Token(raw, 'COMPOUND', compound, matched_as=raw)

    if _looks_like_failed_compound(raw):
        # The token ends in a recognized hundred-suffix but the prefix did
        # not exactly validate (e.g. "teroso" -> prefix "tero" is not a
        # registered 1-99 form in the roman tables we compound against).
        # A naive whole-token fuzzy match here would just ignore the
        # suffix's meaning and match against some unrelated shorter word
        # (this is exactly how "teroso" silently became 1800/13 before) --
        # so we deliberately refuse rather than fuzzy-guess.
        return Token(raw, 'OTHER', ambiguous_source=True)

    fuzzy_val = resolve_fuzzy(raw)
    if fuzzy_val is not None:
        return Token(raw, 'WORD', fuzzy_val, matched_as=f'~{raw}')

    # Bengali/Devanagari script writing English number words phonetically:
    # only attempted here (after own-language exact+fuzzy failed), and only
    # via exact table (already checked above) -- if it's not an exact hit in
    # BENGALI_SCRIPT_ENGLISH / DEVANAGARI_SCRIPT_ENGLISH, we do NOT fuzzy
    # cross-script guess. This is intentionally conservative; extend those
    # tables from real ASR logs rather than loosening this check.

    return Token(raw, 'OTHER')


def tokenize(text: str) -> List[Token]:
    text = normalize_digits(text)
    text = normalize_punctuation(text)
    raw_tokens = TOKEN_RE.findall(text)
    return [classify(t) for t in raw_tokens]


# ======================================================================
# 8. HIERARCHICAL GRAMMAR / ACCUMULATION PARSER
#    - digit runs are atomic and NEVER silently concatenated with an
#      adjacent digit run (spec section 33)
#    - magnitude order + duplicate flush validated (spec sections 35-36)
#    - compound-hundred tokens flush directly (already fully composed)
# ======================================================================

@dataclass
class NumberResult:
    text: str
    value: Optional[Union[int, Decimal]]
    ambiguous: bool
    reason: Optional[str] = None
    tokens: List[Token] = field(default_factory=list)


def parse_tokens(tokens: List[Token]) -> NumberResult:
    result = 0
    current = 0
    last_flushed_magnitude: Optional[int] = None
    seen_any = False
    negative = False
    ambiguous_reason = None
    prev_kind = None

    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]

        if tok.kind == 'CONNECTOR':
            prev_kind = tok.kind
            i += 1
            continue

        if tok.kind == 'NEGATIVE':
            negative = True
            prev_kind = tok.kind
            i += 1
            continue

        if tok.kind == 'DIGIT':
            seen_any = True
            # a raw digit run immediately after ANOTHER raw digit run (no
            # magnitude word between them) is a phrase boundary, never an
            # implicit concatenation or sum -- caller's extract_numbers()
            # actually does the splitting; here we just guard against the
            # accumulator silently mixing two independent digit runs.
            if prev_kind == 'DIGIT':
                ambiguous_reason = f"adjacent digit runs '{tokens[i-1].raw}' and '{tok.raw}' with no connecting word"
                break
            if last_flushed_magnitude is not None and tok.value >= last_flushed_magnitude:
                ambiguous_reason = f"digit run '{tok.raw}' ({tok.value}) is at or above the previously resolved magnitude ({last_flushed_magnitude}) -- possible duplicate/self-correction, not a remainder"
                break
            current += tok.value
            prev_kind = tok.kind
            i += 1
            continue

        if tok.kind == 'COMPOUND':
            seen_any = True
            result += current
            result += tok.value
            current = 0
            prev_kind = tok.kind
            i += 1
            continue

        if tok.kind == 'MAGNITUDE' and tok.value is not None:
            seen_any = True
            val = tok.value
            if val == 100:
                current = (current or 1) * 100
            else:
                if last_flushed_magnitude is not None and val >= last_flushed_magnitude:
                    ambiguous_reason = f"magnitude '{tok.raw}' ({val}) is not smaller than the previous magnitude ({last_flushed_magnitude}) -- invalid order or duplicate"
                    break
                current = (current or 1) * val
                result += current
                current = 0
                last_flushed_magnitude = val
            prev_kind = tok.kind
            i += 1
            continue

        if tok.kind == 'WORD' and tok.value is not None:
            seen_any = True
            val = tok.value
            if current and current % 100 == 0 and val < 100:
                current += val
            elif current and current % 10 == 0 and current >= 20 and val < 10:
                current += val
            else:
                current += val
            prev_kind = tok.kind
            i += 1
            continue

        # OTHER / DECIMAL_MARK reaching here means caller mis-split; treat as
        # a hard boundary (shouldn't normally happen inside parse_tokens).
        break

    text = ' '.join(t.raw for t in tokens[:i]) if i < n else ' '.join(t.raw for t in tokens)

    if ambiguous_reason:
        return NumberResult(text=text, value=None, ambiguous=True, reason=ambiguous_reason, tokens=tokens[:i])

    if not seen_any:
        return NumberResult(text=text, value=None, ambiguous=False, reason='no number found', tokens=tokens)

    final = result + current
    if negative:
        final = -final
    return NumberResult(text=text, value=final, ambiguous=False, tokens=tokens[:i])


def _split_decimal(tokens: List[Token]) -> Tuple[List[Token], Optional[List[Token]]]:
    for idx, t in enumerate(tokens):
        if t.kind == 'DECIMAL_MARK':
            return tokens[:idx], tokens[idx + 1:]
    return tokens, None


def parse_phrase(tokens: List[Token]) -> NumberResult:
    int_part_tokens, frac_tokens = _split_decimal(tokens)
    int_result = parse_tokens(int_part_tokens)

    if frac_tokens is None:
        return int_result
    if int_result.ambiguous:
        return int_result

    # fractional part: digits spoken one at a time is the common ASR pattern
    # ("point five" -> .5, "point two five" -> .25)
    frac_digits = []
    for t in frac_tokens:
        if t.kind == 'DIGIT' and len(t.raw) == 1:
            frac_digits.append(t.raw)
        elif t.kind == 'WORD' and t.value is not None and 0 <= t.value <= 9:
            frac_digits.append(str(t.value))
        else:
            return NumberResult(
                text=' '.join(x.raw for x in tokens), value=None, ambiguous=True,
                reason=f"unrecognized fractional-part token '{t.raw}'", tokens=tokens,
            )
    if not frac_digits:
        return NumberResult(text=' '.join(x.raw for x in tokens), value=None, ambiguous=True,
                             reason='decimal marker with no fractional digits', tokens=tokens)

    int_val = int_result.value if int_result.value is not None else 0
    dec = Decimal(f"{int_val}.{''.join(frac_digits)}")
    return NumberResult(text=' '.join(x.raw for x in tokens), value=dec, ambiguous=False, tokens=tokens)


# ======================================================================
# 9. NUMBER-PHRASE BOUNDARY EXTRACTION (multiple numbers per sentence)
# ======================================================================

def extract_numbers(text: str) -> List[NumberResult]:
    """
    Splits on OTHER tokens (real, non-number words) and on adjacent raw
    digit-run boundaries, parsing each contiguous number-phrase separately.
    "five kg at twenty five rupees" -> [5, 25], never 30.
    """
    tokens = tokenize(text)
    results: List[NumberResult] = []
    group: List[Token] = []
    prev_was_digit = False

    def flush():
        nonlocal group
        if group:
            res = parse_phrase(group)
            if res.value is not None or res.ambiguous:
                results.append(res)
        group = []

    for tok in tokens:
        if tok.kind == 'OTHER':
            flush()
            continue
        if tok.kind == 'DIGIT' and prev_was_digit:
            flush()  # two raw digit runs back-to-back = two numbers
        group.append(tok)
        prev_was_digit = (tok.kind == 'DIGIT')

    flush()
    return results


def text_to_number(text: str) -> Optional[Union[int, Decimal]]:
    """Convenience wrapper: assumes the whole input is a single number
    phrase and returns None for no-match OR ambiguous input alike. Use
    extract_numbers()/parse_phrase() directly if you need to distinguish
    'not a number' from 'ambiguous' or need multiple numbers."""
    results = extract_numbers(text)
    if not results:
        return None
    return results[0].value
