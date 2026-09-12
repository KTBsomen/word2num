# -*- coding: utf-8 -*-
"""
Deterministic multilingual ASR number parser (Bengali / Hindi / English / mixed).
Regional-first architecture with deep native script support (Bengali script & Devanagari script).
Designed for 1:1 portability to Flutter / Dart.

Architecture:
  raw text
    -> digit normalization (Bengali/Devanagari/Arabic-Indic -> ASCII)
    -> text cleaning & regex tokenization (strip currency symbols ₹, $, ৳,
       normalize punctuation, cleanly split attached units like 10lakh -> 10 lakh, ১০টা -> 10 টা)
    -> token classification:
         - exact match via compiled inverted dictionary O(1)
         - structural compound-hundred recognition (e.g. বারোশো -> 1200, পাঁচশ -> 500, बारह सौ -> 1200)
         - short-word dedicated fuzzy matcher (lengths 2-4, stopword-protected)
         - phonetic akshara-to-roman transliteration + unified fuzzy matcher for longer words
    -> hierarchical grammar accumulation + decimal magnitudes ("২.৫ লাখ" -> 250000)
       + sequential/paired number concatenation ("twenty thirty" -> 2030, "চার শূন্য দুই" -> 402)
    -> result: resolved value OR explicit ambiguous/unresolved flag.

No ML model, no network calls, no threadpools, no external dependencies.
All maps use the value: [variants] format for effortless maintenance and expansion.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# ======================================================================
# 1. DIGIT NORMALIZATION & TEXT CLEANING
# ======================================================================

_BENGALI_DIGITS = str.maketrans('০১২৩৪৫৬৭৮৯', '0123456789')
_DEVANAGARI_DIGITS = str.maketrans('०१२३४५६७८९', '0123456789')
_ARABIC_INDIC_DIGITS = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')


def normalize_digits(text: str) -> str:
    """Normalize any non-ASCII Unicode digits into standard ASCII 0-9."""
    text = text.translate(_BENGALI_DIGITS)
    text = text.translate(_DEVANAGARI_DIGITS)
    text = text.translate(_ARABIC_INDIC_DIGITS)
    return text


def normalize_indic_chars(text: str) -> str:
    """Canonicalize decomposed nukta pairs into single precomposed Indic characters."""
    text = text.replace('\u09a1\u09bc', '\u09dc')  # ড় -> ড়
    text = text.replace('\u09a2\u09bc', '\u09dd')  # ঢ় -> ঢ়
    text = text.replace('\u09af\u09bc', '\u09df')  # য় -> য়
    text = text.replace('\u0921\u093c', '\u095c')  # ड़ -> ड़
    text = text.replace('\u0922\u093c', '\u095d')  # ढ़ -> ढ़
    text = text.replace('\u092b\u093c', '\u095e')  # फ़ -> फ़
    text = text.replace('\u092f\u093c', '\u095f')  # य़ -> य़
    return text


def clean_and_normalize_text(text: str) -> str:
    """
    Clean formatting noise, symbols, and attached currency/units:
    - Normalizes non-ASCII digits to ASCII
    - Canonicalizes decomposed Indic characters
    - Strips currency symbols (₹, $, €, £, ৳, ¥)
    - Separates currency abbreviations attached to digits (Rs.500 -> 500)
    - Strips trailing slash-hyphens (500/- -> 500)
    - Removes thousand commas (1,25,000 -> 125000)
    - Splits attached units/magnitudes from digits (10lakh -> 10 lakh, 50kg -> 50 kg, 5ta -> 5 ta, ১০টা -> 10 টা)
    - Strips noise punctuation like (), [], {}, quotes, !, ? without disturbing decimals
    """
    text = normalize_digits(text)
    text = normalize_indic_chars(text)

    # Remove currency abbreviations attached to digits: Rs. 500, Rs.500, INR500, Tk.500
    text = re.sub(r'(?i)\b(?:rs\.?|inr|tk\.?|usd)\s*(?=\d)', ' ', text)
    # Remove trailing slash-hyphen: 500/- or 500/ -
    text = re.sub(r'/\s*-\s*', ' ', text)
    # Remove currency symbols: ₹, $, €, £, ৳, ¥
    text = re.sub(r'[₹$€£৳¥]', ' ', text)

    # Internal thousands commas in numbers: "1,25,000" -> "125000", "2,500" -> "2500"
    text = re.sub(r'(?<=\d),(?=\d)', '', text)

    # Hyphenation between words: "twenty-five" -> "twenty five"
    text = text.replace('-', ' ')

    # Boundary between digits and letters: "10lakh" -> "10 lakh", "50kg" -> "50 kg", "5ta" -> "5 ta", "১০টা" -> "10 টা"
    text = re.sub(r'(?<=\d)(?=[a-zA-Z\u0900-\u09FF])', ' ', text)

    # Strip noisy enclosing/trailing punctuation: (), [], {}, "", '', !, ?, ;, :, ~
    text = re.sub(r'[\(\)\[\]\{\}"\'!?;:~*]', ' ', text)

    # Remove periods and commas that are NOT internal decimals (e.g. "five hundred." -> "five hundred ")
    text = re.sub(r'(?<=[^\d])\.(?=[^\d]|$)', ' ', text)
    text = re.sub(r'(?<=\d)\.(?=[^\d]|$)', ' ', text)
    text = re.sub(r'(?<=[^\d])\.(?=\d)', ' point ', text)  # standalone ".5" -> "point 5"
    text = re.sub(r',', ' ', text)

    # Collapse multiple whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_punctuation(text: str) -> str:
    """Backward-compatible wrapper."""
    return clean_and_normalize_text(text)


# ======================================================================
# 2. PHONETIC AKSHARA-TO-ROMAN TRANSLITERATION
# ======================================================================

_BENGALI_TRANSLIT = {
    'অ': 'a', 'আ': 'a', 'ই': 'i', 'ঈ': 'i', 'উ': 'u', 'ঊ': 'u',
    'ঋ': 'ri', 'এ': 'e', 'ঐ': 'oi', 'ও': 'o', 'ঔ': 'ou',
    'ক': 'k', 'খ': 'kh', 'গ': 'g', 'ঘ': 'gh', 'ঙ': 'ng',
    'চ': 'ch', 'ছ': 'chh', 'জ': 'j', 'ঝ': 'jh', 'ঞ': 'n',
    'ট': 't', 'ঠ': 'th', 'ড': 'd', 'ঢ': 'dh', 'ণ': 'n',
    'ত': 't', 'থ': 'th', 'দ': 'd', 'ধ': 'dh', 'ন': 'n',
    'প': 'p', 'ফ': 'ph', 'ব': 'b', 'ভ': 'bh', 'ম': 'm',
    'য': 'j', 'র': 'r', 'ল': 'l', 'শ': 'sh', 'ষ': 'sh',
    'স': 's', 'হ': 'h',
    '\u09dc': 'r', '\u09dd': 'rh', '\u09df': 'y',  # Precomposed ড়, ঢ়, য়
    '\u09bc': '',  # Bengali Nukta
    'ড়': 'r', 'ঢ়': 'rh', 'য়': 'y',
    'ৎ': 't',
    'া': 'a', 'ি': 'i', 'ী': 'i', 'ু': 'u', 'ূ': 'u', 'ৃ': 'ri',
    'ে': 'e', 'ৈ': 'oi', 'ো': 'o', 'ৌ': 'ou', 'ং': 'ng',
    'ঃ': 'h', '্': '', 'ঁ': '',
}

_DEVANAGARI_TRANSLIT = {
    'अ': 'a', 'आ': 'a', 'इ': 'i', 'ई': 'i', 'उ': 'u', 'ऊ': 'u',
    'ऋ': 'ri', 'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
    'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'n',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'श': 'sh', 'ष': 'sh',
    'स': 's', 'ह': 'h',
    '\u095c': 'r', '\u095d': 'rh', '\u095e': 'ph', '\u095f': 'y',  # Precomposed ड़, ढ़, फ़, य़
    '\u093c': '',  # Devanagari Nukta
    'ड़': 'r', 'ढ़': 'rh',
    'ा': 'a', 'ि': 'i', 'ी': 'i', 'ु': 'u', 'ू': 'u', 'ृ': 'ri',
    'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au', 'ं': 'n',
    'ः': 'h', '्': '', 'ँ': '',
}


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


def transliterate_to_roman(token: str) -> str:
    """Best-effort phonetic skeleton for matching regional script words."""
    token = normalize_indic_chars(token)
    script = script_of(token)
    table = _BENGALI_TRANSLIT if script == 'bengali' else (
        _DEVANAGARI_TRANSLIT if script == 'devanagari' else None
    )
    if table is None:
        return token.lower()
    return ''.join(table.get(ch, ch if ch.isascii() else '') for ch in token)


# ======================================================================
# 3. REGIONAL-FIRST VOCABULARIES (0-99, MAGNITUDES, COMPOUNDS)
# ======================================================================

# Complete Bengali 0-99: Standard Bengali Script, Colloquial Classifiers, Romanized, ASR variants
BENGALI_VOCAB: Dict[int, List[str]] = {
    0: ['শূন্য', 'shunno', 'shunya', 'জিরো', 'zero', 'jero'],
    1: ['এক', 'একটা', 'একটি', 'ek', 'ekta', 'ekti', 'aek', 'ak', 'ik', 'এ্যাক'],
    2: ['দুই', 'দুটো', 'দুইটি', 'দুটি', 'দু', 'dui', 'duto', 'duti', 'du', 'duy', 'dwi'],
    3: ['তিন', 'তিনটে', 'তিনটি', 'তীন', 'tin', 'tinte', 'tinti', 'teen', 'tyn', 'tien'],
    4: ['চার', 'চারটে', 'চারটি', 'চারি', 'char', 'charte', 'charti', 'chaar', 'caar'],
    5: ['পাঁচ', 'পাচ', 'পাঁচটা', 'পাচটা', 'পাঁচটি', 'পাচটি', 'panch', 'paanch', 'pach', 'pachta'],
    6: ['ছয়', 'ছয়', 'ছটা', 'ছটি', 'ছ', 'choy', 'chhota', 'chhoy', 'coy'],
    7: ['সাত', 'সাতটা', 'সাতটি', 'shaat', 'saat', 'satta', 'sat', 'shat'],
    8: ['আট', 'আটটা', 'আটটি', 'আঠ', 'aat', 'ath', 'aath', 'atta'],
    9: ['নয়', 'নয়', 'নটা', 'নটি', 'noy', 'nota', 'noi', 'nay'],
    10: ['দশ', 'দশটা', 'দশটি', 'dosh', 'dash', 'dosta', 'dos'],
    11: ['এগারো', 'এগার', 'এগাড়ো', 'এগাড়', 'egaro', 'egro'],
    12: ['বারো', 'বার', 'বাড়ো', 'বাড়', 'baro', 'baaro'],
    13: ['তেরো', 'তের', 'তেড়ো', 'তেড়'],  # Native script 'তেরো' is 13; roman 'tero' guarded as ambiguous per spec
    14: ['চৌদ্দ', 'চোদ্দ', 'চৌদ্দটা', 'চোদ্দটা', 'choddo', 'chowddo'],
    15: ['পনেরো', 'পনের', 'পনেড়ো', 'পনেড়', 'ponero', 'ponro'],
    16: ['ষোল', 'শোল', 'sholo', 'sholoh'],
    17: ['সতেরো', 'সতের', 'সতেড়ো', 'সতেড়', 'sotero', 'satero', 'shotero'],
    18: ['আঠারো', 'আঠার', 'আঠারটা', 'আঠারটি', 'আঠাড়ো', 'আঠাড়', 'আঠাড়টা', 'আঠাড়টি', 'atharo', 'athero', 'attharo', 'athar', 'aatharo', 'aathar', 'atharah'],
    19: ['উনিশ', 'unish', 'unnish', 'unis', 'unnis'],
    20: ['বিশ', 'বিষ', 'বিস', 'কুড়ি', 'কুড়ি', 'bish', 'bees', 'kuri', 'bis'],
    21: ['একুশ', 'একুশটা', 'ekush', 'ekus', 'ekuus', 'ekkus'],
    22: ['বাইশ', 'baish', 'bais', 'baees'],
    23: ['তেইশ', 'teish', 'teis'],
    24: ['চব্বিশ', 'chobbish', 'chobbis', 'cobbish', 'cobbis'],
    25: ['পঁচিশ', 'পচিশ', 'পচিষ', 'পচিস', 'পঁচিশটা', 'পচিশটা', 'pochish', 'pochis', 'pachish', 'pachis'],
    26: ['ছাব্বিশ', 'chhabbish', 'chhabbis', 'chabbish', 'chabbis'],
    27: ['সাতাশ', 'shatash', 'satash', 'satas', 'shatas'],
    28: ['আটাশ', 'আটাশটা', 'আটাশটি', 'আঠাশ', 'atash', 'athash', 'athas', 'athass', 'atass', 'aathash', 'aathas', 'atas', 'aatash', 'aatas'],
    29: ['ঊনত্রিশ', 'unotrish', 'unotris'],
    30: ['ত্রিশ', 'ত্রিষ', 'ত্রিস', 'তিরিশ', 'তিরিস', 'trish', 'tirish', 'tris', 'tiris'],
    31: ['একত্রিশ', 'ektrish', 'ektris'],
    32: ['বত্রিশ', 'batrish', 'batris'],
    33: ['তেত্রিশ', 'tetrish', 'tetris'],
    34: ['চৌত্রিশ', 'choutrish', 'choutris'],
    35: ['পঁয়ত্রিশ', 'পয়ত্রিশ', 'পয়ত্রিশ', 'poytrish', 'poytris'],
    36: ['ছত্রিশ', 'chotrish', 'chotris'],
    37: ['সাঁইত্রিশ', 'সাইত্রিশ', 'saintrish', 'saintris'],
    38: ['আটত্রিশ', 'attrish', 'attris'],
    39: ['ঊনচল্লিশ', 'unochollish', 'unochollis'],
    40: ['চল্লিশ', 'চলিষ', 'চলিস', 'chollish', 'chollis'],
    41: ['একচল্লিশ', 'ekchollish', 'ekchollis'],
    42: ['বিয়াল্লিশ', 'biallish', 'biallis'],
    43: ['তেতাল্লিশ', 'tetallish', 'tetallis'],
    44: ['চুয়াল্লিশ', 'chuallish', 'chuallis'],
    45: ['পঁয়তাল্লিশ', 'পয়তাল্লিশ', 'পয়তাল্লিশ', 'poytallish', 'poytallis'],
    46: ['ছেচল্লিশ', 'checchollish', 'checchollis'],
    47: ['সাতচল্লিশ', 'shatchollish', 'shatchollis', 'satchollish'],
    48: ['আটচল্লিশ', 'atchollish', 'atchollis'],
    49: ['ঊনপঞ্চাশ', 'unponchash', 'unponchas'],
    50: ['পঞ্চাশ', 'পচাশ', 'পঞ্চাষ', 'পঞ্চাস', 'ponchash', 'ponchas', 'pachas', 'pachash'],
    51: ['একান্ন', 'ekanno'],
    52: ['বাহান্ন', 'bahanno'],
    53: ['তেপ্পান্ন', 'teppanno'],
    54: ['চুয়ান্ন', 'chuanno'],
    55: ['পঞ্চান্ন', 'পচান্ন', 'ponchanno'],
    56: ['ছাপ্পান্ন', 'chappanno'],
    57: ['সাতান্ন', 'shatanno'],
    58: ['আটান্ন', 'atanno'],
    59: ['ঊনষাট', 'unshaat'],
    60: ['ষাট', 'শাট', 'সাট', 'sháth', 'shaath'],
    61: ['একষট্টি', 'ekshotti'],
    62: ['বাষট্টি', 'bashotti'],
    63: ['তেষট্টি', 'teshotti'],
    64: ['চৌষট্টি', 'choushotti'],
    65: ['পঁয়ষট্টি', 'পয়ষট্টি', 'পয়ষট্টি', 'poyshotti'],
    66: ['ছেষট্টি', 'cheshotti'],
    67: ['সাতষট্টি', 'shatshotti'],
    68: ['আটষট্টি', 'atshotti'],
    69: ['ঊনসত্তর', 'unshottor'],
    70: ['সত্তর', 'শত্তর', 'shottor'],
    71: ['একাত্তর', 'ekattor'],
    72: ['বাহাত্তর', 'bahattor'],
    73: ['তিয়াত্তর', 'tiattor'],
    74: ['চুয়াত্তর', 'chuattor'],
    75: ['পঁচাত্তর', 'পচাত্তর', 'pochattor'],
    76: ['ছিয়াত্তর', 'chiattor'],
    77: ['সাতাত্তর', 'shatattor'],
    78: ['আটাত্তর', 'atattor'],
    79: ['ঊনআশি', 'unashi'],
    80: ['আশি', 'আষি', 'আসি', 'ashi'],
    81: ['একাশি', 'ekashi'],
    82: ['বিরাশি', 'birashi'],
    83: ['তিরাশি', 'tirashi'],
    84: ['চুরাশি', 'churashi'],
    85: ['পঁচাশি', 'পচাশি', 'pochashi'],
    86: ['ছিয়াশি', 'chiyashi'],
    87: ['সাতাশি', 'shatashi'],
    88: ['আটাশি', 'atashi'],
    89: ['ঊননব্বই', 'unonobboi'],
    90: ['নব্বই', 'nobboi'],
    91: ['একানব্বই', 'ekanobboi'],
    92: ['বিরানব্বই', 'biranobboi'],
    93: ['তিরানব্বই', 'tiranobboi'],
    94: ['চুরানব্বই', 'churanobboi'],
    95: ['পঁচানব্বই', 'পচানব্বই', 'pochanobboi'],
    96: ['ছিয়ানব্বই', 'chiyanobboi'],
    97: ['সাতানব্বই', 'shatanobboi'],
    98: ['আটানব্বই', 'atanobboi'],
    99: ['নিরানব্বই', 'niranobboi'],
}

# Complete Hindi 0-99: Standard Devanagari Script, Romanized, ASR variants
HINDI_VOCAB: Dict[int, List[str]] = {
    0: ['शून्य', 'shunya', 'shunyo', 'ज़ीरो', 'जीरो', 'zero'],
    1: ['एक', 'ek', 'aek', 'ik', 'ak', 'ek tho', 'ek thoh'],
    2: ['दो', 'do', 'du', 'doh', 'do tho'],
    3: ['तीन', 'teen', 'tin', 'teen tho', 'tyn'],
    4: ['चार', 'char', 'chaar', 'caar'],
    5: ['पाँच', 'पांच', 'पाच', 'panch', 'paanch', 'pach'],
    6: ['छह', 'छः', 'छ', 'chhe', 'che'],
    7: ['सात', 'saat', 'sat'],
    8: ['आठ', 'aath', 'ath', 'aat'],
    9: ['नौ', 'nau', 'now'],
    10: ['दस', 'das', 'dus', 'dos'],
    11: ['ग्यारह', 'gyarah', 'gyaarah'],
    12: ['बारह', 'barah', 'baarah'],
    13: ['तेरह', 'terah', 'terha'],
    14: ['चौदह', 'chaudah', 'chodah'],
    15: ['पंद्रह', 'पंद्रा', 'pandrah', 'pandra'],
    16: ['सोलह', 'सोला', 'solah', 'sola'],
    17: ['सत्रह', 'सतरे', 'satrah', 'saterah'],
    18: ['अठारह', 'अठारा', 'atharah', 'atherah'],
    19: ['उन्नीस', 'unnis', 'unnees'],
    20: ['बीस', 'bees', 'bis'],
    21: ['इक्कीस', 'ikkis', 'ekis', 'ikhees'],
    22: ['बाईस', 'baees', 'bais'],
    23: ['तेईस', 'teis'],
    24: ['चौबीस', 'chaubis'],
    25: ['पच्चीस', 'pachis', 'pachees', 'pachish'],
    26: ['छब्बीस', 'chhabbis'],
    27: ['सत्ताईस', 'sattais'],
    28: ['अट्ठाईस', 'अट्ठाइस', 'अठाईस', 'अठाइस', 'अठास', 'atthais', 'athais', 'athas', 'athass', 'atass', 'atthas', 'attais', 'aathais', 'athaas'],
    29: ['उनतीस', 'unatis', 'unantis'],
    30: ['तीस', 'tees', 'tis'],
    31: ['इकतीस', 'ikatis'],
    32: ['बत्तीस', 'battis'],
    33: ['तैंतीस', 'taintis'],
    34: ['चौंतीस', 'chauntis'],
    35: ['पैंतीस', 'paintis'],
    36: ['छत्तीस', 'chhattis'],
    37: ['सैंतीस', 'saintis'],
    38: ['अड़तीस', 'adtis', 'artis'],
    39: ['उनतालीस', 'unchalis'],
    40: ['चालीस', 'chalis', 'chaalis'],
    41: ['इकतालीस', 'iktalis'],
    42: ['बयालीस', 'bayalis'],
    43: ['तैंतालीस', 'taintalis'],
    44: ['चौवालीस', 'chauwalis'],
    45: ['पैंतालीस', 'paintalis'],
    46: ['छियालीस', 'chhiyalis'],
    47: ['सैंतालीस', 'saintalis'],
    48: ['अड़तालीस', 'adtalis'],
    49: ['उनचास', 'unchaas', 'unchas'],
    50: ['पचास', 'pachas', 'pachash'],
    51: ['इक्यावन', 'ikyawan'],
    52: ['बावन', 'bawan'],
    53: ['तिरपन', 'tirpan'],
    54: ['चौवन', 'chauwan'],
    55: ['पचपन', 'pachpan'],
    56: ['छप्पन', 'chhappan'],
    57: ['सत्तावन', 'sattawan'],
    58: ['अट्ठावन', 'atthawan'],
    59: ['उनसठ', 'unsath'],
    60: ['साठ', 'saath', 'sath'],
    61: ['इकसठ', 'iksath'],
    62: ['बासठ', 'baasath'],
    63: ['तिरसठ', 'tirsath'],
    64: ['चौंसठ', 'chausath'],
    65: ['पैंसठ', 'painsath'],
    66: ['छियासठ', 'chhiyasath'],
    67: ['सड़सठ', 'sadsath'],
    68: ['अड़सठ', 'adsath'],
    69: ['उनहत्तर', 'unhattar'],
    70: ['सत्तर', 'sattar'],
    71: ['इकहत्तर', 'ikhattar'],
    72: ['बहत्तर', 'bahattar'],
    73: ['तिहत्तर', 'tihattar'],
    74: ['चौहत्तर', 'chauhattar'],
    75: ['पचहत्तर', 'pachhattar'],
    76: ['छिहत्तर', 'chhihattar'],
    77: ['सतहत्तर', 'sathattar'],
    78: ['अठहत्तर', 'athhattar'],
    79: ['उन्यासी', 'unyasi'],
    80: ['अस्सी', 'assi', 'asi'],
    81: ['इक्यासी', 'ikyasi'],
    82: ['बयासी', 'bayasi'],
    83: ['तिरासी', 'tirasi'],
    84: ['चौरासी', 'chaurasi'],
    85: ['पचासी', 'pachasi'],
    86: ['छियासी', 'chhiyasi'],
    87: ['सतासी', 'satasi'],
    88: ['अट्ठासी', 'atthasi'],
    89: ['नवासी', 'navasi'],
    90: ['नब्बे', 'nabbe'],
    91: ['इक्यानवे', 'ikyanave'],
    92: ['बानवे', 'baanve'],
    93: ['तिरानवे', 'tiranve'],
    94: ['चौरानवे', 'chauranve'],
    95: ['पंचानवे', 'pachanve'],
    96: ['छियानवे', 'chhiyanve'],
    97: ['सत्तानवे', 'sattanve'],
    98: ['अट्ठानवे', 'atthanve'],
    99: ['निन्यानवे', 'ninyanve'],
}

ENGLISH_VOCAB: Dict[int, List[str]] = {
    0: ['zero', 'nil', 'null', 'nought', 'oh'],
    1: ['one', 'wan'],
    2: ['two'],
    3: ['three', 'tree'],
    4: ['four', 'foor'],
    5: ['five', 'fiv'],
    6: ['six'],
    7: ['seven'],
    8: ['eight', 'eigt', 'aight'],
    9: ['nine', 'nein'],
    10: ['ten'],
    11: ['eleven'],
    12: ['twelve'],
    13: ['thirteen'],
    14: ['fourteen'],
    15: ['fifteen', 'fiveteen'],
    16: ['sixteen'],
    17: ['seventeen'],
    18: ['eighteen'],
    19: ['nineteen'],
    20: ['twenty'],
    21: ['twenty one', 'twenty-one', 'twentyone'],
    22: ['twenty two', 'twenty-two', 'twentytwo'],
    23: ['twenty three', 'twenty-three', 'twentythree'],
    24: ['twenty four', 'twenty-four', 'twentyfour'],
    25: ['twenty five', 'twenty-five', 'twentyfive'],
    26: ['twenty six', 'twenty-six', 'twentysix'],
    27: ['twenty seven', 'twenty-seven', 'twentyseven'],
    28: ['twenty eight', 'twenty-eight', 'twentyeight'],
    29: ['twenty nine', 'twenty-nine', 'twentynine'],
    30: ['thirty'],
    40: ['forty', 'fourty'],
    50: ['fifty'],
    60: ['sixty'],
    70: ['seventy'],
    80: ['eighty'],
    90: ['ninety', 'ninty'],
}

PHONETIC_ENGLISH_VOCAB: Dict[int, List[str]] = {
    0: ['জিরো', 'ज़ीरो', 'जीरो'],
    1: ['ওয়ান', 'वन'],
    2: ['টু', 'टू'],
    3: ['থ্রি', 'थ्री'],
    4: ['ফোর', 'फोर'],
    5: ['ফাইভ', 'फाइव'],
    6: ['সিক্স', 'सिक्स'],
    7: ['সেভেন', 'सेवन'],
    8: ['এইট', 'एट'],
    9: ['নাইন', 'नाइन'],
    10: ['টেন', 'टेन'],
    11: ['ইলেভেন', 'इलेवन'],
    12: ['টুয়েলভ', 'ट्वेल्व'],
    13: ['থার্টিন', 'थर्टीन'],
    14: ['ফোরটিন', 'फोर्टीन'],
    15: ['ফিফটিন', 'फिफ्टीन'],
    16: ['সিক্সটিন', 'सिक्सटीन'],
    17: ['সেভেনটিন', 'सेवनटीन'],
    18: ['এইটিন', 'एटीन'],
    19: ['নাইনটিন', 'नाइनटीन'],
    20: ['টুয়েন্টি', 'ट्वेंटी'],
    30: ['থার্টি', 'थर्टी'],
    40: ['ফোর্টি', 'फोर्टी'],
    50: ['ফিফটি', 'फिफ्टी'],
    60: ['সিক্সটি', 'सिक्सटी'],
    70: ['সেভেন্টি', 'सेवंटी'],
    80: ['এইটি', 'एटी'],
    90: ['নাইন্টি', 'नाइंटी'],
}

MAGNITUDE_VOCAB: Dict[int, List[str]] = {
    100: [
        'hundred', 'hundered', 'hundread', 'sau', 'so', 'sho', 'shô',
        'सौ', 'শ', 'শত', 'শো', 'একশো', 'একশ', 'একশোটা', 'একশটা', 'একশত', 'হান্ড্রেড', 'हंड्रेड',
    ],
    1000: [
        'thousand', 'thousnd', 'thosand', 'hazar', 'hazaar', 'hajar', 'hajaar', 'hazr',
        'হাজার', 'हज़ार', 'हजार', 'থাউজেন্ড', 'थाउजेंड',
    ],
    100_000: [
        'lakh', 'lakhs', 'lac', 'lacs', 'লাখ', 'লক্ষ', 'लाख',
    ],
    1_000_000: [
        'million', 'millions', 'মিলিয়ন', 'मिलियन',
    ],
    10_000_000: [
        'crore', 'crores', 'cr', 'কোটি', 'করোড়', 'करोड़',
    ],
    1_000_000_000: [
        'billion', 'billions', 'বিলিয়ন', 'बिलियन',
    ],
}

FRACTION_MULTIPLIERS: Dict[float, List[str]] = {
    0.5: ['half', 'aadha', 'adha', 'adho', 'आधा', 'আধা'],
    1.25: ['sawa', 'sawaa', 'सवा', 'সওয়া', 'সোয়া', 'সওয়া'],
    1.5: ['dedh', 'derh', 'deyrh', 'डेढ़', 'দেড়', 'দের', 'দেড়'],
    2.5: ['dhai', 'dhaai', 'arai', 'ढाई', 'আড়াই', 'আরাই', 'আড়াই'],
}

# Complete Native Bengali & Hindi Compound Hundreds
COMPOUND_HUNDREDS: Dict[int, List[str]] = {
    150: ['দেড়শো', 'দেড়শো', 'দেড়শ', 'দেড়শ', 'derhso', 'derh sho', 'dedhso', 'डेढ़ सौ', 'डेढ़सौ', 'dedh sau'],
    250: ['আড়াইশো', 'আড়াইশো', 'আড়াইশ', 'আড়াইশ', 'araisho', 'arai sho', 'dhaiso', 'ढाई सौ', 'ढाईसौ', 'dhai sau'],
    1100: ['এগারোশো', 'এগারশো', 'এগারশ', 'egarosho', 'egaro sho', 'ग्यारह सौ', 'ग्यारहसौ', 'gyarah sau'],
    1200: ['বারোশো', 'বারশো', 'বারশ', 'barosho', 'baro sho', 'baroso', 'बारह सौ', 'बारहसौ', 'barah sau'],
    1300: ['তেরোশো', 'তেরশো', 'তেরশ', 'তেল্লশো', 'तेरह सौ', 'तेरहसौ', 'terah sau'],
    1400: ['চৌদ্দশো', 'চোদ্দশো', 'চোদ্দশ', 'choddo sho', 'चौदह सौ', 'चौदहसौ', 'chaudah sau'],
    1500: ['পনেরোশো', 'পনেরশো', 'পনেরশ', 'ponerosho', 'ponero sho', 'पंद्रह सौ', 'पंद्रहसौ', 'pandrah sau'],
    1600: ['ষোলশো', 'ষোলশ', 'sholosho', 'sholo sho', 'सोलह सौ', 'सोलहसौ', 'solah sau'],
    1700: ['সতেরোশো', 'সতেরশো', 'সতেরশ', 'saterosho', 'satero sho', 'सत्रह सौ', 'सत्रहसौ', 'satrah sau'],
    1800: ['আঠারোশো', 'আঠারশো', 'আঠারশ', 'atheroso', 'atharo sho', 'अठारह सौ', 'अठारहसौ', 'atharah sau'],
    1900: ['উনিশশো', 'উনিশশ', 'unish sho', 'उन्नीस सौ', 'उन्नीसौ', 'unnis sau'],
    2000: ['বিশশো', 'কুড়িশো', 'কুড়িশো', 'bish sho', 'बीस सौ', 'बीसौ', 'bees sau'],
    2100: ['একুশশো', 'একুশশ', 'ekusho', 'ekusso', 'ekush sho', 'इक्कीस सौ', 'इक्कीसौ', 'ikkis sau'],
    2200: ['বাইশশো', 'বাইশশ', 'baisho', 'baisso', 'baish sho', 'बाईस सौ', 'बाईसौ', 'bais sau'],
    2300: ['তেইশশো', 'তেইশশ', 'teisho', 'teisso', 'teish sho', 'तेईस सौ', 'तेईसौ', 'teis sau'],
    2400: ['চব্বিশশো', 'চব্বিশশ', 'chobbisho', 'chobbisso', 'chobbish sho', 'चौबीस सौ', 'चौबीसौ', 'chaubis sau'],
    2500: ['পঁচিশশো', 'পচিশশো', 'পঁচিশশ', 'পচিশশ', 'pochish sho', 'pochis sho', 'pochisho', 'pochisso', 'पच्चीस सौ', 'पच्चीसौ', 'pachis sau'],
    2600: ['ছাব্বিশশো', 'ছাব্বিশশ', 'chhabbisho', 'chhabbisso', 'chhabbish sho', 'छब्बीस सौ', 'छब्बीसौ', 'chhabbis sau'],
    2700: ['সাতাশশো', 'সাতাশশ', 'shatasho', 'shatasso', 'satasho', 'satasso', 'shatash sho', 'सत्ताईस सौ', 'सत्ताईसौ', 'sattais sau'],
    2800: ['আটাশশো', 'আটাশশ', 'আঠাশশো', 'atashsho', 'athassho', 'athasso', 'atassho', 'atasho', 'athasho', 'atash sho', 'athas sho', 'अट्ठाईस सौ', 'अट्ठाईसौ', 'atthais sau', 'athas sau'],
    2900: ['ঊনত্রিশশো', 'ঊনত্রিশশ', 'unotrisho', 'unotrisso', 'unotrish sho', 'उनतीस सौ', 'उनतीसौ', 'unatis sau'],
}

REPEATERS: Dict[str, int] = {
    'double': 2, 'ডাবল': 2, 'डबल': 2, 'dooble': 2, 'dabol': 2,
    'triple': 3, 'ট্রিপল': 3, 'ट्रिपल': 3, 'tripple': 3,
}

FRACTION_PREFIXES: Dict[str, Tuple[str, float]] = {
    'সাড়ে': ('SADE', 0.5), 'সাড়ে': ('SADE', 0.5),
    'साढ़े': ('SADE', 0.5), 'साढे': ('SADE', 0.5),
    'saade': ('SADE', 0.5), 'sade': ('SADE', 0.5),
    'saadhe': ('SADE', 0.5), 'sadhe': ('SADE', 0.5),
    'saare': ('SADE', 0.5), 'sare': ('SADE', 0.5),

    'পৌনে': ('PAUNE', -0.25), 'পনে': ('PAUNE', -0.25), 'পৌনেটা': ('PAUNE', -0.25),
    'पौने': ('PAUNE', -0.25), 'पउने': ('PAUNE', -0.25),
    'paune': ('PAUNE', -0.25), 'pone': ('PAUNE', -0.25),

    'সওয়া': ('SAWA', 0.25), 'সওয়া': ('SAWA', 0.25), 'সোয়া': ('SAWA', 0.25),
    'सवा': ('SAWA', 0.25), 'sawa': ('SAWA', 0.25), 'sowa': ('SAWA', 0.25),
}

CONNECTORS = {'and', 'plus', 'with', 'ও', 'আর', 'এবং', 'aur', 'और', 'तथा', 'एवं'}
DECIMAL_MARKERS = {'point', 'dot', 'দশমিক', 'दशमलव'}
NEGATIVE_MARKERS = {'minus', 'negative', 'মাইনাস', 'माइनस'}

STOPWORDS = {
    'taka', 'টাকা', 'rupee', 'rupees', 'rupaye', 'रुपये', 'रुपया', 'টাকাটা',
    'rs', 'inr', 'tk', 'usd', 'dollar', 'dollars', 'bucks', 'cent', 'cents', 'paisa', 'poisa', 'পয়সা', 'पैसे',
    'kg', 'কেজি', 'किलो', 'gram', 'gm', 'g', 'গ্রাম', 'ग्राम', 'litre', 'ltr', 'km', 'meter', 'm',
    'piece', 'pieces', 'পিস', 'পিচ', 'পিসেস', 'পিছ', 'পিসটা', 'টা', 'টি', 'খানা', 'খানি',
    'product', 'প্রোডাক্ট', 'उत्पाद', 'rate', 'রেট', 'दर', 'price', 'দাম', 'কীমত', 'कीमत',
    'quantity', 'পরিমাণ', 'मात्रा', 'st', 'nd', 'rd', 'th',
    'approx', 'approximately', 'around', 'about', 'nearly', 'almost', 'roughly', 'total', 'worth', 'only',
    'pray', 'প্রায়', 'kachakachi', 'কাছাকাছি', 'moto', 'মতো', 'মতন', 'motamuti', 'মোটামুটি', 'mot', 'মোট',
    'lagbhag', 'लगभग', 'kareeb', 'karib', 'करीब', 'aaspas', 'आसपास', 'kul', 'कुल',
    'um', 'uh', 'er', 'ah', 'like', 'well', 'okay', 'ok', 'mane', 'মানে', 'matlab', 'मतलब', 'yaani', 'দাও', 'give',
    # Disjunctions / ranges (separate numbers, not additive)
    'বা', 'অথবা', 'থেকে', 'পর্যন্ত', 'या', 'अथवा', 'से', 'तक', 'or', 'to',
}

SHORT_STOPWORDS = {
    'to', 'in', 'by', 'on', 'at', 'is', 'it', 'an', 'as', 'or', 'if', 'we', 'he',
    'me', 'us', 'am', 'my', 'up', 'for', 'won', 'ate', 'too', 'na', 'je', 'ki', 'se', 'ka', 'ko',
    'বা', 'या', 'से', 'না',
}

# Normalize all Indic dictionary keys and stopwords to ensure both precomposed and decomposed nukta characters match
REPEATERS = {**REPEATERS, **{normalize_indic_chars(k): v for k, v in list(REPEATERS.items())}}
FRACTION_PREFIXES = {**FRACTION_PREFIXES, **{normalize_indic_chars(k): v for k, v in list(FRACTION_PREFIXES.items())}}
STOPWORDS = {normalize_indic_chars(w) for w in STOPWORDS} | set(STOPWORDS)
SHORT_STOPWORDS = {normalize_indic_chars(w) for w in SHORT_STOPWORDS} | set(SHORT_STOPWORDS)
CONNECTORS = {normalize_indic_chars(w) for w in CONNECTORS} | set(CONNECTORS)
DECIMAL_MARKERS = {normalize_indic_chars(w) for w in DECIMAL_MARKERS} | set(DECIMAL_MARKERS)
NEGATIVE_MARKERS = {normalize_indic_chars(w) for w in NEGATIVE_MARKERS} | set(NEGATIVE_MARKERS)


# ======================================================================
# 4. COMPILED LOOKUP MAPS & DUAL FUZZY MATCHER (Flutter/Dart-friendly)
# ======================================================================

EXACT_MAP: Dict[str, Tuple[Union[int, float], str]] = {}
UNIFIED_FUZZY_CORPUS: List[Tuple[str, Union[int, float], str]] = []
SHORT_WORD_CORPUS: List[Tuple[str, Union[int, float], str]] = []


def _register_vocab(vocab: Dict[Any, List[str]], kind: str, allow_fuzzy: bool = True):
    for val, variants in vocab.items():
        for variant in variants:
            v_clean = normalize_indic_chars(variant.lower().strip())
            if not v_clean:
                continue
            EXACT_MAP[v_clean] = (val, kind)

            if allow_fuzzy and v_clean not in STOPWORDS and v_clean not in SHORT_STOPWORDS:
                trans = transliterate_to_roman(v_clean)
                UNIFIED_FUZZY_CORPUS.append((trans, val, kind))
                if trans != v_clean:
                    UNIFIED_FUZZY_CORPUS.append((v_clean, val, kind))

                # If short word (lengths 2-4), add to dedicated short word matcher
                if 2 <= len(v_clean) <= 4 or (trans and 2 <= len(trans) <= 4):
                    SHORT_WORD_CORPUS.append((v_clean, val, kind))
                    if trans != v_clean:
                        SHORT_WORD_CORPUS.append((trans, val, kind))


_register_vocab(BENGALI_VOCAB, 'WORD')
_register_vocab(HINDI_VOCAB, 'WORD')
_register_vocab(ENGLISH_VOCAB, 'WORD')
_register_vocab(PHONETIC_ENGLISH_VOCAB, 'WORD', allow_fuzzy=False)
_register_vocab(MAGNITUDE_VOCAB, 'MAGNITUDE')
_register_vocab(FRACTION_MULTIPLIERS, 'FRACTION', allow_fuzzy=False)
_register_vocab(COMPOUND_HUNDREDS, 'COMPOUND', allow_fuzzy=False)

# Structural compound hundreds in native Bengali (১-৯৯ + 'শ' / 'শো')
for val, variants in BENGALI_VOCAB.items():
    if 1 <= val <= 99:
        for var in variants:
            var_norm = normalize_indic_chars(var.strip())
            if script_of(var_norm) == 'bengali' and var_norm not in ('তেরো', 'তের', 'তেড়ো', 'তেড়'):
                EXACT_MAP[var_norm + 'শ'] = (val * 100, 'COMPOUND')
                EXACT_MAP[var_norm + 'শো'] = (val * 100, 'COMPOUND')

# Structural compound hundreds in native Hindi (१-९९ + 'सौ')
for val, variants in HINDI_VOCAB.items():
    if 1 <= val <= 99:
        for var in variants:
            var_norm = normalize_indic_chars(var.strip())
            if script_of(var_norm) == 'devanagari':
                EXACT_MAP[var_norm + 'सौ'] = (val * 100, 'COMPOUND')
                EXACT_MAP[var_norm + ' सौ'] = (val * 100, 'COMPOUND')

# 'k' attached or detached to digits: 50k -> 50000
EXACT_MAP['k'] = (1000, 'MAGNITUDE')


def edit_distance(a: str, b: str) -> int:
    """Standard dynamic programming Levenshtein distance."""
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


# ----------------------------------------------------------------------
# DEDICATED SHORT-WORD FUZZY MATCHER (Lengths 2-4)
# ----------------------------------------------------------------------
def resolve_fuzzy_short(token: str) -> Optional[Tuple[Union[int, float], str]]:
    """
    Dedicated fuzzy matcher for short words (lengths 2-4).
    Since short number words in small scale (1-10) are phonetically distinct,
    an edit distance of 1 can match typos reliably without false positives.
    Stopwords ('to', 'in', 'by', etc.) are strictly guarded.
    """
    q = token.lower()
    q_len = len(q)
    if q_len < 2 or q_len > 4:
        return None
    if q in SHORT_STOPWORDS or q in STOPWORDS or q in ('tero', 'teroso'):
        return None

    q_trans = transliterate_to_roman(q)
    best_distance_per_entry: Dict[Union[int, float], Tuple[int, str]] = {}

    for cand_word, cand_val, cand_kind in SHORT_WORD_CORPUS:
        if abs(len(cand_word) - q_len) > 1 and abs(len(cand_word) - len(q_trans)) > 1:
            continue
        d1 = edit_distance(q, cand_word)
        d2 = edit_distance(q_trans, cand_word)
        d = min(d1, d2)
        if d <= 1:
            if cand_val not in best_distance_per_entry or d < best_distance_per_entry[cand_val][0]:
                best_distance_per_entry[cand_val] = (d, cand_kind)

    if not best_distance_per_entry:
        return None

    ranked = sorted(best_distance_per_entry.items(), key=lambda kv: kv[1][0])
    best_value, (best_dist, best_kind) = ranked[0]
    if len(ranked) == 1:
        return best_value, best_kind

    _, (second_dist, _) = ranked[1]
    if second_dist - best_dist >= 1:
        return best_value, best_kind

    return None


# ----------------------------------------------------------------------
# GENERAL UNIFIED FUZZY MATCHER (Lengths >= 5)
# ----------------------------------------------------------------------
_MAX_DIST_BY_LEN = {5: 1, 6: 2, 7: 2, 8: 2}


def _max_dist(length: int) -> int:
    if length in _MAX_DIST_BY_LEN:
        return _MAX_DIST_BY_LEN[length]
    return 1 if length < 5 else 3


def resolve_fuzzy(token: str) -> Optional[Tuple[Union[int, float], str]]:
    """
    Unified, single-pass fuzzy search over the entire vocabulary corpus.
    Transliterates native script inputs into phonetic roman space.
    Uses length-difference pre-filtering to skip non-matching candidates instantly.
    Only returns a result if the top-scoring candidate has a clear lead (margin >= 1).
    """
    q = token.lower()
    q_len = len(q)
    if q_len < 5:
        return resolve_fuzzy_short(token)

    # Guard: if token ends in "so", "sho", "sau" but failed exact match (e.g. "teroso"),
    # do NOT fuzzy guess it against 1800 or 13.
    for suffix in ('so', 'sho', 'sau'):
        if q.endswith(suffix) and q_len > len(suffix) + 1:
            return None

    q_trans = transliterate_to_roman(q)
    limit = _max_dist(q_len)
    best_distance_per_entry: Dict[Union[int, float], Tuple[int, str]] = {}

    for cand_word, cand_val, cand_kind in UNIFIED_FUZZY_CORPUS:
        if abs(len(cand_word) - q_len) > limit and abs(len(cand_word) - len(q_trans)) > limit:
            continue
        d1 = edit_distance(q, cand_word)
        d2 = edit_distance(q_trans, cand_word)
        d = min(d1, d2)
        if d <= limit:
            if cand_val not in best_distance_per_entry or d < best_distance_per_entry[cand_val][0]:
                best_distance_per_entry[cand_val] = (d, cand_kind)

    if not best_distance_per_entry:
        return None

    ranked = sorted(best_distance_per_entry.items(), key=lambda kv: kv[1][0])
    best_value, (best_dist, best_kind) = ranked[0]
    if len(ranked) == 1:
        return best_value, best_kind

    _, (second_dist, _) = ranked[1]
    if second_dist - best_dist >= 1:
        return best_value, best_kind

    return None


# ======================================================================
# 5. TOKENIZATION & CLASSIFICATION
# ======================================================================

@dataclass
class Token:
    raw: str
    kind: str  # DIGIT | DECIMAL | WORD | MAGNITUDE | FRACTION | COMPOUND | CONNECTOR | DECIMAL_MARK | NEGATIVE | OTHER
    value: Optional[Union[int, float, Decimal]] = None
    matched_as: Optional[str] = None
    ambiguous_source: bool = False


TOKEN_RE = re.compile(
    r'\d+\.\d+'
    r'|[a-zA-Z\u0900-\u09FF]+\d+\w*'
    r'|\d+'
    r'|[\u0980-\u09FF\u0900-\u097Fa-zA-Z]+'
    r'|[^\s]+'
)


def _is_alphanumeric_code(raw: str) -> bool:
    has_letters = bool(re.search(r'[a-zA-Z\u0900-\u09FF]', raw))
    has_digits = bool(re.search(r'\d', raw))
    if has_letters and has_digits:
        if re.search(r'^[a-zA-Z\u0900-\u09FF]+\d', raw) or re.search(r'\d+[a-zA-Z\u0900-\u09FF]+\d', raw):
            return True
    return False


def try_compound_hundred(token: str) -> Optional[int]:
    """
    Check if token is <number_word> + <hundred_suffix>.
    Prefix must EXACTLY match a known number (1 to 99) in EXACT_MAP.
    No fuzzy guessing on prefix: safely accepts 'tinso' -> 300,
    'baroso' -> 1200, 'তিনশো' -> 300, 'तीनसौ' -> 300, while rejecting 'teroso'.
    """
    lower = token.lower()
    for suffix in ('sho', 'sau', 'so'):
        if lower.endswith(suffix) and len(lower) > len(suffix):
            prefix = lower[:-len(suffix)]
            if prefix in ('tero',):  # 'tero' is guarded as ambiguous per spec
                continue
            if prefix in EXACT_MAP:
                val, kind = EXACT_MAP[prefix]
                if kind in ('WORD', 'FRACTION'):
                    return int(val * 100)

            # Support collapsed sibilants when root number word ends in 'sh' or 's':
            # e.g. "ekusho" -> prefix "eku" + "sh" = "ekush" (21) -> 2100
            # "atasho" -> prefix "ata" + "sh" = "atash" (28) -> 2800
            # "athasho" -> prefix "atha" + "sh" = "athash" (28) -> 2800
            if suffix == 'sho':
                for s_suf in ('sh', 's'):
                    cand = prefix + s_suf
                    if cand in EXACT_MAP:
                        val, kind = EXACT_MAP[cand]
                        if kind in ('WORD', 'FRACTION'):
                            return int(val * 100)
            elif suffix == 'so':
                for s_suf in ('s', 'sh'):
                    cand = prefix + s_suf
                    if cand in EXACT_MAP:
                        val, kind = EXACT_MAP[cand]
                        if kind in ('WORD', 'FRACTION'):
                            return int(val * 100)

    # Native script
    for suffix in ('শো', 'শ', 'সৌ'):
        if token.endswith(suffix) and len(token) > len(suffix):
            prefix = token[:-len(suffix)]
            prefix_norm = normalize_indic_chars(prefix)
            if prefix_norm in EXACT_MAP:
                val, kind = EXACT_MAP[prefix_norm]
                if kind in ('WORD', 'FRACTION'):
                    return int(val * 100)

    return None


def classify(raw: str) -> Token:
    if re.fullmatch(r'\d+\.\d+', raw):
        return Token(raw, 'DECIMAL', Decimal(raw))

    if raw.isdigit():
        return Token(raw, 'DIGIT', int(raw))

    if _is_alphanumeric_code(raw):
        return Token(raw, 'OTHER', ambiguous_source=True)

    raw_norm = normalize_indic_chars(raw.lower().strip())
    lower = raw_norm

    if raw_norm in REPEATERS:
        return Token(raw=raw, kind='REPEATER', value=REPEATERS[raw_norm], matched_as='REPEATER')
    if raw_norm in FRACTION_PREFIXES:
        p_type, p_delta = FRACTION_PREFIXES[raw_norm]
        return Token(raw=raw, kind='FRACTION_PREFIX', value=p_delta, matched_as=p_type)

    if raw in CONNECTORS or lower in CONNECTORS:
        return Token(raw, 'CONNECTOR')
    if raw in DECIMAL_MARKERS or lower in DECIMAL_MARKERS:
        return Token(raw, 'DECIMAL_MARK')
    if raw in NEGATIVE_MARKERS or lower in NEGATIVE_MARKERS:
        return Token(raw, 'NEGATIVE')

    if raw in STOPWORDS or lower in STOPWORDS:
        return Token(raw, 'OTHER')

    # Exact dictionary lookup (handles native script and roman)
    if raw in EXACT_MAP:
        val, kind = EXACT_MAP[raw]
        return Token(raw, kind, val, matched_as=raw)
    if lower in EXACT_MAP:
        val, kind = EXACT_MAP[lower]
        return Token(raw, kind, val, matched_as=lower)

    # Structural compound hundreds (e.g. tinso, charso, pachso, baroso, তিনশো, পাঁচশ, three सौ, etc.)
    compound = try_compound_hundred(raw)
    if compound is not None:
        return Token(raw, 'COMPOUND', compound, matched_as=f'compound:{raw}')

    # Bengali classifier suffix on digits (e.g. "5ta" or "১০টা" if not pre-separated)
    m_classifier = re.match(r'^(\d+)(?:ta|to|ti|te|টা|টো|টি|টে)$', lower)
    if m_classifier:
        return Token(raw, 'DIGIT', int(m_classifier.group(1)))

    # Ordinal numbers: "1st", "2nd", "3rd", "4th"
    m_ordinal = re.match(r'^(\d+)(?:st|nd|rd|th)$', lower)
    if m_ordinal:
        return Token(raw, 'DIGIT', int(m_ordinal.group(1)))

    # Transliterated exact check
    trans = transliterate_to_roman(raw)
    if trans in EXACT_MAP:
        val, kind = EXACT_MAP[trans]
        return Token(raw, kind, val, matched_as=f'trans:{trans}')

    # Bengali grammatical case inflections (-এর, -তে, -কে, -র)
    for suf in ('এর', 'তে', 'কে', 'র'):
        if raw_norm.endswith(suf) and len(raw_norm) > len(suf):
            stem = raw_norm[:-len(suf)]
            stem_tok = classify(stem)
            if stem_tok.kind != 'OTHER' and stem_tok.value is not None:
                return Token(raw=raw, kind=stem_tok.kind, value=stem_tok.value, matched_as=f'{stem_tok.matched_as}+{suf}')

    # Fuzzy search (handles short words and long words with transliteration)
    fuzzy_res = resolve_fuzzy(raw)
    if fuzzy_res is not None:
        val, kind = fuzzy_res
        return Token(raw, kind, val, matched_as=f'~{raw}')

    return Token(raw, 'OTHER')


def _expand_repeaters(tokens: List[Token]) -> List[Token]:
    expanded: List[Token] = []
    i = 0
    n = len(tokens)
    while i < n:
        t = tokens[i]
        if t.kind == 'REPEATER' and i + 1 < n:
            count = int(t.value)
            next_t = tokens[i + 1]
            if next_t.kind in ('WORD', 'DIGIT') and next_t.value is not None:
                for _ in range(count):
                    expanded.append(Token(raw=next_t.raw, kind=next_t.kind, value=next_t.value, matched_as=next_t.matched_as))
                i += 2
                continue
        expanded.append(t)
        i += 1
    return expanded


def _fold_fraction_prefixes(tokens: List[Token]) -> List[Token]:
    folded: List[Token] = []
    i = 0
    n = len(tokens)
    while i < n:
        t = tokens[i]
        if t.kind == 'FRACTION_PREFIX' and i + 1 < n:
            delta = float(t.value)
            next_t = tokens[i + 1]
            if next_t.kind == 'COMPOUND' and next_t.value is not None:
                base_n = next_t.value / 100
                new_val = (base_n + delta) * 100
                val_to_use = int(new_val) if new_val == int(new_val) else new_val
                folded.append(Token(raw=f"{t.raw} {next_t.raw}", kind='COMPOUND', value=val_to_use, matched_as=f"{t.matched_as} {next_t.matched_as}"))
                i += 2
                continue
            elif next_t.kind == 'MAGNITUDE' and next_t.value is not None:
                new_val = (1.0 + delta) * next_t.value
                val_to_use = int(new_val) if new_val == int(new_val) else new_val
                folded.append(Token(raw=f"{t.raw} {next_t.raw}", kind='COMPOUND', value=val_to_use, matched_as=f"{t.matched_as} {next_t.matched_as}"))
                i += 2
                continue
            elif next_t.kind in ('WORD', 'DIGIT') and next_t.value is not None:
                new_val = next_t.value + delta
                val_to_use = int(new_val) if new_val == int(new_val) else new_val
                folded.append(Token(raw=f"{t.raw} {next_t.raw}", kind='FRACTION', value=val_to_use, matched_as=f"{t.matched_as} {next_t.matched_as}"))
                i += 2
                continue
        folded.append(t)
        i += 1
    return folded


def tokenize(text: str) -> List[Token]:
    cleaned = clean_and_normalize_text(text)
    raw_tokens = TOKEN_RE.findall(cleaned)
    tokens = []
    for t in raw_tokens:
        tok = classify(t)
        tokens.append(tok)
    tokens = _expand_repeaters(tokens)
    tokens = _fold_fraction_prefixes(tokens)
    return tokens


# ======================================================================
# 6. HIERARCHICAL GRAMMAR & ACCUMULATION
# ======================================================================

class LeadingZeroNum(str):
    """
    String representation for numbers with significant leading zeros (e.g. '088', '007').
    Inherits from str so it preserves formatting and serializes naturally,
    while comparing equal to both '088' and int 88.
    """
    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, int):
            try:
                return int(str(self)) == other
            except ValueError:
                return False
        return super().__eq__(other)

    def __hash__(self) -> int:
        return super().__hash__()

    def __int__(self) -> int:
        return int(str(self))


@dataclass
class NumberResult:
    text: str
    value: Optional[Union[int, Decimal, str]]
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
    prev_token = None
    fraction_multiplier: Optional[float] = None
    digit_str: Optional[str] = None

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

        if tok.kind == 'FRACTION':
            seen_any = True
            fraction_multiplier = tok.value
            digit_str = None
            prev_kind = tok.kind
            i += 1
            continue

        if tok.kind == 'DECIMAL':
            seen_any = True
            digit_str = None
            if i + 1 < n and tokens[i + 1].kind == 'MAGNITUDE' and tokens[i + 1].value is not None:
                mag_val = tokens[i + 1].value
                dec_val = tok.value * mag_val
                val_to_add = int(dec_val) if dec_val == int(dec_val) else dec_val
                result += val_to_add
                last_flushed_magnitude = mag_val
                i += 2
                prev_kind = 'MAGNITUDE'
                continue
            else:
                return NumberResult(
                    text=' '.join(t.raw for t in tokens),
                    value=-tok.value if negative else tok.value,
                    ambiguous=False,
                    tokens=tokens,
                )

        if tok.kind == 'DIGIT':
            seen_any = True
            if prev_kind == 'DIGIT':
                ambiguous_reason = f"adjacent digit runs '{tokens[i-1].raw}' and '{tok.raw}' with no connecting word"
                break
            if last_flushed_magnitude is not None and tok.value >= last_flushed_magnitude:
                ambiguous_reason = f"digit run '{tok.raw}' ({tok.value}) is at or above previously resolved magnitude ({last_flushed_magnitude})"
                break
            val = tok.value
            if current == 0:
                current = val
                digit_str = tok.raw if tok.raw.isdigit() else None
            elif current % 100 == 0 and val < 100:
                current += val
                digit_str = None
            elif current % 10 == 0 and current >= 20 and 1 <= val <= 9:
                current += val
                digit_str = None
            elif digit_str is not None and tok.raw.isdigit():
                digit_str += tok.raw
                current = int(digit_str)
            else:
                current = int(f"{current}{val}")
                digit_str = None
            prev_kind = tok.kind
            prev_token = tok
            i += 1
            continue

        if tok.kind == 'COMPOUND':
            seen_any = True
            result += current
            result += tok.value
            current = 0
            digit_str = None
            prev_kind = tok.kind
            prev_token = tok
            i += 1
            continue

        if tok.kind == 'MAGNITUDE' and tok.value is not None:
            seen_any = True
            val = tok.value
            digit_str = None

            if prev_kind == 'MAGNITUDE' and prev_token and prev_token.value == val:
                ambiguous_reason = f"duplicate magnitude '{tok.raw}' ({val})"
                break

            if fraction_multiplier is not None:
                composed = int(fraction_multiplier * val)
                result += composed
                fraction_multiplier = None
                last_flushed_magnitude = val
                prev_kind = tok.kind
                prev_token = tok
                i += 1
                continue

            if val == 100:
                current = (current or 1) * 100
            else:
                if last_flushed_magnitude is not None and val >= last_flushed_magnitude:
                    if last_flushed_magnitude == 100 and val == 10_000_000:
                        current = (current or 1) * val
                        result += current
                        current = 0
                        last_flushed_magnitude = val
                        prev_kind = tok.kind
                        prev_token = tok
                        i += 1
                        continue
                    ambiguous_reason = f"magnitude '{tok.raw}' ({val}) is not smaller than previous magnitude ({last_flushed_magnitude})"
                    break
                current = (current or 1) * val
                result += current
                current = 0
                last_flushed_magnitude = val

            prev_kind = tok.kind
            prev_token = tok
            i += 1
            continue

        if tok.kind == 'WORD' and tok.value is not None:
            seen_any = True
            val = tok.value

            is_single_digit = (0 <= val <= 9)
            prev_is_single_digit = (prev_token and prev_token.kind in ('WORD', 'DIGIT') and
                                    isinstance(prev_token.value, int) and 0 <= prev_token.value <= 9)

            if prev_is_single_digit and is_single_digit:
                digit_str = (digit_str or str(prev_token.value)) + str(val)
                current = int(digit_str)
            elif is_single_digit and prev_token is None:
                digit_str = str(val)
                current = val
            elif current == 0:
                current = val
                digit_str = str(val) if is_single_digit else None
            elif current % 100 == 0 and val < 100:
                current += val
                digit_str = None
            elif current % 10 == 0 and current >= 20 and 1 <= val <= 9:
                current += val
                digit_str = None
            else:
                current = int(f"{current}{val}")
                digit_str = None

            prev_kind = tok.kind
            prev_token = tok
            i += 1
            continue

        break

    text = ' '.join(t.raw for t in tokens[:i]) if i < n else ' '.join(t.raw for t in tokens)

    if ambiguous_reason:
        return NumberResult(text=text, value=None, ambiguous=True, reason=ambiguous_reason, tokens=tokens[:i])

    if not seen_any:
        return NumberResult(text=text, value=None, ambiguous=False, reason='no number found', tokens=tokens)

    if result == 0 and digit_str is not None and digit_str.startswith('0') and len(digit_str) > 1:
        final = LeadingZeroNum(digit_str)
    else:
        final = result + current
        if negative:
            final = -final
    return NumberResult(text=text, value=final, ambiguous=False, tokens=tokens[:i])


def _split_decimal(tokens: List[Token]) -> Tuple[List[Token], Optional[List[Token]]]:
    for idx, t in enumerate(tokens):
        if t.kind == 'DECIMAL_MARK':
            return tokens[:idx], tokens[idx + 1:]
    return tokens, None


def parse_phrase(tokens: Union[List[Token], str]) -> NumberResult:
    if isinstance(tokens, str):
        tokens = tokenize(tokens)
    int_part_tokens, frac_tokens = _split_decimal(tokens)
    int_result = parse_tokens(int_part_tokens)

    if frac_tokens is None:
        return int_result
    if int_result.ambiguous:
        return int_result

    mag_tokens = []
    pure_frac_tokens = []
    for t in frac_tokens:
        if t.kind == 'MAGNITUDE':
            mag_tokens.append(t)
        else:
            pure_frac_tokens.append(t)

    frac_digits = []
    for t in pure_frac_tokens:
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
        return NumberResult(
            text=' '.join(x.raw for x in tokens), value=None, ambiguous=True,
            reason='decimal marker with no fractional digits', tokens=tokens,
        )

    int_val = int_result.value if int_result.value is not None else 0
    dec = Decimal(f"{int_val}.{''.join(frac_digits)}")

    if mag_tokens:
        mag_result = parse_tokens(mag_tokens)
        if mag_result.value is not None:
            total = dec * Decimal(mag_result.value)
            val = int(total) if total == int(total) else total
            return NumberResult(text=' '.join(x.raw for x in tokens), value=val, ambiguous=False, tokens=tokens)

    return NumberResult(text=' '.join(x.raw for x in tokens), value=dec, ambiguous=False, tokens=tokens)


# ======================================================================
# 7. NUMBER-PHRASE BOUNDARY EXTRACTION
# ======================================================================

def extract_numbers(text: str) -> List[NumberResult]:
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
            flush()
        group.append(tok)
        prev_was_digit = (tok.kind == 'DIGIT')

    flush()
    return results


def text_to_number(text: str) -> Optional[Union[int, Decimal, str]]:
    results = extract_numbers(text)
    if not results:
        return None
    return results[0].value
