# word2num: High-Performance Multilingual Speech-to-Number Parser

A regional-first, deterministic, zero-dependency number parser specifically designed to handle real-world **ASR (Automatic Speech Recognition)** and **NER (Named Entity Recognition)** outputs.

Supports **Bengali (বাংলা)**, **Hindi (हिन्दी)**, **English**, **Phonetic Romanized Transliterations (Banglish / Hinglish)**, and **Script-Transcribed English Phonetics** (e.g., `ওয়ান`, `টু`, `वन`, `टू`).

---

## Table of Contents
1. [Why This Library Exists](#why-this-library-exists)
2. [Core Design Philosophy & Flutter / Dart Portability](#core-design-philosophy--flutter--dart-portability)
3. [Architecture & Pipeline Overview](#architecture--pipeline-overview)
4. [Internal Mechanics: How Each Function Works](#internal-mechanics-how-each-function-works)
   - [1. Text Cleaning & Unicode Normalization](#1-text-cleaning--unicode-normalization)
   - [2. Akshara-to-Roman Phonetic Transliteration](#2-akshara-to-roman-phonetic-transliteration)
   - [3. Dual Fuzzy Matchers (Short vs. Long Words)](#3-dual-fuzzy-matchers-short-vs-long-words)
   - [4. Structural Compound Hundreds Engine](#4-structural-compound-hundreds-engine)
   - [5. Tokenization & Grammatical Case Inflection Stripping](#5-tokenization--grammatical-case-inflection-stripping)
   - [6. Repeater Expansion Engine](#6-repeater-expansion-engine)
   - [7. Fraction Prefix Folding Engine](#7-fraction-prefix-folding-engine)
   - [8. Hierarchical Grammar & Serial Digit Accumulator](#8-hierarchical-grammar--serial-digit-accumulator)
   - [9. Leading-Zero Preservation (`LeadingZeroNum`)](#9-leading-zero-preservation-leadingzeronum)
   - [10. Decimal & Magnitude Resolution](#10-decimal--magnitude-resolution)
   - [11. Multi-Number Sentence Extraction](#11-multi-number-sentence-extraction)
5. [Accuracy & Ambiguity Guardrails](#accuracy--ambiguity-guardrails)
6. [How to Expand the Codebase (Add Words, Dialects & Languages)](#how-to-expand-the-codebase-add-words-dialects--languages)
7. [Running the Test Suite](#running-the-test-suite)
8. [License](#license)

---

## Why This Library Exists

Most number-parsing libraries assume neat, written English text (like `"one hundred and twenty"`). But in real-world conversational AI, speech assistants, and voice commerce, inputs come from speech recognizers (ASR) followed by Named Entity Recognition (NER) models:

1. **ASR Noise & Mixed Scripts**: Voice inputs often transcribe as a chaotic mix of native scripts, Romanized words, and raw digits:
   - `"সাড়ে সতেরোশ"` $\rightarrow$ `1750`
   - `"ট্রিপল নাইন 52"` $\rightarrow$ `99952`
   - `"জিরো ডাবল এইট"` $\rightarrow$ `088` (retaining the leading `0`)
   - `"five hundred 50"` $\rightarrow$ `550`
2. **Heavy Regional Dialects & Colloquialisms**:
   - Compound hundreds in Bengali and Hindi (`বারোশো` = 1200, `tinso` = 300, `আড়াইশো` = 250, `সাড়ে তিনশ` = 350, `পৌনে দুশো` = 175).
   - Grammatical case inflections attached to numbers: `"আড়াইশোর"` (price of 250), `"হাজারের"` (of thousand), `"পাঁচশোতে"` (in 500).
3. **Serial Digit Sequences vs. Math Addition**:
   - Spoken phone numbers, OTPs, roll numbers, or license plates: `"ওয়ান সেভেন"` is `17` (not $1+7=8$).
   - Spoken years: `"twenty twenty"` is `2020` (not $20+20=40$).
4. **False Positive Traps**:
   - Alphanumeric codes (`covid19`, `mp3`, `h2o`, `f16`) must **never** be mangled into numbers.
   - Non-words and typos like `teroso` must be safely rejected instead of hallucinating `1800` or `1300`.

---

## Core Design Philosophy & Flutter / Dart Portability

This entire library is built with **zero external dependencies**:
- **No machine learning models** (no PyTorch, TensorFlow, HuggingFace, or ONNX).
- **No C-extensions** or complex native binaries.
- **No thread pools or concurrency primitives**.
- **Pure procedural logic**: Uses only standard Hash Maps (`dict`), Sets (`set`), Dynamic Arrays (`list`), regular expressions (`re`), and dynamic programming.

### Why?
Because this codebase is architected for **1:1 Flutter / Dart portability**:
Every single function in `word2num.py` has an exact equivalent in Dart:
- Python `dict` $\rightarrow$ Dart `Map`
- Python `list` $\rightarrow$ Dart `List`
- Python `set` $\rightarrow$ Dart `Set`
- Python `re` $\rightarrow$ Dart `RegExp`
- Python `Token` dataclass $\rightarrow$ Dart `class Token`

Anyone can read this code, reason about it, and port it to Dart or any other language in an afternoon.

---

## Architecture & Pipeline Overview

When text enters `text_to_number(text)` or `extract_numbers(text)`, it passes through a deterministic multi-stage pipeline:

```
                  Raw Input Text
                        │
                        ▼
   ┌─────────────────────────────────────────┐
   │ 1. Clean & Normalize Text               │
   │    - Indic digits to ASCII              │
   │    - Nukta canonicalization (ড়, ঢ়, য়)   │
   │    - Strip currency symbols & noise      │
   │    - Separate attached units (10lakh)   │
   └────────────────────┬────────────────────┘
                        │
                        ▼
   ┌─────────────────────────────────────────┐
   │ 2. Tokenize & Classify Tokens           │
   │    - Alphanumeric code trap check       │
   │    - Exact dictionary lookup            │
   │    - Grammatical inflection stripping   │
   │    - Compound hundred analyzer          │
   │    - Dual fuzzy matcher (short vs long) │
   └────────────────────┬────────────────────┘
                        │
                        ▼
   ┌─────────────────────────────────────────┐
   │ 3. Repeater Expansion                   │
   │    - 'ডাবল এইট'  --> 'এইট', 'এইট'       │
   │    - 'ট্রিপল ৪'  --> '৪', '৪', '৪'      │
   └────────────────────┬────────────────────┘
                        │
                        ▼
   ┌─────────────────────────────────────────┐
   │ 4. Fraction Prefix Folding              │
   │    - 'সাড়ে' (+0.5) + 'তিনশ' --> 350     │
   │    - 'পৌনে' (-0.25) + 'দুশো' --> 175    │
   └────────────────────┬────────────────────┘
                        │
                        ▼
   ┌─────────────────────────────────────────┐
   │ 5. Hierarchical Grammar & Accumulator   │
   │    - Remainder addition (500 + 50 = 550)│
   │    - Scale multiplication (5 * 1000)    │
   │    - Serial concatenation (999 + 52)    │
   │    - Spoken year concatenation (20 20)  │
   │    - Preserves leading zeroes ('088')   │
   └────────────────────┬────────────────────┘
                        │
                        ▼
                 Resolved Result
          (int / Decimal / LeadingZeroNum)
```

---

## Internal Mechanics: How Each Function Works

### 1. Text Cleaning & Unicode Normalization

#### `normalize_digits(text: str) -> str`
Converts non-ASCII Unicode digits into standard ASCII `0`–`9`:
- Bengali digits (`০১২৩৪৫৬৭৮৯` $\rightarrow$ `0123456789`)
- Devanagari digits (`०१२३४५६७८९` $\rightarrow$ `0123456789`)
- Arabic-Indic digits (`٠١٢٣٤٥٦٧৮৯` $\rightarrow$ `0123456789`)

#### `normalize_indic_chars(text: str) -> str`
In Indic Unicode, letters with dots (nuktas) can be stored in two ways:
1. Decomposed: Base consonant + Nukta mark (`ড` U+09A1 + `়` U+09BC)
2. Precomposed: Single composite character (`ড়` U+09DC)

If a user types decomposed characters or ASR produces them, exact lookups fail. `normalize_indic_chars` canonicalizes all decomposed nukta sequences into single precomposed codepoints across Bengali and Devanagari (`ড়`, `ঢ়`, `য়`, `ड़`, `ढ़`, `फ़`, `य़`).

#### `clean_and_normalize_text(text: str) -> str`
Strips speech and formatting noise:
- Strips currency symbols (`₹`, `$`, `€`, `£`, `৳`, `¥`) and abbreviations (`Rs. 500` $\rightarrow$ `500`).
- Strips invoice slash-hyphens (`500/-` $\rightarrow$ `500`).
- Removes internal thousands commas (`1,25,000` $\rightarrow$ `125000`).
- Splits attached units from digits (`10lakh` $\rightarrow$ `10 lakh`, `50kg` $\rightarrow$ `50 kg`, `১০টা` $\rightarrow$ `10 টা`).
- Removes stray quotes, brackets, and punctuation while carefully protecting internal decimals (`500.50` remains intact).

---

### 2. Akshara-to-Roman Phonetic Transliteration

#### `transliterate_to_roman(text: str) -> str`
Instead of having to manually register thousands of spelling variants for every native Bengali and Hindi word, `transliterate_to_roman()` maps Indic characters into a standardized phonetic Roman alphabet:
- Bengali: `এক` $\rightarrow$ `ek`, `দুই` $\rightarrow$ `dui`, `আটাশ` $\rightarrow$ `atash`
- Hindi: `एक` $\rightarrow$ `ek`, `दो` $\rightarrow$ `do`, `अट्ठाईस` $\rightarrow$ `atthais`

This creates a **shared phonetic space** where native script inputs and Romanized speech inputs (`"ekus"`, `"ekush"`, `"একুশ"`) converge effortlessly.

---

### 3. Dual Fuzzy Matchers (Short vs. Long Words)

A major breakthrough in this codebase is the **Dual Fuzzy Architecture**:

#### Why can't one fuzzy matcher handle everything?
- In short words (2–4 letters), words are densely packed: `to`, `in`, `no`, `two`, `ten`. An unconstrained edit distance of 1 or 2 causes massive false positives (e.g., matching English `"to"` to `"two"` or `"do"`).
- In long words (5+ letters), words are phonetically sparse: `pochattor` (75), `achashsho` (2800). Here, edit distances of 1 or 2 are safe and necessary to catch ASR typos.

#### `resolve_fuzzy_short(token: str)` (Lengths 2–4)
- Only runs if word length is between 2 and 4.
- Strictly guarded by `SHORT_STOPWORDS` (words like `to`, `in`, `by`, `on`, `or`, `se`, `ba` can **never** fuzzy-match a number).
- Maximum edit distance allowed: **1**.
- Requires a strict confidence lead (the best match must beat the second-best match by $\ge 1$).

#### `resolve_fuzzy(token: str)` (Lengths $\ge$ 5)
- Dynamic edit distance thresholds based on token length:
  - Length 5: edit distance $\le 1$
  - Length 6–8: edit distance $\le 2$
  - Length 9+: edit distance $\le 3$
- Length-difference pre-filtering: candidates whose lengths differ from the query by more than the threshold are skipped instantly in $O(1)$.
- Suffix guard: tokens ending in `-so`, `-sho`, `-sau` that failed exact match (like `teroso`) are guarded and rejected to avoid confusing them with 1800 or 13.

---

### 4. Structural Compound Hundreds Engine

#### `try_compound_hundred(token: str) -> Optional[int]`
In South Asian languages, hundreds from 100 to 9900 are spoken as compound single words:
- Bengali: `বারোশো` (1200), `তেরোশো` (1300), `সতেরোশ` (1700), `পঁচিশশো` (2500)
- Hindi: `बारह सौ` (1200), `पच्चीस सौ` (2500)
- Romanized: `baroso` (1200), `tinso` (300), `athasso` (2800)

`try_compound_hundred` decomposes the word into:
$$\text{prefix} + \text{hundred\_suffix}$$
1. Checks suffixes: `-sho`, `-so`, `-sau`, `-শো`, `-শ`, `-সৌ`.
2. **Crucial Safety Rule**: The extracted prefix must **strictly match an exact known number** (1 to 99) in `EXACT_MAP`.
   - `tin` (3) + `so` $\rightarrow 3 \times 100 = 300$ (Accepted).
   - `baro` (12) + `sho` $\rightarrow 12 \times 100 = 1200$ (Accepted).
   - `tero` + `so` $\rightarrow$ Rejected (prevents false matches).
3. **Collapsed Sibilant Handling**: In Romanized speech, double `sh`/`s` often collapse:
   - `athasho` $\rightarrow$ stem `athash` (28) $\rightarrow 2800$.
   - `ekusho` $\rightarrow$ stem `ekush` (21) $\rightarrow 2100$.

---

### 5. Tokenization & Grammatical Case Inflection Stripping

#### `classify(raw: str) -> Token`
Assigns each token a `kind` and resolved `value`:
- `DIGIT`: Raw digits (`"52"`, `"100"`).
- `DECIMAL`: Floating point (`"500.50"`).
- `WORD`: Base number words (`"পাঁচ"`, `"five"`).
- `MAGNITUDE`: Multipliers (`"হাজার"`, `"hundred"`, `"thousand"`, `"lakh"`, `"crore"`).
- `COMPOUND`: Pre-composed hundreds (`"বারোশো"` $\rightarrow 1200$, `"আড়াইশো"` $\rightarrow 250$).
- `FRACTION_PREFIX`: Fractional modifiers (`"সাড়ে"` $\rightarrow +0.5$, `"পৌনে"` $\rightarrow -0.25$).
- `REPEATER`: Multipliers (`"ডাবল"` $\rightarrow 2$, `"ট্রিপল"` $\rightarrow 3$).
- `CONNECTOR`: Joining words (`"and"`, `"plus"`, `"এবং"`, `"aur"`).
- `DECIMAL_MARK`: Decimal markers (`"point"`, `"dot"`, `"দশমিক"`).
- `NEGATIVE`: Minus markers (`"minus"`, `"মাইনাস"`).
- `OTHER`: Non-number words (`"potatoes"`, `"apple"`, `"give"`).

#### Morphological Inflection Stripping (Declension)
In Bengali speech, nouns and numbers have grammatical case endings:
- Genitive `-এর` / `-র`: `"হাজারের"` (of 1000) $\rightarrow$ stem `"হাজার"` (1000); `"আড়াইশোর"` (of 250) $\rightarrow$ stem `"আড়াইশো"` (250).
- Locative `-তে`: `"পাঁচশোতে"` (in 500) $\rightarrow$ stem `"পাঁচশো"` (500).
- Accusative `-কে`: `"একশোকে"` (to 100) $\rightarrow$ stem `"একশো"` (100).

`classify()` strips these suffixes dynamically, re-classifying the bare stem. If the stem resolves to a number, the token is recognized without needing every grammatical permutation in the dictionary.

---

### 6. Repeater Expansion Engine

#### `_expand_repeaters(tokens: List[Token]) -> List[Token]`
In spoken phone numbers and codes, speakers frequently use repeaters:
- `"ডাবল এইট"` (double eight) $\rightarrow$ `Token(8)`, `Token(8)`
- `"ট্রিপল ফোর"` (triple four) $\rightarrow$ `Token(4)`, `Token(4)`, `Token(4)`
- `"ডাবল জিরো"` (double zero) $\rightarrow$ `Token(0)`, `Token(0)`

Runs during tokenization before grammar parsing, replacing the repeater and the target token with $N$ duplicate copies of the target token.

---

### 7. Fraction Prefix Folding Engine

#### `_fold_fraction_prefixes(tokens: List[Token]) -> List[Token]`
Handles South Asian fractional prefixes:
- **`সাড়ে` / `साढ़े` (Sade)**: $+0.5$
- **`পৌনে` / `पौने` (Paune)**: $-0.25$
- **`সওয়া` / `सवा` (Sawa)**: $+0.25$

Folds mathematically into the following token:
1. **On Compound Hundreds ($N \times 100$)**:
   - `"সাড়ে সতেরোশ"` $\rightarrow (17 + 0.5) \times 100 = 1750$
   - `"সাড়ে তিনশ"` $\rightarrow (3 + 0.5) \times 100 = 350$
   - `"পৌনে দুশো"` $\rightarrow (2 - 0.25) \times 100 = 175$
   - `"সোয়া দুশো"` $\rightarrow (2 + 0.25) \times 100 = 225$
2. **On Magnitudes ($M$)**:
   - `"সাড়ে পাঁচ হাজার"` $\rightarrow (5 + 0.5) \times 1000 = 5500$
   - `"পৌনে পাঁচ হাজার"` $\rightarrow (5 - 0.25) \times 1000 = 4750$
   - `"পৌনে এক লাখ"` $\rightarrow (1 - 0.25) \times 100,000 = 75,000$
3. **On Base Numbers ($N$)**:
   - `"সাড়ে পাঁচ"` $\rightarrow 5 + 0.5 = 5.5$

---

### 8. Hierarchical Grammar & Serial Digit Accumulator

#### `parse_tokens(tokens: List[Token]) -> NumberResult`
The core calculation engine. It evaluates tokens from left to right:

1. **Magnitudes ($\ge 1000$)**:
   - When a magnitude (thousand, lakh, crore) arrives, it flushes the current subtotal:
     $$\text{result} += \text{current} \times \text{magnitude}$$
2. **Hundreds ($100$)**:
   - Multiplies `current` by 100 without flushing:
     $$\text{current} = (\text{current} \lor 1) \times 100$$
3. **Hierarchical Remainder Addition**:
   - If `current` is a multiple of 100 and the next word is $< 100$:
     $$\text{current} += \text{val} \quad (\text{"five hundred 50"} \rightarrow 500 + 50 = 550)$$
   - If `current` is a multiple of 10 ($\ge 20$) and the next word is 1–9:
     $$\text{current} += \text{val} \quad (\text{"twenty 5"} \rightarrow 20 + 5 = 25)$$
4. **Serial Digit Concatenation**:
   - When consecutive single digits (0–9) arrive, they concatenate:
     $$\text{"one two three"} \rightarrow 1, 2, 3 \rightarrow 123$$
   - Mixed speech and raw digits:
     $$\text{"ট্রিপল নাইন 52"} \rightarrow 999 \text{ and } 52 \rightarrow 99952$$
5. **Spoken Year / Paired Number Concatenation**:
   - When two 2-digit numbers arrive back-to-back without a connector:
     $$\text{"twenty thirty"} \rightarrow 20, 30 \rightarrow 2030 \quad (\text{not } 20+30=50)$$
     $$\text{"twenty twenty four"} \rightarrow 20, 20, 4 \rightarrow 2024$$

---

### 9. Leading-Zero Preservation (`LeadingZeroNum`)

#### Why is this critical?
In spoken phone numbers, country codes, OTPs, and room numbers:
- `"জিরো ডাবল এইট"` represents `088` (Bangladesh dial code / serial prefix).
- `"ডাবল জিরো সেভেন"` represents `007`.

In standard Python or Dart, an integer **cannot** store leading zeroes (`088` in Python is a syntax error, and in Dart `int.parse("088")` becomes `88`). Converting `088` to integer `88` permanently destroys the data!

#### The Solution: `LeadingZeroNum`
```python
class LeadingZeroNum(str):
    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, int):
            try:
                return int(str(self)) == other
            except ValueError:
                return False
        return super().__eq__(other)

    def __int__(self) -> int:
        return int(str(self))
```

- **Inherits from `str`**: Perfectly preserves `"088"` formatting.
- **Smart Equality**: `res == "088"` is `True`, AND `res == 88` is `True`.
- **JSON Serializable**: Serializes directly into JSON strings as `"088"`.
- **Dart Port**: In Dart, simply return `String "088"`.

---

### 10. Decimal & Magnitude Resolution

#### `parse_phrase(tokens: Union[List[Token], str]) -> NumberResult`
Supports spoken decimals combined with South Asian magnitudes:
- `"2.5 lakh"` $\rightarrow 2.5 \times 100,000 = 250,000$
- `"1.5 crore"` $\rightarrow 1.5 \times 10,000,000 = 15,000,000$
- `"two point five lakh"` $\rightarrow 250,000$
- `"500.50"` $\rightarrow \text{Decimal}('500.50')$

Accepts either a pre-tokenized `List[Token]` or a raw `str`.

---

### 11. Multi-Number Sentence Extraction

#### `extract_numbers(text: str) -> List[NumberResult]`
Parses real-world sentences containing multiple independent numbers, splitting at `OTHER` words (such as item names, verbs) and adjacent digit boundaries:
```python
extract_numbers("give me 2 kg potatoes for 50 rupees")
# -> [NumberResult(value=2), NumberResult(value=50)]

extract_numbers("two or three")
# -> [NumberResult(value=2), NumberResult(value=3)]

extract_numbers("one lakh or two lakh")
# -> [NumberResult(value=100000), NumberResult(value=200000)]
```
It ensures `2 kg` and `50 rupees` are **never** accidentally added into `52`!

---

## Accuracy & Ambiguity Guardrails

Many parsers try to guess at all costs, resulting in silent, catastrophic data corruption. `word2num` strictly guards against invalid numbers:

1. **Alphanumeric Code Traps**:
   - Model names, chemical formulas, and codes are never converted into numbers:
     - `covid19` $\rightarrow$ `None` (NOT `19`)
     - `mp3` $\rightarrow$ `None` (NOT `3`)
     - `h2o` $\rightarrow$ `None` (NOT `2`)
     - `f16` $\rightarrow$ `None` (NOT `16`)
2. **Duplicate Magnitudes**:
   - `"hundred hundred"` $\rightarrow$ Flagged ambiguous (`duplicate magnitude 'hundred'`)
   - `"one lakh lakh"` $\rightarrow$ Flagged ambiguous (`duplicate magnitude 'lakh'`)
   - `"one thousand thousand"` $\rightarrow$ Flagged ambiguous (`duplicate magnitude 'thousand'`)
3. **Invalid Magnitude Hierarchy**:
   - `"one thousand one lakh"` $\rightarrow$ Flagged ambiguous (`magnitude 'lakh' is not smaller than previous magnitude 'thousand'`)
4. **Uncertain Typos**:
   - `"teroso"` $\rightarrow$ Rejected (guarded against falsely matching `1800` or `1300`).

---

## How to Expand the Codebase (Add Words, Dialects & Languages)

Expanding `word2num` is exceptionally straightforward because all vocabularies are stored in clean, human-readable dictionaries of the form:
```python
VALUE: [variants...]
```

### 1. Adding a New Regional Spelling or Typo
Open [word2num.py](file:///c:/Users/somen/myProjects/word2num/word2num.py) and locate `BENGALI_VOCAB` or `HINDI_VOCAB`.

For example, to add new dialect spellings for 28:
```python
28: [
    'আটাশ', 'আটাশটা', 'আটাশটি', 'আঠাশ',
    'atash', 'athash', 'athas', 'aathash', 'aathas', 'atas'
],
```
Simply add your new spelling into the array!
When the module loads, `_register_vocab()` automatically registers it into `EXACT_MAP`, generates its Roman transliteration, and indexes it into the dual fuzzy matcher.

### 2. Adding a New Compound Hundred
Add an entry to `COMPOUND_HUNDREDS`:
```python
3500: ['পঁয়ত্রিশশো', 'পঁয়ত্রিশশ', 'pointrish sho', 'paintis sau'],
```

### 3. Adding a New Fraction Prefix
Add an entry to `FRACTION_PREFIXES`:
```python
'আড়াই': ('ARAI', 2.5),
'ढाई': ('ARAI', 2.5),
```

### 4. Adding a New Language (e.g. Tamil or Gujarati)
1. Create a `TAMIL_VOCAB: Dict[int, List[str]] = { 1: ['ஒன்று', 'ondru'], ... }`.
2. In `_register_vocab`, register your vocabulary:
   ```python
   _register_vocab(TAMIL_VOCAB, 'WORD')
   ```
3. (Optional) Add akshara transliteration mappings in `transliterate_to_roman` for that script.

---

## Running the Test Suite

The test suite in [test_word2num.py](file:///c:/Users/somen/myProjects/word2num/test_word2num.py) contains **357 automated assertions** testing every section of the pipeline:
- Native Bengali script (0–99 standard & irregular forms)
- Native Hindi script (0–99 standard & irregular forms)
- Compound hundreds (1100–9900)
- Dual fuzzy matcher validation
- Spoken year and serial digit concatenation
- Attached symbols, currencies, and decimal magnitudes
- Alphanumeric trap rejection and ambiguity handling
- Regional ASR phrases, repeaters, fractions, and grammatical case inflections

Run the test suite:
```powershell
python -X utf8 test_word2num.py
```

### Output:
```text
==================================================
TEST SUITE COMPLETE: 357 passed, 0 failed
==================================================
```

Interactive REPL mode is also available:
```powershell
python -X utf8 test_word2num.py --interactive
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

