# -*- coding: utf-8 -*-
"""
Synthetic ASR-realistic test generator + evaluator for number_parser.py

Generates 10,000+ (text, expected) cases covering:
  - Bengali (West Bengal register incl. kuri/bish variants), Hindi (Delhi/UP/Bihar
    register using existing vocab), English, Benglish, Hinglish, mixed scripts
  - Digit forms in Bengali/Devanagari numerals mixed with words
  - Compound hundreds, structural compounds, fraction prefixes (sade/paune/sawa),
    standalone fractions (adha/dedh/dhai), magnitude combinations, decimals,
    currency/format noise, repeaters, sequential digit concatenation
  - Real-world shop-talk sentences (multi-number) using extract_numbers()
  - Specific 'probe' cases designed to stress-test known tricky spots

Each case has an EXPECTED value computed independently from the building
blocks used to construct it (not by calling the parser), so this is a real
expected-vs-actual test, not a self-consistency check.
"""

import sys, random, json, csv, string
sys.path.insert(0, '/home/claude')
import word2num as npmod

random.seed(42)

BENGALI_VOCAB = npmod.BENGALI_VOCAB
HINDI_VOCAB = npmod.HINDI_VOCAB
ENGLISH_VOCAB = npmod.ENGLISH_VOCAB
MAGNITUDE_VOCAB = npmod.MAGNITUDE_VOCAB
FRACTION_MULTIPLIERS = npmod.FRACTION_MULTIPLIERS
FRACTION_PREFIXES = npmod.FRACTION_PREFIXES
COMPOUND_HUNDREDS = npmod.COMPOUND_HUNDREDS
script_of = npmod.script_of
text_to_number = npmod.text_to_number
extract_numbers = npmod.extract_numbers

BN_DIGIT_TRANS = str.maketrans('0123456789', '০১২৩৪৫৬৭৮৯')
HI_DIGIT_TRANS = str.maketrans('0123456789', '०१२३४५६७८९')


def bn_digits(n):
    return str(n).translate(BN_DIGIT_TRANS)


def hi_digits(n):
    return str(n).translate(HI_DIGIT_TRANS)


cases = []  # list of dict: text, expected, category, lang, expect_type


def add(text, expected, category, lang, expect_type='scalar'):
    cases.append({
        'text': text, 'expected': expected, 'category': category,
        'lang': lang, 'expect_type': expect_type,
    })


# ----------------------------------------------------------------------
# 1. SIMPLE NUMBER WORDS (0-99) ACROSS BENGALI / HINDI / ENGLISH VOCAB
# ----------------------------------------------------------------------
UNIT_BUNDLES = {
    'bn': [' কেজি', ' গ্রাম', ' টাকা', ' পিস', ' লিটার', ''],
    'hi': [' किलो', ' रुपये', ' पीस', ' लीटर', ''],
    'en': [' kg', ' rupees', ' pieces', ' litre', ''],
}
SENTENCE_TEMPLATES = {
    'bn': ["আমাকে {n} কেজি চাল দাও", "দাম {n} টাকা", "{n} পিস দিন", "ভাই {n} টা দাও দিকি"],
    'hi': ["मुझे {n} किलो चावल चाहिए", "दाम {n} रुपये है", "{n} पीस दे दो", "भाई {n} दे दो"],
    'en': ["give me {n} kg rice", "price is {n} rupees", "give {n} pieces please"],
}

for vocab, langname in [(BENGALI_VOCAB, 'bn'), (HINDI_VOCAB, 'hi'), (ENGLISH_VOCAB, 'en')]:
    for value, variants in vocab.items():
        for variant in variants:
            add(variant, value, 'simple_number', langname)
            unit = random.choice(UNIT_BUNDLES[langname])
            add(f"{variant}{unit}", value, 'simple_number_unit', langname)
            tmpl = random.choice(SENTENCE_TEMPLATES[langname])
            add(tmpl.format(n=variant), value, 'simple_number_sentence', langname)


# ----------------------------------------------------------------------
# 2. TYPO / NOISY-ASR VARIANTS (probe the fuzzy matcher)
# ----------------------------------------------------------------------
def make_typo(word):
    if len(word) < 4:
        return None
    idx = random.randrange(1, len(word) - 1)
    chars = list(word)
    if chars[idx].isascii() and chars[idx].isalpha():
        chars[idx] = random.choice([c for c in string.ascii_lowercase if c != chars[idx]])
        return ''.join(chars)
    return None

for vocab, langname in [(BENGALI_VOCAB, 'bn'), (HINDI_VOCAB, 'hi'), (ENGLISH_VOCAB, 'en')]:
    for value, variants in vocab.items():
        for variant in variants:
            if script_of(variant) == 'latin' and len(variant) >= 4 and ' ' not in variant:
                t = make_typo(variant)
                if t and t != variant:
                    add(t, value, 'simple_number_typo', langname)


# ----------------------------------------------------------------------
# 3. COMPOUND HUNDREDS (150, 250, 1100..2900)
# ----------------------------------------------------------------------
for value, variants in COMPOUND_HUNDREDS.items():
    for variant in variants:
        add(variant, value, 'compound_hundred', 'mixed')
        if script_of(variant) == 'bengali':
            add(f"{variant} টাকা", value, 'compound_hundred_unit', 'bn')
        elif script_of(variant) == 'devanagari':
            add(f"{variant} रुपये", value, 'compound_hundred_unit', 'hi')
        else:
            add(f"{variant} rupees", value, 'compound_hundred_unit', 'en')


# ----------------------------------------------------------------------
# 4. STRUCTURAL COMPOUND HUNDREDS (word + শ/শো, word + সৌ)
# ----------------------------------------------------------------------
GUARDED_BN = {'তেরো', 'তের', 'তেড়ো', 'তেড়'}
for value, variants in BENGALI_VOCAB.items():
    if 1 <= value <= 99:
        bn_words = [v for v in variants if script_of(v) == 'bengali' and v not in GUARDED_BN]
        for bn_word in bn_words[:2]:
            for suf in ['শ', 'শো']:
                add(bn_word + suf, value * 100, 'structural_compound_bn', 'bn')

for value, variants in HINDI_VOCAB.items():
    if 1 <= value <= 99:
        hi_words = [v for v in variants if script_of(v) == 'devanagari']
        for hi_word in hi_words[:2]:
            add(hi_word + 'सौ', value * 100, 'structural_compound_hi', 'hi')
            add(hi_word + ' सौ', value * 100, 'structural_compound_hi', 'hi')

# Guarded ambiguous case: 'তেরোশ'/'তেরোশো' must NOT resolve to 1300
add('তেরোশ', None, 'structural_compound_guarded', 'bn')
add('তেরোশো', None, 'structural_compound_guarded', 'bn')
add('teroso', None, 'structural_compound_guarded', 'en')


# ----------------------------------------------------------------------
# 5. FRACTION PREFIXES: sade/paune/sawa + number / magnitude / compound
# ----------------------------------------------------------------------
FRACTION_PREFIX_SAMPLES = {'SADE': 0.5, 'PAUNE': -0.25, 'SAWA': 0.25}
prefix_words = {
    'SADE': [('সাড়ে', 'bn'), ('साढ़े', 'hi'), ('sade', 'en')],
    'PAUNE': [('পৌনে', 'bn'), ('पौने', 'hi'), ('paune', 'en')],
    'SAWA': [('সওয়া', 'bn'), ('सवा', 'hi'), ('sawa', 'en')],
}

number_word_lookup = {}
for value in range(0, 100):
    words = []
    words += [(w, 'bn') for w in BENGALI_VOCAB.get(value, []) if script_of(w) == 'bengali']
    words += [(w, 'hi') for w in HINDI_VOCAB.get(value, []) if script_of(w) == 'devanagari']
    words += [(w, 'en') for w in ENGLISH_VOCAB.get(value, [])]
    number_word_lookup[value] = words

for ptype, delta in FRACTION_PREFIX_SAMPLES.items():
    for pword, plang in prefix_words[ptype]:
        for n in range(1, 21):
            for nword, nlang in number_word_lookup.get(n, [])[:2]:
                text = f"{pword} {nword}"
                expected = n + delta
                if expected == int(expected):
                    expected = int(expected)
                add(text, expected, f'fraction_prefix_{ptype}', plang)

magnitude_samples = {100: 'hundred', 1000: 'thousand', 100000: 'lakh', 10000000: 'crore'}

# prefix + bare magnitude: "sade hazar" -> 1500, "sawa lakh" -> 125000
for ptype, delta in FRACTION_PREFIX_SAMPLES.items():
    for pword, plang in prefix_words[ptype]:
        for magval, magword in magnitude_samples.items():
            text = f"{pword} {magword}"
            expected = int((1 + delta) * magval)
            add(text, expected, f'fraction_prefix_bare_magnitude_{ptype}', plang)

# prefix + number + magnitude: "sade tin hazar" -> 3500
for ptype, delta in FRACTION_PREFIX_SAMPLES.items():
    for pword, plang in prefix_words[ptype]:
        for n in [1, 2, 3, 4, 5]:
            for nword, nlang in number_word_lookup.get(n, [])[:1]:
                for magval, magword in magnitude_samples.items():
                    text = f"{pword} {nword} {magword}"
                    expected = int((n + delta) * magval)
                    add(text, expected, f'fraction_prefix_magnitude_{ptype}', plang)

# prefix + compound hundred: "sade baroso" -> 1250
for value, variants in COMPOUND_HUNDREDS.items():
    variant = variants[0]
    for ptype, delta in FRACTION_PREFIX_SAMPLES.items():
        pword, plang = prefix_words[ptype][0]
        text = f"{pword} {variant}"
        base_n = value / 100
        newval = (base_n + delta) * 100
        expected = int(newval) if newval == int(newval) else newval
        add(text, expected, f'fraction_prefix_compound_{ptype}', 'mixed')


# ----------------------------------------------------------------------
# 6. STANDALONE FRACTION WORDS (adha/sawa/dedh/dhai)
# ----------------------------------------------------------------------
for val, variants in FRACTION_MULTIPLIERS.items():
    for variant in variants:
        # real-world shop phrasing: "adha kg", "dedh kg" (weight unit is a STOPWORD)
        for unit in ['kg', 'কেজি', 'किलो']:
            text = f"{variant} {unit}"
            add(text, val, 'fraction_standalone_weightunit', 'mixed')
    for variant in variants[:2]:
        # combined with a real MAGNITUDE word - should compose properly
        for magval, magword in magnitude_samples.items():
            text = f"{variant} {magword}"
            expected = int(val * magval)
            add(text, expected, 'fraction_standalone_magnitude', 'mixed')


# ----------------------------------------------------------------------
# 7. MAGNITUDE COMBINATIONS (digit/word x hundred/thousand/lakh/crore/million/billion)
# ----------------------------------------------------------------------
for n in [1, 2, 3, 5, 7, 9, 10, 15, 20, 25, 50, 99, 100, 200, 500]:
    for magval, magvariants in MAGNITUDE_VOCAB.items():
        magword = magvariants[0]
        for nword_source in ['digit', 'bn', 'hi', 'en']:
            if nword_source == 'digit':
                nrepr = str(n)
            elif nword_source == 'bn':
                cand = [w for w in BENGALI_VOCAB.get(n, []) if script_of(w) == 'bengali']
                if not cand:
                    continue
                nrepr = cand[0]
            elif nword_source == 'hi':
                cand = [w for w in HINDI_VOCAB.get(n, []) if script_of(w) == 'devanagari']
                if not cand:
                    continue
                nrepr = cand[0]
            else:
                cand = ENGLISH_VOCAB.get(n)
                if not cand:
                    continue
                nrepr = cand[0]
            text = f"{nrepr} {magword}"
            expected = n * magval
            add(text, expected, 'magnitude_combo', nword_source)


# ----------------------------------------------------------------------
# 8. DECIMAL + MAGNITUDE (digit-decimal and native-digit-decimal)
# ----------------------------------------------------------------------
decimal_cases = [(2.5, 'lakh', 250000), (1.5, 'thousand', 1500),
                  (3.75, 'crore', 37500000), (0.5, 'million', 500000),
                  (2.25, 'lakh', 225000)]
for val, magword, expected in decimal_cases:
    add(f"{val} {magword}", int(expected), 'decimal_magnitude_digit', 'en')
    intpart, fracpart = str(val).split('.')
    bn_val = f"{bn_digits(intpart)}.{bn_digits(fracpart)}"
    add(f"{bn_val} {magword}", int(expected), 'decimal_magnitude_bn_digit', 'bn')
    hi_val = f"{hi_digits(intpart)}.{hi_digits(fracpart)}"
    add(f"{hi_val} {magword}", int(expected), 'decimal_magnitude_hi_digit', 'hi')

word_decimal_cases = [
    ("দুই দশমিক পাঁচ লাখ", 250000),
    ("तीन दशमलव पाँच लाख", 350000),
    ("one point five million", 1500000),
    ("এক দশমিক পাঁচ হাজার", 1500),
]
for text, expected in word_decimal_cases:
    add(text, expected, 'decimal_magnitude_word', 'mixed')


# ----------------------------------------------------------------------
# 9. CURRENCY / FORMATTING NOISE
# ----------------------------------------------------------------------
noise_cases = [
    ("₹500", 500), ("Rs.500/-", 500), ("Rs. 500", 500), ("৳৫০০", 500), ("INR500", 500),
    ("1,25,000", 125000), ("2,500", 2500), ("10lakh", 1000000), ("50kg", 50),
    ("5ta", 5), ("১০টা", 10), ("1st", 1), ("2nd", 2), ("3rd", 3), ("4th", 4), ("10th", 10),
    ("twenty-five", 25), ("$100", 100), ("Tk.200", 200),
]
for text, expected in noise_cases:
    add(text, expected, 'noise_formatting', 'mixed')


# ----------------------------------------------------------------------
# 10. REPEATERS (double/triple)
# ----------------------------------------------------------------------
repeater_words = [('double', 2, 'en'), ('triple', 3, 'en'), ('ডাবল', 2, 'bn'),
                   ('ট্রিপল', 3, 'bn'), ('डबल', 2, 'hi'), ('ट्रिपल', 3, 'hi')]
for word, mult, rlang in repeater_words:
    for d in range(0, 10):
        if rlang == 'bn':
            cand = [w for w in BENGALI_VOCAB.get(d, []) if script_of(w) == 'bengali']
        elif rlang == 'hi':
            cand = [w for w in HINDI_VOCAB.get(d, []) if script_of(w) == 'devanagari']
        else:
            cand = ENGLISH_VOCAB.get(d, [])
        if not cand:
            continue
        dword = cand[0]
        text = f"{word} {dword}"
        expected = int(str(d) * mult)
        add(text, expected, 'repeater', rlang)


# ----------------------------------------------------------------------
# 11. SEQUENTIAL / PAIRED DIGIT-WORD CONCATENATION (per module docstring)
# ----------------------------------------------------------------------
seq_cases = [
    ("twenty thirty", 2030), ("চার শূন্য দুই", 402), ("nine nine", 99),
    ("এক দুই তিন", 123), ("do teen", 23),
]
for text, expected in seq_cases:
    add(text, expected, 'sequential_concat', 'mixed')


# ----------------------------------------------------------------------
# 12. DIGIT-IN-NATIVE-SCRIPT MIXED WITH WORDS (Bengali/Devanagari numerals)
# ----------------------------------------------------------------------
for n in range(1, 1000, 7):
    add(f"{bn_digits(n)} টাকা", n, 'digit_mixed_bn', 'bn')
    add(f"{hi_digits(n)} रुपये", n, 'digit_mixed_hi', 'hi')
    add(f"দাম {bn_digits(n)} টাকা মতো", n, 'digit_mixed_bn_sentence', 'bn')
    add(f"दाम {hi_digits(n)} रुपये है", n, 'digit_mixed_hi_sentence', 'hi')


# ----------------------------------------------------------------------
# 13. REGIONAL DISTINCTION: West Bengal 'কুড়ি' vs. 'বিশ' for 20
# ----------------------------------------------------------------------
regional_wb_20 = [
    ("কুড়ি টাকা", 20), ("বিশ টাকা", 20), ("কুড়ি কেজি চাল দাও", 20),
    ("kuri taka", 20), ("bish taka", 20), ("কুড়িটা আম দাও", 20),
]
for text, expected in regional_wb_20:
    add(text, expected, 'regional_wb_bengali_20', 'bn')


# ----------------------------------------------------------------------
# 14. SHOP-TALK MULTI-NUMBER SENTENCES (Delhi/UP/Bihar Hindi, West Bengal
#     Bengali, English, and mixed) -- evaluated with extract_numbers()
# ----------------------------------------------------------------------
shop_templates_bn = [
    "{item} {n1} কেজি {n2} টাকা",
    "আমাকে {n1} কেজি {item} আর {n2} পিস {item2} দিন",
    "{n1} টাকার {item} আর {n2} টাকার {item2}",
    "দাদা {n1} কেজি চাল দাও, দাম হবে {n2} টাকা",
    "{item} দাম কত? {n1} টাকা কেজি। আচ্ছা {n2} কেজি দিন",
]
shop_templates_hi = [
    "{item} {n1} किलो {n2} रुपये",
    "मुझे {n1} किलो {item} और {n2} पीस {item2} चाहिए",
    "{n1} रुपये का {item} और {n2} रुपये का {item2}",
    "भैया {n1} किलो चावल दो, दाम {n2} रुपये",
    "{item} का भाव क्या है? {n1} रुपये किलो। ठीक है {n2} किलो दे दो",
]
shop_templates_en = [
    "{item} {n1} kg {n2} rupees",
    "give me {n1} kg {item} and {n2} pieces {item2}",
    "price of {item} is {n1} rupees, give me {n2} kg",
]
items_bn = ['চাল', 'আলু', 'ডাল', 'চিনি', 'আটা', 'পেঁয়াজ']
items_hi = ['चावल', 'आलू', 'दाल', 'चीनी', 'आटा', 'प्याज']
items_en = ['rice', 'potato', 'sugar', 'flour', 'onion']


def repr_num(n, lang):
    choice = random.choice(['digit', 'word'])
    if choice == 'digit' or n > 99:
        return str(n)
    if lang == 'bn':
        cand = [w for w in BENGALI_VOCAB.get(n, []) if script_of(w) == 'bengali']
    elif lang == 'hi':
        cand = [w for w in HINDI_VOCAB.get(n, []) if script_of(w) == 'devanagari']
    else:
        cand = ENGLISH_VOCAB.get(n, [])
    return random.choice(cand) if cand else str(n)


for i in range(5000):
    lang = random.choice(['bn', 'hi', 'en'])
    n1 = random.randint(1, 999)
    n2 = random.randint(1, 999)
    r1 = repr_num(n1, lang)
    r2 = repr_num(n2, lang)
    if lang == 'bn':
        tmpl = random.choice(shop_templates_bn)
        item, item2 = random.choice(items_bn), random.choice(items_bn)
    elif lang == 'hi':
        tmpl = random.choice(shop_templates_hi)
        item, item2 = random.choice(items_hi), random.choice(items_hi)
    else:
        tmpl = random.choice(shop_templates_en)
        item, item2 = random.choice(items_en), random.choice(items_en)
    text = tmpl.format(n1=r1, n2=r2, item=item, item2=item2)
    add(text, [n1, n2], 'shop_multi_number', lang, expect_type='list')


# ----------------------------------------------------------------------
# 15. HAND-CRAFTED REALISTIC SHOP PHRASES (carefully verified oracle)
# ----------------------------------------------------------------------
handcrafted = [
    # (text, expected, lang)
    ("দেড়শো টাকা কেজি", 150, 'bn'),
    ("আড়াইশো গ্রাম চিনি", 250, 'bn'),
    ("বারোশো টাকা দাম", 1200, 'bn'),
    ("panch so taka", 500, 'bn'),
    ("তিনশো টাকা দিয়ে দাও", 300, 'bn'),
    ("do sau rupaye", 200, 'hi'),
    ("teen sau pachas rupaye", 350, 'hi'),
    ("पाँच सौ रुपये", 500, 'hi'),
    ("panch sau", 500, 'hi'),
    ("dedh sau rupaye", 150, 'hi'),
    ("ढाई सौ ग्राम", 250, 'hi'),
    ("sade tin so taka", 350, 'bn'),
    ("sawa char kg", 4.25, 'mixed'),
    ("paune char kg", 3.75, 'mixed'),
    ("dedh kg lagega", 1.5, 'mixed'),  # standalone fraction + non-stopword word - probe
    ("panch kilo aloo", 5, 'hi'),
    ("dosh kg chal dao", 10, 'bn'),
    ("bees rupaye kilo", 20, 'hi'),
    ("kuri taka kg", 20, 'bn'),
    ("একশো পঞ্চাশ টাকা", 150, 'bn'),
    ("एक सौ पचास रुपये", 150, 'hi'),
    ("dui hazar panchsho taka", 2500, 'bn'),
    ("do hazar paanch sau rupaye", 2500, 'hi'),
    ("pach hazar", 5000, 'bn'),
    ("panch hazar rupaye", 5000, 'hi'),
    ("dash hajar taka", 10000, 'bn'),
    ("dus hazar rupaye", 10000, 'hi'),
    ("ek lakh rupaye", 100000, 'hi'),
    ("এক লাখ টাকা", 100000, 'bn'),
    ("dedh lakh", 150000, 'mixed'),
    ("ढाई लाख रुपये", 250000, 'hi'),
    ("panch crore", 50000000, 'hi'),
    ("dui kuti", None, 'bn'),  # not real vocab, expect no match (nonsense probe)
]
for text, expected, lang in handcrafted:
    add(text, expected, 'handcrafted_realistic', lang)


# ----------------------------------------------------------------------
# 16. CONNECTOR-JOINED SEPARATE NUMBERS (probe for merge bugs)
# ----------------------------------------------------------------------
connector_probe = [
    ("পঞ্চাশ আর কুড়ি", [50, 20]),
    ("fifty and twenty", [50, 20]),
    ("पचास और बीस", [50, 20]),
    ("dash ar pach", [10, 5]),
    ("do aur teen", [2, 3]),
    ("100 আর 200", [100, 200]),
]
for text, expected in connector_probe:
    add(text, expected, 'connector_merge_probe', 'mixed', expect_type='list')


# ----------------------------------------------------------------------
# 17. NEGATIVE NUMBERS
# ----------------------------------------------------------------------
neg_cases = [
    ("minus 5", -5), ("মাইনাস পাঁচ", -5), ("माइनस पांच", -5), ("negative 10", -10),
]
for text, expected in neg_cases:
    add(text, expected, 'negative_number', 'mixed')


print(f"TOTAL CASES GENERATED: {len(cases)}")

# ----------------------------------------------------------------------
# EVALUATION
# ----------------------------------------------------------------------
def to_comparable(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)


results = []
pass_count = 0
fail_count = 0
by_category = {}

for c in cases:
    text = c['text']
    expected = c['expected']
    cat = c['category']
    by_category.setdefault(cat, {'pass': 0, 'fail': 0})

    if c['expect_type'] == 'list':
        nums = extract_numbers(text)
        actual = [n.value for n in nums]
        actual_cmp = [to_comparable(v) for v in actual]
        expected_cmp = [to_comparable(v) for v in expected]
        ok = actual_cmp == expected_cmp
    else:
        actual = text_to_number(text)
        ok = to_comparable(actual) == to_comparable(expected)

    if ok:
        pass_count += 1
        by_category[cat]['pass'] += 1
    else:
        fail_count += 1
        by_category[cat]['fail'] += 1

    results.append({
        'text': text, 'lang': c['lang'], 'category': cat,
        'expected': json.dumps(expected, ensure_ascii=False),
        'actual': json.dumps(actual if c['expect_type'] != 'list' else actual, default=str, ensure_ascii=False),
        'pass': ok,
    })

total = len(cases)
print(f"PASS: {pass_count}  FAIL: {fail_count}  ACCURACY: {pass_count/total*100:.2f}%")

# Write full results CSV
with open('test_results.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['text', 'lang', 'category', 'expected', 'actual', 'pass'])
    w.writeheader()
    for r in results:
        w.writerow(r)

# Category summary
cat_summary = []
for cat, stats in sorted(by_category.items(), key=lambda kv: -kv[1]['fail']):
    tot = stats['pass'] + stats['fail']
    acc = stats['pass'] / tot * 100 if tot else 0
    cat_summary.append((cat, stats['pass'], stats['fail'], tot, acc))

with open('category_summary.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['category', 'pass', 'fail', 'total', 'accuracy_pct'])
    for row in cat_summary:
        w.writerow(row)

print("\n=== CATEGORY BREAKDOWN (worst first) ===")
for cat, p, fl, tot, acc in cat_summary:
    print(f"{cat:40s} pass={p:5d} fail={fl:5d} total={tot:5d} acc={acc:6.2f}%")

# Sample mismatches per category (up to 8 each) for the report
mismatch_samples = {}
for r in results:
    if not r['pass']:
        mismatch_samples.setdefault(r['category'], [])
        if len(mismatch_samples[r['category']]) < 8:
            mismatch_samples[r['category']].append(r)

with open('mismatch_samples.json', 'w', encoding='utf-8') as f:
    json.dump(mismatch_samples, f, ensure_ascii=False, indent=2)

print("\nSaved: test_results.csv, category_summary.csv, mismatch_samples.json")