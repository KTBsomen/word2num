# -*- coding: utf-8 -*-
from word2num import text_to_number, extract_numbers, parse_phrase, tokenize
from decimal import Decimal

passed = 0
failed = 0

def check(text, expected, note=""):
    global passed, failed
    got = text_to_number(text)
    ok = got == expected
    passed += ok
    failed += not ok
    status = "OK " if ok else "FAIL"
    print(f"{status} {text!r:45s} -> got={got!r:15} expected={expected!r} {note}")

def check_ambiguous(text, note=""):
    """Expect None (unresolved OR flagged ambiguous) -- i.e. NOT a confident wrong number."""
    global passed, failed
    results = extract_numbers(text)
    is_amb_or_none = (not results) or all(r.ambiguous or r.value is None for r in results)
    passed += is_amb_or_none
    failed += not is_amb_or_none
    status = "OK " if is_amb_or_none else "FAIL"
    detail = [(r.value, r.ambiguous, r.reason) for r in results]
    print(f"{status} {text!r:45s} -> should be ambiguous/unresolved, got={detail} {note}")

print("=== English 0-99 + magnitudes ===")
check("one", 1)
check("twenty one", 21)
check("one hundred", 100)
check("one hundred five", 105)
check("one thousand", 1000)
check("one thousand five", 1005)
check("twenty five thousand", 25000)
check("one lakh", 100000)
check("one crore", 10000000)
check("one hundred twenty three", 123)
check("one thousand two hundred five", 1205)
check("one lakh twenty five thousand", 125000)
check("one crore five lakh", 10500000)
check("two crore five lakh twenty three thousand four hundred fifty six", 20523456)

print("\n=== Roman Hindi (complete 0-99 spot checks incl. irregulars) ===")
check("ek", 1)
check("do", 2)
check("teen", 3)
check("satrah", 17)
check("atharah", 18)
check("pachis", 25)
check("pachas", 50)
check("bees", 20)
check("chalis", 40)
check("saath", 60)
check("sattar", 70)
check("assi", 80)
check("nabbe", 90)

print("\n=== Roman Bengali (complete 0-99 spot checks incl. irregulars) ===")
check("ek", 1)
check("dui", 2)
check("tin", 3)
check("sotero", 17)
check("atharo", 18)
check("bish", 20)

print("\n=== Compound hundreds: merged AND separated forms ===")
check("baroso", 1200)
check("baro sho", 1200)
check("barosho", 1200)
check("saterosho", 1700)
check("satero sho", 1700)
check("atheroso", 1800)
check("atharo sho", 1800)

print("\n=== THE critical ambiguity test: must NOT silently pick nearest word ===")
check_ambiguous("teroso", "(must not become 1800 or any confident wrong value)")
check_ambiguous("tero", "(must not become 0 via 'zero' false-positive, or anything else)")

print("\n=== Mixed scripts / mixed digit+word ===")
check("২ হাজার ৫", 2005)
check("2 হাজার 5", 2005)
check("२ हजार ५", 2005)
check("ek thousand five", 1005)
check("এক thousand পাঁচ", 1005)
check("দুই হাজার ৫", 2005)
check("25 hundred", 2500)
check("25 hundred 5", 2505)
check("1 lakh 25 thousand 500", 125500)
check("2 crore 5 lakh 25", 20500025)

print("\n=== Digits are atomic; adjacent digit runs are separate numbers ===")
r = extract_numbers("25 5")
vals = [x.value for x in r]
ok = vals == [25, 5]
print(f"{'OK ' if ok else 'FAIL'} '25 5' -> {vals} expected [25, 5] (two numbers, never 30 or 255)")
passed += ok; failed += not ok

print("\n=== Bengali-script phonetic English ASR ===")
check("ওয়ান", 1)
check("টু", 2)
check("থ্রি", 3)
check("থার্টি", 30)
check("থার্টি ওয়ান", 31)

print("\n=== Devanagari-script phonetic English ASR ===")
check("वन", 1)
check("टू", 2)

print("\n=== Stopword protection: real words must never become numbers ===")
check("পাঁচশ টাকা", 500, "(পাঁচশ = native-script compound '5'+hundred-suffix = 500; টাকা correctly excluded)")
check_ambiguous("টাকা")
check_ambiguous("kg")
check_ambiguous("rate")

print("\n=== Number phrase boundaries: multiple numbers per sentence ===")
r = extract_numbers("five kg at twenty five rupees")
vals = [x.value for x in r]
ok = vals == [5, 25]
print(f"{'OK ' if ok else 'FAIL'} 'five kg at twenty five rupees' -> {vals} expected [5, 25]")
passed += ok; failed += not ok

print("\n=== Magnitude order / duplicate validation ===")
check_ambiguous("one thousand one lakh", "(lakh must come before thousand)")
check_ambiguous("one lakh lakh", "(duplicate magnitude)")
check_ambiguous("one thousand thousand", "(duplicate magnitude)")
check("one hundred crore", 1000000000, "(valid Indian usage: hundred before crore multiplies)")

print("\n=== Connectors ===")
check("one hundred and five", 105)
check("একশ আর পাঁচ", 105, "(একশ = native compound 'ek'+hundred-suffix = 100, আর = connector, পাঁচ = 5)")

print("\n=== Decimals ===")
r = parse_phrase(tokenize("twenty five point five"))
print(f"{'OK ' if r.value == Decimal('25.5') else 'FAIL'} 'twenty five point five' -> {r.value} expected 25.5")
passed += (r.value == Decimal('25.5')); failed += (r.value != Decimal('25.5'))

print("\n=== Negative numbers ===")
check("minus five", -5)

print("\n=== ASR noise / spelling variants ===")
check("ek hazar", 1000)
check("ek hajar", 1000)
check("ek hazaar", 1000)
check("do hazar panch", 2005)

print(f"\n\n{'='*50}\nTOTAL: {passed} passed, {failed} failed\n{'='*50}")
while True:
    text = input("Enter text: ")
    print(f"{text_to_number(text)}")