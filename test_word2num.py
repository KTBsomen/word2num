# -*- coding: utf-8 -*-
"""
Comprehensive test suite for multilingual ASR number parser.
Regional-first testing with heavy focus on native Bengali (বাংলা) and Hindi (हिन्दी) scripts.
"""

import sys
import io
from decimal import Decimal
from word2num import text_to_number, extract_numbers, parse_phrase, tokenize

# Ensure UTF-8 output encoding on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

passed = 0
failed = 0


def check(text, expected, note=""):
    global passed, failed
    got = text_to_number(text)
    ok = got == expected
    passed += ok
    failed += not ok
    status = "OK " if ok else "FAIL"
    print(f"{status} {text!r:50s} -> got={got!r:15} expected={expected!r} {note}")


def check_extract(text, expected_values, note=""):
    global passed, failed
    results = extract_numbers(text)
    vals = [r.value for r in results]
    ok = vals == expected_values
    passed += ok
    failed += not ok
    status = "OK " if ok else "FAIL"
    print(f"{status} {text!r:50s} -> got={vals!r:15} expected={expected_values!r} {note}")


def check_ambiguous(text, note=""):
    """Expect None (unresolved OR flagged ambiguous) -- i.e. NOT a confident wrong number."""
    global passed, failed
    results = extract_numbers(text)
    is_amb_or_none = (not results) or all(r.ambiguous or r.value is None for r in results)
    passed += is_amb_or_none
    failed += not is_amb_or_none
    status = "OK " if is_amb_or_none else "FAIL"
    detail = [(r.value, r.ambiguous, r.reason) for r in results]
    print(f"{status} {text!r:50s} -> should be ambiguous/unresolved, got={detail} {note}")


# ======================================================================
# SECTION 1: NATIVE BENGALI SCRIPT (বাংলা) - PRIMARY REGIONAL FOCUS
# ======================================================================
print("=== Native Bengali Script (0-99 Standard & Irregulars) ===")
check("শূন্য", 0)
check("এক", 1)
check("দুই", 2)
check("তিন", 3)
check("চার", 4)
check("পাঁচ", 5)
check("পাচ", 5, "(ASR candrabindu-dropped)")
check("ছয়", 6)
check("সাত", 7)
check("আট", 8)
check("নয়", 9)
check("দশ", 10)
check("এগারো", 11)
check("বারো", 12)
check("তেরো", 13)
check("চৌদ্দ", 14)
check("চোদ্দ", 14)
check("পনেরো", 15)
check("ষোল", 16)
check("সতেরো", 17)
check("আঠারো", 18)
check("উনিশ", 19)
check("বিশ", 20)
check("কুড়ি", 20)
check("একুশ", 21)
check("পঁচিশ", 25)
check("পচিশ", 25, "(candrabindu-dropped)")
check("ত্রিশ", 30)
check("চল্লিশ", 40)
check("পঞ্চাশ", 50)
check("পচাশ", 50, "(ASR transliterated)")
check("ষাট", 60)
check("সত্তর", 70)
check("আশি", 80)
check("নব্বই", 90)
check("পঁচাত্তর", 75)
check("নিরানব্বই", 99)

print("\n=== Native Bengali Compound Hundreds ===")
check("বারোশো", 1200)
check("বারশো", 1200)
check("বারশ", 1200)
check("তেরোশো", 1300)
check("তেরশো", 1300)
check("তেরশ", 1300)
check("চৌদ্দশো", 1400)
check("চোদ্দশো", 1400)
check("পনেরোশো", 1500)
check("পনেরশ", 1500)
check("ষোলশো", 1600)
check("সতেরোশো", 1700)
check("সতেরশ", 1700)
check("আঠারোশো", 1800)
check("আঠারশ", 1800)
check("উনিশশো", 1900)
check("বিশশো", 2000)
check("পঁচিশশো", 2500)
check("পচিশশো", 2500)
check("পাঁচশ", 500)
check("পাচশ", 500)
check("একশ", 100)
check("একশো", 100)
check("দুইশ", 200)
check("তিনশ", 300)
check("চারশ", 400)

print("\n=== Native Bengali Classifiers & Suffixes ===")
check("একটা", 1)
check("একটি", 1)
check("দুটো", 2)
check("দুইটি", 2)
check("দুটি", 2)
check("তিনটি", 3)
check("তিনটে", 3)
check("চারটে", 4)
check("চারটি", 4)
check("পাঁচটা", 5)
check("পাচটা", 5)
check("দশটা", 10)
check("একুশটা", 21)
check("পঁচিশটা", 25)
check("১০টা", 10)
check("১০০টা", 100)
check("একশোটা", 100)

print("\n=== Native Bengali Decimal Magnitudes & Spoken Fractions ===")
check("২.৫ লাখ", 250000)
check("১.৫ কোটি", 15000000)
check("৩.৫ হাজার", 3500)
check("০.৫ লাখ", 50000)
check("দেড় লাখ", 150000)
check("আড়াই হাজার", 2500)
check("আড়াই কোটি", 25000000)
check("আধা লাখ", 50000)
check("এক হাজার পাঁচ", 1005)
check("দুই কোটি পাঁচ লাখ", 20500000)
check("একশ আর পাঁচ", 105)

print("\n=== Native Bengali Currency, Fillers & Disfluencies ===")
check("পাঁচশ টাকা", 500)
check("৳৫০০", 500)
check("৫০০/-", 500)
check("১,২৫,০০০", 125000)
check("৫০০টাকা", 500)
check("মানে প্রায় পাঁচশ টাকা", 500)
check("মোটামুটি দশ হাজার", 10000)
check("প্রায় দুই কোটি টাকা", 20000000)

print("\n=== Native Bengali Serial Numbers & Disjunctions ===")
check("চার শূন্য দুই", 402, "(spoken serial digits: 4 0 2 -> 402)")
check("সাত শূন্য সাত", 707, "(spoken serial digits: 7 0 7 -> 707)")
check("এক দুই তিন", 123, "(spoken serial digits: 1 2 3 -> 123)")
check_extract("দুই বা তিন", [2, 3], "(disjunction 'ba')")
check_extract("পাঁচ থেকে দশ", [5, 10], "(range 'theke')")

print("\n=== Transliteration-based Bengali ASR Typos ===")
check("আঠার", 18, "(dropped ending -o)")
check("পচিষ", 25, "(sh/s/sh consonant interchange)")
check("পচিস", 25, "(sh/s/sh consonant interchange)")


# ======================================================================
# SECTION 2: NATIVE DEVANAGARI / HINDI SCRIPT (हिन्दी) - PRIMARY REGIONAL FOCUS
# ======================================================================
print("\n=== Native Hindi Script (0-99 Standard & Irregulars) ===")
check("शून्य", 0)
check("एक", 1)
check("दो", 2)
check("तीन", 3)
check("चार", 4)
check("पाँच", 5)
check("पांच", 5)
check("पाच", 5)
check("छह", 6)
check("छः", 6)
check("सात", 7)
check("आठ", 8)
check("नौ", 9)
check("दस", 10)
check("ग्यारह", 11)
check("बारह", 12)
check("तेरह", 13)
check("चौदह", 14)
check("पंद्रह", 15)
check("सोलह", 16)
check("सत्रह", 17)
check("अठारह", 18)
check("उन्नीस", 19)
check("बीस", 20)
check("इक्कीस", 21)
check("पच्चीस", 25)
check("तीस", 30)
check("चालीस", 40)
check("पचास", 50)
check("साठ", 60)
check("सत्तर", 70)
check("अस्सी", 80)
check("नब्बे", 90)
check("पचहत्तर", 75)
check("निन्यानवे", 99)

print("\n=== Native Hindi Compound Hundreds ===")
check("बारह सौ", 1200)
check("बारहसौ", 1200)
check("तेरह सौ", 1300)
check("तेरहसौ", 1300)
check("चौदह सौ", 1400)
check("पंद्रह सौ", 1500)
check("सोलह सौ", 1600)
check("सत्रह सौ", 1700)
check("अठारह सौ", 1800)
check("उन्नीस सौ", 1900)
check("बीस सौ", 2000)
check("पच्चीस सौ", 2500)
check("पाँच सौ", 500)
check("पांच सौ", 500)
check("एक सौ", 100)
check("दो सौ", 200)
check("तीन सौ", 300)

print("\n=== Native Hindi Decimal Magnitudes & Spoken Fractions ===")
check("२.५ लाख", 250000)
check("१.५ करोड़", 15000000)
check("३.५ हजार", 3500)
check("डेढ़ लाख", 150000)
check("ढाई करोड़", 25000000)
check("ढाई हजार", 2500)
check("सवा हजार", 1250)
check("आधा लाख", 50000)
check("दो हजार पांच", 2005)

print("\n=== Native Hindi Currency, Fillers & Disfluencies ===")
check("लगभग पाँच हजार रुपये", 5000)
check("करीब बीस लाख", 2000000)
check("₹२०००", 2000)
check("२०००/-", 2000)
check("१,५०,०००", 150000)
check("चार शून्य दो", 402, "(serial digits 4 0 2 -> 402)")
check_extract("दो या तीन", [2, 3], "(disjunction 'ya')")
check_extract("पाँच से दस", [5, 10], "(range 'se')")


# ======================================================================
# SECTION 3: DEDICATED SHORT-WORD FUZZY MATCHING (Lengths 2-4)
# ======================================================================
print("\n=== Short-Word Dedicated Fuzzy Matching (Lengths 2-4) ===")
check("ak", 1, "(short typo for ek -> 1)")
check("ik", 1, "(short typo for ek -> 1)")
check("duy", 2, "(short typo for dui -> 2)")
check("dwi", 2, "(short typo for dui -> 2)")
check("tyn", 3, "(short typo for tin -> 3)")
check("caar", 4, "(short typo for char -> 4)")
check("coy", 6, "(short typo for choy -> 6)")
check("sat", 7, "(short variant for saat -> 7)")
check("ath", 8, "(short variant for aath -> 8)")
check("nay", 9, "(short variant for noy -> 9)")
check("dos", 10, "(short typo for dosh -> 10)")

print("\n=== Short-Word Stopword Protection (Must NOT false-positive) ===")
check_ambiguous("to", "(preposition 'to' must NOT become 2)")
check_ambiguous("for", "(preposition 'for' must NOT become 4)")
check_ambiguous("won", "(verb 'won' must NOT become 1)")
check_ambiguous("ate", "(verb 'ate' must NOT become 8)")
check_ambiguous("by", "(preposition 'by' must NOT become a number)")
check_ambiguous("in", "(preposition 'in' must NOT become a number)")


# ======================================================================
# SECTION 4: ROMANIZED BENGALI, ROMANIZED HINDI & ENGLISH BASELINE
# ======================================================================
print("\n=== English 0-99 + Magnitudes ===")
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

print("\n=== Roman Hindi Spot Checks ===")
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

print("\n=== Roman Bengali Spot Checks ===")
check("ek", 1)
check("dui", 2)
check("tin", 3)
check("sotero", 17)
check("atharo", 18)
check("athar", 18)
check("bish", 20)
check("athas", 28, "(Bengali/Hindi 28: আটাশ / अट्ठाईस / athas, NOT 18)")
check("athass", 28)
check("atass", 28)
check("athash", 28)
check("atash", 28)
check("atthais", 28)

print("\n=== Roman Compound Hundreds ===")
check("baroso", 1200)
check("baro sho", 1200)
check("barosho", 1200)
check("saterosho", 1700)
check("satero sho", 1700)
check("atheroso", 1800)
check("atharo sho", 1800)
check("atharsho", 1800)
check("athasso", 2800, "(athas [28] + so [100] = 2800)")
check("athassho", 2800)
check("athasho", 2800)
check("atashsho", 2800)
check("atasho", 2800)
check("ekusho", 2100)
check("ekusso", 2100)
check("pochisho", 2500)
check("pochisso", 2500)
check("tinso", 300)
check("tinso panch", 305)
check("dui hazar tinso panch", 2305, "(dui hazar [2000] + tinso [300] + panch [5] = 2305)")
check("charso bees", 420)
check("পাঁচ হাজার তিনশ পাঁচ", 5305)
check("पाँच हजार तीन सौ पाँच", 5305)
check("আঠাড়োশো তিরিশ", 1830, "(Bengali ASR with ড়: আঠাড়োশো [1800] + তিরিশ [30] = 1830)")
check("আঠারোশো তিরিশ", 1830)
check("আঠাড়শো তিরিশ", 1830)
check("আঠাড়ো", 18)
check("আঠাড়", 18)

print("\n=== Script-transcribed Phonetic English ASR ===")
check("ওয়ান", 1)
check("টু", 2)
check("থ্রি", 3)
check("থার্টি", 30)
check("থার্টি ওয়ান", 31)
check("वन", 1)
check("टू", 2)


# ======================================================================
# SECTION 5: SPOKEN SERIAL, PAIRED & YEAR CONCATENATION
# ======================================================================
print("\n=== Spoken Paired / Year / Serial Number Concatenation ===")
check("twenty thirty", 2030, "(spoken year: twenty thirty -> 2030, NOT 20+30=50)")
check("twenty twenty", 2020, "(spoken year: twenty twenty -> 2020, NOT 20+20=40)")
check("twenty twenty four", 2024, "(spoken year: twenty twenty four -> 2024)")
check("nineteen ninety five", 1995, "(spoken year: nineteen ninety five -> 1995)")
check("one two three", 123, "(spoken serial digits: 1 2 3 -> 123, NOT 1+2+3=6)")
check("one two three four", 1234, "(spoken serial digits: 1 2 3 4 -> 1234)")
check("four zero two", 402, "(spoken room/flight number: 4 0 2 -> 402)")
check("seven zero seven", 707, "(spoken serial number: 7 0 7 -> 707)")


# ======================================================================
# SECTION 6: FORMATTING, ATTACHED SYMBOLS & DECIMALS
# ======================================================================
print("\n=== Attached Punctuation, Currencies & Formatting ===")
check("five hundred.", 500)
check("five hundred?", 500)
check('"five hundred"', 500)
check("(one hundred)", 100)
check("100!", 100)
check("500/-", 500)
check("₹500", 500)
check("$500", 500)
check("Rs. 500", 500)
check("Rs.500", 500)
check("10lakh", 1000000)
check("2.5lakh", 250000)
check("50k", 50000)
check("100k", 100000)
check("5kg", 5)
check("500gm", 500)
check("1st", 1)
check("2nd", 2)
check("3rd", 3)
check("4th", 4)
check("2.5 lakh", 250000)
check("1.5 crore", 15000000)
check("two point five lakh", 250000)
check("three point five crore", 35000000)
check("500.50", Decimal("500.50"))
check("minus five", -5)
check_extract("25 5", [25, 5], "(adjacent digit runs separated)")
check_extract("two or three", [2, 3])
check_extract("one lakh or two lakh", [100000, 200000])
check_extract("five to six", [5, 6])
check_extract("10 to 15", [10, 15])
check_extract("give me 2 kg potatoes for 50 rupees", [2, 50])


# ======================================================================
# SECTION 7: ALPHANUMERIC TRAPS & AMBIGUITY PROTECTION
# ======================================================================
print("\n=== Alphanumeric Traps & Ambiguity Protection ===")
check_ambiguous("covid19", "(medical alphanumeric code must NOT become 19)")
check_ambiguous("mp3", "(file format code must NOT become 3)")
check_ambiguous("h2o", "(chemical formula must NOT become 2)")
check_ambiguous("f16", "(aircraft model must NOT become 16)")
check_ambiguous("teroso", "(roman teroso must NOT become 1800 or 1300)")
check_ambiguous("tero", "(roman tero must NOT become 0)")
check_ambiguous("hundred hundred", "(duplicate magnitude)")
check_ambiguous("one lakh lakh", "(duplicate magnitude)")
check_ambiguous("one thousand thousand", "(duplicate magnitude)")
check_ambiguous("one thousand one lakh", "(invalid magnitude order)")


# ======================================================================
# SECTION 8: REGIONAL ASR, REPEATERS, FRACTION PREFIXES & INFLECTIONS
# ======================================================================
print("\n=== Section 8: Regional ASR, Repeaters, Fractions & Inflections ===")
# Exact user test cases
check("সাড়ে সতেরোশ", 1750, "(sade [17 + 0.5] * 100 = 1750)")
check("জিরো ডাবল এইট", "088", "(0 + double 8 -> 088 preserving leading zero)")
check("ওয়ান সেভেন", 17, "(spoken serial digits: 1 7 -> 17)")
check("ট্রিপল ফোর", 444, "(triple 4 -> 444)")
check("পনে দুশো", 175, "(paune 200 -> [2 - 0.25] * 100 = 175)")
check("আড়াইশোর", 250, "(250 with Bengali genitive inflection -র)")
check("সাড়ে তিনশ", 350, "(sade 300 -> [3 + 0.5] * 100 = 350)")

# Additional regional variations
check("পৌনে দুশো", 175)
check("সোয়া দুশো", 225)
check("আড়াইশো", 250)
check("দেড়শো", 150)
check("ডাবল ফাইভ", 55)
check("ট্রিপল নাইন", 999)
check("ট্রিপল নাইন 52", 99952, "(mixed word + digit serial run: triple 9 + 52 -> 99952)")
check("ট্রিপল নাইন ৫২", 99952, "(mixed word + Bengali digit serial run)")
check("ডাবল জিরো সেভেন", "007", "(double 0 + 7 -> 007)")
check("088", "088", "(raw digits with leading zero preserved)")
check("007", "007", "(raw digits with leading zero preserved)")
check("জিরো ওয়ান সেভেন", "017", "(0 + 1 + 7 -> 017)")
check("সাড়ে পাঁচ হাজার", 5500)
check("পৌনে পাঁচ হাজার", 4750)
check("সওয়া পাঁচ হাজার", 5250)
check("পৌনে এক লাখ", 75000)
check("দেড় লাখ", 150000)
check("আড়াই লাখ", 250000)
check("সাড়ে বারোশো", 1250)
check("পৌনে বারোশো", 1175)
check("হাজারের", 1000, "(Bengali genitive inflection: হাজার + এর)")
check("পাঁচশোতে", 500, "(Bengali locative inflection: পাঁচশো + তে)")
check("একশোকে", 100, "(Bengali accusative inflection: একশো + কে)")

# Hindi equivalents
check("साढ़े सत्रह सौ", 1750, "(Hindi: sade 17 sau = 1750)")
check("पौने दो सौ", 175, "(Hindi: paune 2 sau = 175)")
check("सवा दो सौ", 225, "(Hindi: sawa 2 sau = 225)")
check("ढाई सौ", 250, "(Hindi: 250)")
check("डेढ़ सौ", 150, "(Hindi: 150)")
check("डबल आठ", 88, "(Hindi: double 8 = 88)")
check("ट्रिपल चार", 444, "(Hindi: triple 4 = 444)")
check("ज़ीरो डबल आठ", "088", "(Hindi: zero double 8 = 088)")
check("वन सेवन", 17, "(Hindi phonetic English: 1 7 = 17)")


# ======================================================================
# SUMMARY REPORT
# ======================================================================
print(f"\n\n{'='*50}")
print(f"TEST SUITE COMPLETE: {passed} passed, {failed} failed")
print(f"{'='*50}")

if __name__ == '__main__':
    if '--interactive' in sys.argv:
        print("\nEntering interactive mode. Type 'exit' or Ctrl+C to quit.")
        while True:
            try:
                line = input("Enter text: ").strip()
                if not line or line.lower() == 'exit':
                    break
                print(f"text_to_number: {text_to_number(line)}")
                print(f"extract_numbers: {[(r.text, r.value, r.ambiguous, r.reason) for r in extract_numbers(line)]}")
            except (KeyboardInterrupt, EOFError):
                break
    else:
        sys.exit(0 if failed == 0 else 1)