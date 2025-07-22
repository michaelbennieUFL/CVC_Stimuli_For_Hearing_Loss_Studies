import csv
import math
from typing import Tuple

from src.generate_real_word_cvc.CMUreader import *
from src.speechGeneration.TestVocabMaker import *

import requests, random, functools, time

# --- ❶  Simple in-memory cache --------------------------------------
@functools.lru_cache(maxsize=10_000)
def _query_ngram_api(token: str) -> float:
    """
    Return api.ngrams.dev relative frequency or raise on network/JSON errors.
    """
    url     = "https://api.ngrams.dev/eng/search"
    headers = {
        "User-Agent": "cvc-script/1.0 (+https://yourlab.org)",
        "Accept": "application/json",
        "Referer": "https://ngrams.dev/",
    }
    try:
        r = requests.get(url, params={"query": token, "flags": "cs"},
                         headers=headers, timeout=5)
        r.raise_for_status()
        data = r.json()
        return float(data["ngrams"][0]["relTotalMatchCount"])
    except Exception as e:
        raise RuntimeError(f"ngram API failure for '{token}': {e}") from e


import spacy

nlp = spacy.load("en_core_web_sm")  # or 'en_core_web_md' if installed

def get_lemma_spacy(word: str) -> str:
    doc = nlp(word)
    return doc[0].lemma_


# --- Load COCA Spoken Relative Frequencies --------------------------
N_SPOKEN = 81_916_566
FREQ_FLOOR = 1 / N_SPOKEN  # ≈ 1.221e-8

def load_spoken_freq_table(path: str = "./wordlist/coca_spoken_rank.tsv") -> dict:
    freq = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            freq[row["Word"].upper()] = float(row["Spoken_Relative_Frequency"])
    return freq

COCA_SPOKEN_FREQ = load_spoken_freq_table()

def get_spoken_freq_log10(word: str) -> float:
    """
    Return log10 of COCA Spoken Relative Frequency (or log10 of fallback if fake word).
    """
    f = COCA_SPOKEN_FREQ.get(word.upper(), FREQ_FLOOR)
    if f==FREQ_FLOOR:
        word =get_lemma_spacy(word)
        f = COCA_SPOKEN_FREQ.get(word.upper(), FREQ_FLOOR)
    return math.log10(f/(FREQ_FLOOR))


# --- ❷  Public wrapper with fallback --------------------------------
def lookup_ngram_freq(word: str,

                      retries: int = 3,
                      backoff: float = 0.5) -> float:
    """
    • Returns the API frequency for *real* tokens.
    • If the API returns 0.0 **or** raises → assign a pseudo-frequency default to 1e-9

    Caches successful look-ups via `functools.lru_cache`.
    """
    token = word.upper()          # api.ngrams.dev is case-sensitive

    for attempt in range(retries):
        try:
            freq = _query_ngram_api(token)
            if freq > 0.0:
                return freq
            break                      # 0.0 → treat as fake
        except RuntimeError as err:
            if attempt == retries - 1:
                print(err)
            time.sleep(backoff * (attempt + 1))

    # Fallback = “fake” word
    return 10**-10


def find_c1vc2_cases(
        cvc_word_list: List[Dict[str, List[str]]],
        category1_vowels: List[str],
        category2_vowels: List[str]
    ) -> Dict[str, Dict[Tuple[str, str], List[Dict[str, List[str]]]]]:
    """
    Given the list returned by result_sets["Voiced CVC"] and two
    vowel-category lists, return every (C1, C2) pair that satisfies
    one of the three conditions described in the prompt.

    Parameters
    ----------
    cvc_word_list
        The list produced by generateTestWordList()["Voiced CVC"].
        Each element is a dict like
        {'word': 'BAD', 'pronunciation': ['B','AE1','D']}.
    category1_vowels
        Vowel symbols (no stress digits) that make up Category 1
        – e.g. ["IH", "AE"].
    category2_vowels
        Vowel symbols (no stress digits) that make up Category 2
        – e.g. ["EY", "EH", "OW"].

    Returns
    -------
    dict with keys "case1", "case2", "case3".
    Each value is a dict that maps a (C1, C2) tuple to the list of
    *word records* (the same dicts you passed in) that belong to the
    winning category for that (C1, C2) pair.
    """

    # Helper: strip any stress digits from a vowel symbol
    def _base_vowel(v: str) -> str:
        return ''.join(ch for ch in v if not ch.isdigit())

    Outer_Category_ = set(category1_vowels)
    Inner_Category_ = set(category2_vowels)

    # ------------------------------------------------------------------
    # 1) Collect all words by (C1, C2) and the vowel that sits between.
    # ------------------------------------------------------------------
    combos: Dict[Tuple[str, str], Dict[str, Dict[str, List[str]]]] = defaultdict(dict)

    for entry in cvc_word_list:
        pron = entry["pronunciation"]
        if len(pron) != 3:        # safety: ignore non-CVC items
            continue
        c1, vowel, c2 = pron
        vowel_base = _base_vowel(vowel)
        combos[(c1, c2)][vowel_base] = entry   # overwrite is fine – duplicate words share the same (C1,V,C2)

    # ------------------------------------------------------------------
    # 2) Decide which (C1, C2) pairs land in which case.
    # ------------------------------------------------------------------


    #1. all the c1vc2 words from Category 1 exist but all the c1vc2 words from Category 2 dont exist
    # 2. all the c1vc2 words from Category 2 exist but all the c1vc2 words from Category 1 dont exist
    # 3. all the c1vc2 words from Category 1 exist and all the c1vc2 words from Category 2  exist
    case1, case2, case3 = {}, {}, {}

    for (c1, c2), vmap in combos.items():
        have_Outer_Category_ = Outer_Category_.issubset(vmap.keys())          # *all* Cat 1 vowels present?
        have_Inner_Category_ = Inner_Category_.issubset(vmap.keys())          # *all* Cat 2 vowels present?

        if have_Outer_Category_ and not have_Inner_Category_:                 # ------- Condition 1 -------
            case1[(c1, c2)] = [vmap[v] for v in Outer_Category_]   # keep only the Cat 1 members

        elif have_Inner_Category_ and not have_Outer_Category_:               # ------- Condition 2 -------
            case2[(c1, c2)] = [vmap[v] for v in Inner_Category_]

        elif have_Outer_Category_ and have_Inner_Category_:                   # ------- Condition 3 -------
            wanted = Outer_Category_ | Inner_Category_
            case3[(c1, c2)] = [vmap[v] for v in wanted]

        # If neither condition matches (mixed / incomplete), ignore.

    return {"case1": case1, "case2": case2, "case3": case3}


import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
#  Helper: build (C1,C2) → {vowel: word_record}
# ---------------------------------------------------------------------
def _index_c1vc2(cvc_word_list: List[Dict[str, List[str]]]) -> Dict[Tuple[str, str], Dict[str, List[Dict]]]:
    """
    Returns (C1, C2) → {vowel → [word_record1, word_record2, ...]}
    Keeps all matching words rather than overwriting.
    """
    combos = defaultdict(lambda: defaultdict(list))
    strip = lambda v: ''.join(ch for ch in v if not ch.isdigit())

    for entry in cvc_word_list:
        pron = entry.get("pronunciation")
        if not pron or len(pron) != 3:
            continue
        c1, v, c2 = pron
        base_v = strip(v)
        combos[(c1, c2)][base_v].append(entry)

    return combos

def max_word_and_freq_log10(word_records):
    if not word_records:
        return None, 0.0
    best = max(word_records, key=lambda w: get_spoken_freq_log10(w["word"]))
    return best["word"], max(get_spoken_freq_log10(best["word"]),0)



def summarize_c1vc2_input_output_cases(
        cvc_word_list: List[Dict[str, List[str]]],
        category1_vowels: List[str],          # outer / “input” vowels
        category2_vowels: List[str],          # inner / “output” vowels
        tsv_path: str = "c1vc2_summary.tsv"
    ) -> pd.DataFrame:

    combos           = _index_c1vc2(cvc_word_list)
    Outer_Category_  = [v.upper() for v in category1_vowels]   # keep order!
    Inner_Category_  = set(v.upper() for v in category2_vowels)

    INPUT_ORDER   = ["RW_RW", "RW_FW", "FW_FW"]
    OUTPUT_ORDER  = ["All_Real", "Mixed", "All_Fake"]

    rows = []
    for (c1, c2), vmap in combos.items():
        have = set(vmap.keys())



        # -----  New frequency columns --------------------------------
        #  Outer_Left_Freq  = freq of first outer vowel (if present)
        #  Outer_Right_Freq = freq of second outer vowel (if present)
        outer_left_freq  = 0
        outer_right_freq = 0
        outer_Category_Words=["",""]
        if len(Outer_Category_) >= 1 and Outer_Category_[0] in vmap:
            outer_Category_Words[0],outer_left_freq = max_word_and_freq_log10(vmap[Outer_Category_[0]])
        if len(Outer_Category_) >= 2 and Outer_Category_[1] in vmap:
            outer_Category_Words[1],outer_right_freq = max_word_and_freq_log10(vmap[Outer_Category_[1]])


        #  Middle_Mean_Freq = mean freq of ALL present inner-category words
        present_inner = Inner_Category_ & have
        inner_freqs = []
        inner_words = []

        for v in present_inner:
            word, freq = max_word_and_freq_log10(vmap[v])
            inner_words.append(word)
            inner_freqs.append(freq)

        middle_mean_freq = sum(inner_freqs) / len(inner_freqs) if inner_freqs else 0.0

        original_length=len(Outer_Category_)
        if outer_left_freq<0.01 and original_length>= 1 :
            have.discard(Outer_Category_[0])
            outer_Category_Words[0] = ""
        if outer_right_freq<0.01 and original_length>= 2 :
            have.discard(Outer_Category_[1])
            outer_Category_Words[1]=""
        if middle_mean_freq<0.01:
            have.difference_update(Inner_Category_)
            inner_words=[]

        if "GOOD" in outer_Category_Words:
            print("!!!!!!!!!!!!!")
        # -----  INPUT-side case  -----
        present_outer = [v for v in Outer_Category_ if v in have]
        if len(present_outer) == len(Outer_Category_):
            input_case = "RW_RW"
        elif len(present_outer) == 0:
            input_case = "FW_FW"
        else:
            input_case = "RW_FW"

        # -----  OUTPUT-side case -----
        present_inner = Inner_Category_ & have
        if len(present_inner) == len(Inner_Category_):
            output_case = "All_Real"
        elif len(present_inner) == 0:
            output_case = "All_Fake"
        else:
            output_case = "Mixed"




        rows.append({
            "C1": c1,
            "C2": c2,
            "Input_Case":  input_case,
            "Output_Case": output_case,

            # new columns
            "Outer_Left_Freq":  outer_left_freq,
            "Outer_Right_Freq": outer_right_freq,
            "Middle_Mean_Freq": middle_mean_freq,

            # keep the old metadata
            "OuterLen": len(present_outer),
            "InnerLen": len(present_inner),
            "Outer_Category_Vowel":  ",".join(Outer_Category_),
            "Inner_Category_Vowel":  ",".join(sorted(Inner_Category_)),
            "Outer_Category_Present":",".join(sorted(present_outer)),
            "Inner_Category_Present":",".join(sorted(present_inner)),
            "Outer_Category_Words":  ",".join(v for v in outer_Category_Words),
            "Inner_Category_Words":  ",".join(inner_words),
        })

    # ----------  sort & export as before ----------
    df = pd.DataFrame(rows)
    df["Input_Case"]  = pd.Categorical(df["Input_Case"],  categories=INPUT_ORDER,  ordered=True)
    df["Output_Case"] = pd.Categorical(df["Output_Case"], categories=OUTPUT_ORDER, ordered=True)

    sort_cols = ["Input_Case", "Output_Case", "Outer_Category_Vowel",
                 "OuterLen", "InnerLen", "C1", "C2"]
    df = df.sort_values(sort_cols, ignore_index=True)
    df.to_csv(tsv_path, sep="\t", index=False)
    return df






import collections


def find_superset_words(word_list, target_word):
    """
    Finds all words in a list that are a character-superset of a target word.

    The comparison is case-insensitive and respects character counts. For example,
    'DOUBLE-QUOTE' is a superset of 'quote', but not of 'bubble'.

    Args:
        word_list (list): A list of strings to search through.
        target_word (str): The word whose characters must be contained in the results.

    Returns:
        list: A list of words from word_list that are supersets of the target_word.
    """
    # Create a frequency count of characters for the lowercase target word.
    # This is done once to be efficient.
    target_counts = collections.Counter(target_word.lower())

    superset_matches = []

    for word in word_list:
        # Optimization: a word can't be a superset if it's shorter than the target.
        if len(word) < len(target_word):
            continue

        # Create a frequency count for the current word from the list.
        word_counts = collections.Counter(word.lower())

        # Check if the word contains all characters from the target with sufficient counts.
        # The all() function ensures every character condition is met.
        if all(word_counts[char] >= count for char, count in target_counts.items()):
            superset_matches.append(word)

    return superset_matches

def find_words_with_substring(word_set, target_substring):
    """
    Filters entries in a CMUdict-style word set where the 'word' field
    contains the target substring (case-insensitive).

    Args:
        word_set (dict): Dictionary where each value is a dict with a 'word' key.
        target_substring (str): The substring to search for (case-insensitive).

    Returns:
        list: List of entries (dicts) matching the condition.
    """
    target = target_substring.lower()

    result=[]


    for entry in word_set:
        if target in entry['word'].lower():
            result.append(entry)

    return result


if __name__ == "__main__":
    dictLocation = "../../input_data/"

    # ---------- Build the master CVC set ----------
    original_word_set = generateCombinedWordsets(dictLocation, stress_sensitive=False)

    unique_l2_words = load_unique_words(
        dictLocation + "dictionaries/Oxford_3000_5000_AmericanEnglish.txt"
    )
    filtered_word_set = filter_cmudict_words(original_word_set, unique_l2_words)

    result_sets = generateTestWordList(original_word_set)
    voiced_cvc  = result_sets["Voiced CVC"]

    # quick test
    print("results:", find_words_with_substring(voiced_cvc, "bat"))
    print("results:", find_words_with_substring(voiced_cvc, "bhatt"))
    # ---------- Define (Cat-1, Cat-2) jobs ----------
    MONOTHONG_JOBS = [
        ("Job1_Inner_Category__AA_AH_UH", ["IH", "AE"], ["EH"])
    ]

    all_frames = []

    for label, Outer_Category_, Inner_Category_ in MONOTHONG_JOBS:
        out_file = f"wordlist/{label}.tsv"
        print(f"→ Building {out_file} …")

        df = summarize_c1vc2_input_output_cases(          # ← now returns freq cols
            voiced_cvc,
            category1_vowels=Outer_Category_,
            category2_vowels=Inner_Category_,
            tsv_path=out_file
        )
        df["JobLabel"] = label
        all_frames.append(df)

    # ---------- One combined TSV ----------
    print("→ Building combined summary …")

    mega = pd.concat(all_frames, ignore_index=True)

    # category ordering (unchanged)
    INPUT_ORDER  = ["RW_RW", "RW_FW", "FW_FW"]
    OUTPUT_ORDER = ["All_Real", "Mixed", "All_Fake"]

    mega["Input_Case"]  = pd.Categorical(mega["Input_Case"],  INPUT_ORDER,  ordered=True)
    mega["Output_Case"] = pd.Categorical(mega["Output_Case"], OUTPUT_ORDER, ordered=True)

    mega = mega.sort_values(
        ["Input_Case", "Output_Case", "Outer_Category_Vowel",
         "OuterLen", "InnerLen", "C1", "C2"]
    )

    # ---------- Desired column order  🔄  (new freq cols added) ----------
    desired_order = [
        "Input_Case", "Output_Case",
        "OuterLen", "InnerLen", "C1", "C2",
        "Outer_Category_Vowel", "Inner_Category_Vowel",
        "Outer_Category_Present", "Inner_Category_Present",
        "Outer_Category_Words", "Inner_Category_Words",
        "Outer_Left_Freq", "Outer_Right_Freq", "Middle_Mean_Freq",
        "JobLabel",

    ]

    mega = mega[desired_order]
    mega.to_csv("wordlist/ALL_CVC_jobs.tsv", sep="\t", index=False)

    print("All done! Per-job TSVs plus ALL_CVC_jobs.tsv have been written.")
