from collections import defaultdict
from typing import List, Dict, Tuple

from src.generate_real_word_cvc.CMUreader import *
from src.generate_real_word_cvc.TestVocabMaker import *


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

    cat1 = set(category1_vowels)
    cat2 = set(category2_vowels)

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
        have_cat1 = cat1.issubset(vmap.keys())          # *all* Cat 1 vowels present?
        have_cat2 = cat2.issubset(vmap.keys())          # *all* Cat 2 vowels present?

        if have_cat1 and not have_cat2:                 # ------- Condition 1 -------
            case1[(c1, c2)] = [vmap[v] for v in cat1]   # keep only the Cat 1 members

        elif have_cat2 and not have_cat1:               # ------- Condition 2 -------
            case2[(c1, c2)] = [vmap[v] for v in cat2]

        elif have_cat1 and have_cat2:                   # ------- Condition 3 -------
            wanted = cat1 | cat2
            case3[(c1, c2)] = [vmap[v] for v in wanted]

        # If neither condition matches (mixed / incomplete), ignore.

    return {"case1": case1, "case2": case2, "case3": case3}


import itertools
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple

# ---------------------------------------------------------------------
#  Helper: build the (C1,C2) → {vowel: word_record} index
# ---------------------------------------------------------------------
def _index_c1vc2(cvc_word_list):
    combos = defaultdict(dict)

    def _base(v):          # strip stress digits
        return ''.join(ch for ch in v if not ch.isdigit())

    for entry in cvc_word_list:
        pron = entry["pronunciation"]
        if len(pron) != 3:
            continue
        c1, vowel, c2 = pron
        combos[(c1, c2)][_base(vowel)] = entry   # last duplicate wins
    return combos


# ---------------------------------------------------------------------
#  Main routine (sorted & relabelled)
# ---------------------------------------------------------------------
def summarize_c1vc2_conditions(
        cvc_word_list: List[Dict[str, List[str]]],
        category1_vowels: List[str],
        category2_vowels: List[str],
        min_inner_size: int = 2,
        tsv_path: str = "c1vc2_summary.tsv"
    ) -> pd.DataFrame:

    combos = _index_c1vc2(cvc_word_list)

    cat1 = set(category1_vowels)
    all_inner_subsets = [
        set(s)
        for r in range(min_inner_size, len(category2_vowels) + 1)
        for s in itertools.combinations(category2_vowels, r)
    ]

    # mapping numeric → label and an ordering key so we can sort “Case”
    CASE_LABEL = {1: "RW=>FW", 2: "FW=>RW", 3: "RW=>RW"}
    CASE_ORDER = {"RW=>FW": 0, "FW=>RW": 1, "RW=>RW": 2}

    rows = []

    for (c1, c2), vmap in combos.items():
        have = set(vmap.keys())

        outer_words = {v: vmap[v]["word"] for v in cat1 if v in vmap}
        have_all_outer = cat1.issubset(have)
        have_any_outer = bool(cat1 & have)

        for inner in all_inner_subsets:
            inner_words = {v: vmap[v]["word"] for v in inner if v in vmap}
            have_all_inner = inner.issubset(have)
            have_any_inner = bool(inner & have)

            # decide case
            case_num = None
            if have_all_outer and not have_any_inner:         # RW=>FW
                case_num = 1
            elif have_all_inner and not have_any_outer:       # FW=>RW
                case_num = 2
            elif have_all_outer and have_all_inner:           # RW=>RW
                case_num = 3
            if case_num is None:
                continue

            case_label = CASE_LABEL[case_num]

            rows.append({
                "C1": c1,
                "C2": c2,
                "Case": case_label,
                "OuterLen": len(outer_words),
                "InnerLen": len(inner_words),
                "OuterVowels": ",".join(sorted(cat1)),
                "InnerVowels": ",".join(sorted(inner)),
                "OuterWords": ",".join(outer_words.get(v, "") for v in cat1),
                "InnerWords": ",".join(inner_words.get(v, "") for v in inner)
            })

    df = pd.DataFrame(rows, columns=[
        "C1", "C2", "Case", "OuterLen", "InnerLen",
        "OuterVowels", "InnerVowels", "OuterWords", "InnerWords"
    ])

    # sort: C1 → C2 → Case → OuterLen → InnerLen
    df = df.sort_values(
        by=["C1", "C2", "Case", "OuterLen", "InnerLen"],
        key=lambda col: col.map(CASE_ORDER) if col.name == "Case" else col
    )

    df.to_csv(tsv_path, sep="\t", index=False)
    return df



if __name__ == "__main__":
    dictLocation = "../../input_data/"

    # ---------- Build the master CVC set ----------
    original_word_set = generateCombinedWordsets(dictLocation)
    unique_l2_words  = load_unique_words(
        dictLocation + "dictionaries/Oxford_3000_5000_AmericanEnglish.txt"
    )
    filtered_word_set = filter_cmudict_words(original_word_set, unique_l2_words)
    result_sets       = generateTestWordList(filtered_word_set)
    voiced_cvc        = result_sets["Voiced CVC"]

    # ---------- Define (Cat-1, Cat-2) jobs ----------
    JOBS = [
        ("IH_AE_vs_EY_EH_OW", ["IH", "AE"], ["EY", "EH", "OW"]),
        ("EH_UW_vs_OW_OY_UH", ["EH", "UW"], ["OW", "OY", "UH"]),
        ("AA_UW_vs_AH_UH",    ["AA", "UW"], ["AH", "UH"]),
        ("IY_AE_vs_IH_EY_EH_OW", ["IY", "AE"], ["IH", "EY", "EH", "OW"]),
    ]

    all_frames = []    # collect dataframes for an optional mega-table

    for label, cat1, cat2 in JOBS:
        out_file = f"{label}.tsv"
        print(f"→ Building {out_file} …")

        df = summarize_c1vc2_conditions(
            voiced_cvc,
            category1_vowels = cat1,
            category2_vowels = cat2,
            min_inner_size   = 2,
            tsv_path         = out_file
        )
        df["JobLabel"] = label           # keep provenance if we merge later
        all_frames.append(df)

    # ---------- One combined TSV (optional) ----------
    mega = (
        pd.concat(all_frames, ignore_index=True)
          .sort_values(["JobLabel", "C1", "C2", "Case", "OuterLen", "InnerLen"])
    )
    mega.to_csv("ALL_CVC_jobs.tsv", sep="\t", index=False)

    print("All done!  Per-job TSVs plus ALL_CVC_jobs.tsv have been written.")