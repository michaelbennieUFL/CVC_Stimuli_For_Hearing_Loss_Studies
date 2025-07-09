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


import itertools
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
#  Helper: build (C1,C2) → {vowel: word_record}
# ---------------------------------------------------------------------
def _index_c1vc2(cvc_word_list):
    combos = defaultdict(dict)
    strip = lambda v: ''.join(ch for ch in v if not ch.isdigit())

    for entry in cvc_word_list:
        pron = entry["pronunciation"]
        if len(pron) != 3:
            continue
        c1, v, c2 = pron
        combos[(c1, c2)][strip(v)] = entry       # last duplicate wins
    return combos


# ---------------------------------------------------------------------
#  New summariser (single row per C1–C2 with maximal coverage)
# ---------------------------------------------------------------------
def summarize_c1vc2_conditions(
        cvc_word_list: List[Dict[str, List[str]]],
        category1_vowels: List[str],
        category2_vowels: List[str],
        min_Inner_Category__real: int = 2,           # ← NEW
        tsv_path: str = "c1vc2_summary.tsv"
    ) -> pd.DataFrame:

    combos = _index_c1vc2(cvc_word_list)
    Outer_Category_ = set(category1_vowels)
    Inner_Category_ = set(category2_vowels)

    rows = []
    for (c1, c2), vmap in combos.items():
        have = set(vmap.keys())

        # keep only if at least N Cat-2 vowels present
        present_Inner_Category_ = Inner_Category_ & have
        if len(present_Inner_Category_) < min_Inner_Category__real:
            continue

        # determine Cat-1 coverage
        present_Outer_Category_ = Outer_Category_ & have
        if len(present_Outer_Category_) == len(Outer_Category_):
            case = "RW_RW"
        elif len(present_Outer_Category_) == 0:
            case = "FW_FW"
        else:
            case = "RW_FW"

        rows.append({
            "C1": c1,
            "C2": c2,
            "Case": case,
            "OuterLen": len(present_Outer_Category_),
            "InnerLen": len(present_Inner_Category_),
            "Outer_Category_Vowel": ",".join(sorted(Outer_Category_)),                    # ← ALL Category 1 vowels
            "Inner_Category_Vowel": ",".join(sorted(Inner_Category_)),                    # ← ALL Category 2 vowels
            "Outer_Category_Present": ",".join(sorted(present_Outer_Category_)),
            "Inner_Category_Present": ",".join(sorted(present_Inner_Category_)),
            "Outer_Category_Words": ",".join(vmap[v]["word"] for v in sorted(present_Outer_Category_)),
            "Inner_Category_Words": ",".join(vmap[v]["word"] for v in sorted(present_Inner_Category_)),
        })



    df = (pd.DataFrame(rows)
            .sort_values(["C1", "C2", "Case"], ignore_index=True))
    df.to_csv(tsv_path, sep="\t", index=False)
    return df



def summarize_c1vc2_input_output_cases(
        cvc_word_list: List[Dict[str, List[str]]],
        category1_vowels: List[str],          # “outer” – source/input vowels
        category2_vowels: List[str],          # “inner” – target/output vowels
        tsv_path: str = "c1vc2_summary.tsv"
    ) -> pd.DataFrame:
    """
    Produce one row per (C1, C2) pair, labelling both the input-side
    real/fake pattern (outer vowels) and the output-side pattern
    (inner vowels).  No minimum-coverage filters are applied.
    """
    combos           = _index_c1vc2(cvc_word_list)
    Outer_Category_  = set(category1_vowels)
    Inner_Category_  = set(category2_vowels)

    INPUT_ORDER   = ["RW_RW", "RW_FW", "FW_FW"]
    OUTPUT_ORDER  = ["All_Real", "Mixed", "All_Fake"]

    rows = []
    for (c1, c2), vmap in combos.items():
        have = set(vmap.keys())

        # ---------- INPUT side (outer) ----------
        present_outer = Outer_Category_ & have
        if len(present_outer) == len(Outer_Category_):
            input_case = "RW_RW"
        elif len(present_outer) == 0:
            input_case = "FW_FW"
        else:
            input_case = "RW_FW"

        # ---------- OUTPUT side (inner) ----------
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
            "OuterLen": len(present_outer),
            "InnerLen": len(present_inner),
            "Outer_Category_Vowel":  ",".join(sorted(Outer_Category_)),
            "Inner_Category_Vowel":  ",".join(sorted(Inner_Category_)),
            "Outer_Category_Present":",".join(sorted(present_outer)),
            "Inner_Category_Present":",".join(sorted(present_inner)),
            "Outer_Category_Words":  ",".join(vmap[v]["word"] for v in sorted(present_outer)),
            "Inner_Category_Words":  ",".join(vmap[v]["word"] for v in sorted(present_inner)),
        })

    df = pd.DataFrame(rows)

    # ----------  custom sort order ----------
    df["Input_Case"]  = pd.Categorical(df["Input_Case"],  categories=INPUT_ORDER,  ordered=True)
    df["Output_Case"] = pd.Categorical(df["Output_Case"], categories=OUTPUT_ORDER, ordered=True)

    sort_cols = ["Input_Case", "Output_Case", "Outer_Category_Vowel", "OuterLen", "InnerLen", "C1", "C2"]
    df = df.sort_values(sort_cols, ignore_index=True)

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
    # JOBS = [
    #     ("Job1_Inner_Category__AA_AH_UH", ["UH", "AE"], ["AA", "AH", "EH"]),
    #     ("Job2_Inner_Category__AE_AH_UH", ["EH", "AA"], ["AE", "AH"]),
    #     ("Job3_Inner_Category__AH_UH", ["AA", "UW"], ["AH", "UH"]),
    # ]



    MONOTHONG_JOBS = [
        ("Job1_Inner_Category__AA_AH_UH", ["IH", "AE"], ["EH"]),
        ("Job2_Inner_Category__AE_AH_UH", ["EH", "UW"], ["UH"]),
    ]

    all_frames = []    # collect dataframes for an optional mega-table

    for label, Outer_Category_, Inner_Category_ in MONOTHONG_JOBS:
        out_file = f"wordlist/{label}.tsv"
        print(f"→ Building {out_file} …")

        df = summarize_c1vc2_input_output_cases(
            voiced_cvc,
            category1_vowels=Outer_Category_,
            category2_vowels=Inner_Category_,
            tsv_path=out_file
        )
        df["JobLabel"] = label           # keep provenance if we merge later
        all_frames.append(df)

    # ---------- One combined TSV (optional) ----------
    print("→ Building combined summary …")

    mega = pd.concat(all_frames, ignore_index=True)

    # Ensure consistent category ordering for sorting
    INPUT_ORDER  = ["RW_RW", "RW_FW", "FW_FW"]
    OUTPUT_ORDER = ["All_Real", "Mixed", "All_Fake"]

    mega["Input_Case"]  = pd.Categorical(mega["Input_Case"], categories=INPUT_ORDER, ordered=True)
    mega["Output_Case"] = pd.Categorical(mega["Output_Case"], categories=OUTPUT_ORDER, ordered=True)

    mega = mega.sort_values([
        "Input_Case", "Output_Case", "Outer_Category_Vowel",
        "OuterLen", "InnerLen", "C1", "C2"
    ])

    desired_order = [
        "Input_Case", "Output_Case", "OuterLen", "InnerLen", "C1", "C2",
        "Outer_Category_Vowel", "Inner_Category_Vowel",
        "Outer_Category_Present", "Inner_Category_Present",
        "Outer_Category_Words", "Inner_Category_Words",
        "JobLabel"
    ]

    mega = mega[desired_order]

    mega.to_csv("wordlist/ALL_CVC_jobs.tsv", sep="\t", index=False)

    print("All done! Per-job TSVs plus ALL_CVC_jobs.tsv have been written.")
