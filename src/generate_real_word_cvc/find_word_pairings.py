#!/usr/bin/env python3
"""
Summarise (Input_Case, Output_Case) patterns by starting consonant C1
and also choose the single *best* C2 for each required combination
(max-average spoken-frequency row).  “All Fake” remains a list.

Pattern-A: { ("RW_RW","All_Fake"), ("FW_FW","All_Real"), ("FW_FW","All_Fake") }
Pattern-B: { ("RW_RW","All_Fake"), ("FW_FW","All_Real"), ("RW_RW","All_Real") }
Pattern-C: Pattern-A ∪ Pattern-B
"""

from __future__ import annotations
import argparse, pathlib, sys
import pandas as pd
from typing import Dict, List, Tuple, Set

# ------------------------------------------------------------------ #
# I/O helpers                                                        #
# ------------------------------------------------------------------ #
def load_job_tsv(path: str | pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    need = {"C1", "C2", "Input_Case", "Output_Case"} - set(df.columns)
    if need:
        raise ValueError(f"TSV missing columns: {', '.join(need)}")
    return df

def group_by_c1(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {c1: sub for c1, sub in df.groupby("C1", sort=False)}

# ------------------------------------------------------------------ #
# pattern definitions                                                #
# ------------------------------------------------------------------ #
PATTERNS: Dict[str, List[Tuple[str, str]]] = {
    "A": [("RW_RW", "All_Fake"),
          ("FW_FW", "All_Real"),
          ("FW_FW", "All_Fake")],

    "B": [("RW_RW", "All_Fake"),
          ("FW_FW", "All_Real"),
          ("RW_RW", "All_Real")],

    "C": [("RW_RW", "All_Fake"),
          ("FW_FW", "All_Real"),
          ("RW_RW", "All_Real"),
          ("FW_FW", "All_Fake")],
}

COLUMN_NAMES = {
    ("RW_RW", "All_Fake"):  "Real-Out Fake-In",
    ("FW_FW", "All_Real"):  "Fake-Out Real-In",
    ("FW_FW", "All_Fake"):  "All Fake",
    ("RW_RW", "All_Real"):  "Real-Out Real-In",
}

# Which combos should be *optimised* to a single best C2?
NEEDS_BEST = {
    ("RW_RW", "All_Fake"),
    ("FW_FW", "All_Real"),
    ("RW_RW", "All_Real"),
}

# ------------------------------------------------------------------ #
# scoring helper                                                     #
# ------------------------------------------------------------------ #
def row_score(row: pd.Series, numeric_cols: List[str]) -> float:
    """Mean of available numeric columns; zero if none present."""
    vals = [float(row[c]) for c in numeric_cols if c in row and pd.notna(row[c])]
    return sum(vals) / len(vals) if vals else 0.0

# ------------------------------------------------------------------ #
# summarisation helpers                                              #
# ------------------------------------------------------------------ #
def summarise_lists(group: pd.DataFrame,
                    combos: List[Tuple[str, str]]) -> Dict[str, str] | None:
    """Return {column_name → 'C2,C2…'}  or None if any combo absent."""
    bucket: Dict[Tuple[str, str], Set[str]] = {}
    for _, row in group.iterrows():
        bucket.setdefault((row["Input_Case"], row["Output_Case"]), set()).add(row["C2"])
    if any(key not in bucket for key in combos):
        return None
    return {COLUMN_NAMES[k]: ",".join(sorted(bucket[k])) for k in combos}

def summarise_best(group: pd.DataFrame,
                   combos: List[Tuple[str, str]],
                   numeric_cols: List[str]) -> Dict[str, str]:
    """
    Return {column_name → bestC2 or list} for combos that exist.
    For combos in NEEDS_BEST pick the row with max average score.
    """
    best: Dict[Tuple[str, str], Tuple[str, float]] = {}   # combo → (C2, score)
    lists: Dict[Tuple[str, str], Set[str]] = {}
    for _, row in group.iterrows():
        key = (row["Input_Case"], row["Output_Case"])
        c2  = row["C2"]
        if key in NEEDS_BEST:
            sc = row_score(row, numeric_cols)
            if key not in best or sc > best[key][1]:
                best[key] = (c2, sc)
        lists.setdefault(key, set()).add(c2)

    out = {}
    for key in combos:
        col = COLUMN_NAMES[key]
        if key in NEEDS_BEST:
            out[col] = best[key][0] if key in best else ""
        else:                       # “All Fake” keeps full list
            out[col] = ",".join(sorted(lists.get(key, [])))
    return out

# ------------------------------------------------------------------ #
# main                                                               #
# ------------------------------------------------------------------ #
def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Summarise C1 patterns and pick best C2 by frequency."
    )


    df = load_job_tsv("wordlist/Job1_Inner_Category__AA_AH_UH.tsv")
    by_c1 = group_by_c1(df)

    # figure out which numeric columns exist (once)
    numeric_cols = [c for c in df.columns if c.endswith("_Freq") or c.endswith("_Frequency")]

    for tag, combo_list in PATTERNS.items():
        rows_lists = []
        rows_best  = []
        for c1, grp in by_c1.items():
            summary = summarise_lists(grp, combo_list)
            if not summary:
                continue
            best     = summarise_best(grp, combo_list, numeric_cols)

            rows_lists.append({"C1": c1, **summary})
            rows_best .append({"C1": c1, **best})

        if not rows_lists:
            continue

        ordered = ["C1"] + [COLUMN_NAMES[c] for c in combo_list]

        print(f"\n=== Pattern {tag}  –  FULL LIST ===")
        print(pd.DataFrame(rows_lists)[ordered].to_string(index=False))

        print(f"\n=== Pattern {tag}  –  BEST WORDS ===")
        print(pd.DataFrame(rows_best)[ordered].to_string(index=False))

if __name__ == "__main__":
    sys.exit(main())



