import pandas as pd
from pathlib import Path
from typing import List, Tuple

def load_and_stack(file_list: List[str]) -> pd.DataFrame:
    """
    Load multiple TSV/CSV files into one DataFrame.
    Detects TSV by extension .tsv or .tab; otherwise uses CSV defaults.
    Reads everything as strings to avoid type surprises.
    """
    frames = []
    for f in file_list:
        p = Path(f)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {f}")
        if p.suffix.lower() in {".tsv", ".tab"}:
            df = pd.read_csv(p, sep="\t", dtype=str)
        else:
            df = pd.read_csv(p, dtype=str)
        df["__source_file__"] = p.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def _resolve_col(df: pd.DataFrame, *candidates: str) -> str:
    """Case-insensitive column resolver with helpful error messages."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    raise KeyError(f"Could not find any of columns {candidates}. Available: {list(df.columns)}")

def compute_eh_rate_by_pairs(
    df: pd.DataFrame,
    pairs: List[Tuple[str, str]],
    vowel1_vals=("IH", "AE"),
    ans_cols=("Ans1V", "Ans2V", "Ans3V"),
    c1_colname="C1",
    c2_colname="C2",
    vowel1_colname="Vowel1",
    vowel2_colname="Vowel2",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      pair_table: per (C1, C2) rate table with N and EH_count
      grouped_table: combined rate table for every adjacent pair in `pairs`
                     (e.g., (D,F)+(D,SH), (G,T)+(G,G), (L,P)+(L,JH))
    """
    # Resolve actual columns robustly (case-insensitive)
    C1 = _resolve_col(df, c1_colname)
    C2 = _resolve_col(df, c2_colname)
    V1 = _resolve_col(df, vowel1_colname)
    V2 = _resolve_col(df, vowel2_colname)
    Acols = [_resolve_col(df, a) for a in ans_cols]

    # Normalize whitespace/case
    for col in [C1, C2, V1, V2] + Acols:
        df[col] = df[col].fillna("").astype(str).str.strip()

    # Filter: Vowel1 ∈ {IH, AE} and Vowel2 != Vowel1
    filt = df[df[V1].isin(vowel1_vals) & (df[V2] != df[V1])].copy()

    # Any EH across answer-vowel columns
    any_eh = False
    for col in Acols:
        any_eh = any_eh | filt[col].eq("EH")
    filt["__any_EH__"] = any_eh

    # Per-pair table
    rows = []
    for c1_val, c2_val in pairs:
        sub = filt[(filt[C1] == c1_val) & (filt[C2] == c2_val)]
        n = len(sub)
        eh = int(sub["__any_EH__"].sum()) if n > 0 else 0
        rate = eh / n if n > 0 else float("nan")
        rows.append({"C1": c1_val, "C2": c2_val, "N_trials": n, "EH_count": eh, "EH_rate_anyAnsV": rate})
    pair_table = pd.DataFrame(rows)

    # Grouped rates for each adjacent pair (1&2, 3&4, 5&6, ...)
    grouped_rows = []
    for i in range(0, len(pairs), 2):
        p1 = pairs[i]
        p2 = pairs[i + 1] if i + 1 < len(pairs) else None
        sub1 = filt[(filt[C1] == p1[0]) & (filt[C2] == p1[1])]
        if p2 is not None:
            sub2 = filt[(filt[C1] == p2[0]) & (filt[C2] == p2[1])]
            combined = pd.concat([sub1, sub2], ignore_index=True)
            label = f"{p1[0]}-{p1[1]} / {p2[0]}-{p2[1]}"
        else:
            combined = sub1
            label = f"{p1[0]}-{p1[1]}"
        n = len(combined)
        eh = int(combined["__any_EH__"].sum()) if n > 0 else 0
        rate = eh / n if n > 0 else float("nan")
        grouped_rows.append({"Pair Group": label, "N_trials": n, "EH_count": eh, "EH_rate_anyAnsV": rate})
    grouped_table = pd.DataFrame(grouped_rows)

    return pair_table, grouped_table

# -------------------------
# Example usage on your files:
if __name__ == "__main__":
    files = [
        "./data/NH293/NH293_CVC_human_readable_2.tsv",
    ]

    big = load_and_stack(files)

    requested_pairs = [
        ("D", "F"), ("D", "SH"),
        ("G", "T"), ("G", "G"),
        ("L", "P"), ("L", "JH"),
    ]

    pair_table, grouped_table = compute_eh_rate_by_pairs(big, requested_pairs)

    # Save CSVs if you want
    pair_table.to_csv("eh_rates_by_pair.csv", index=False)
    grouped_table.to_csv("eh_rates_by_group.csv", index=False)

    print(pair_table)
    print(grouped_table)
