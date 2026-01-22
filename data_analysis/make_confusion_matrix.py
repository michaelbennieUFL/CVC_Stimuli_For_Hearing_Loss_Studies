#!/usr/bin/env python3
"""
Generate classic (uncolored) confusion-matrix tables of vowel-combination responses.

Y-axis: combinations of {IH, EH, AE} (powerset excluding empty set), ordered by subset size (1 -> 3),
        then lexicographically within size.
X-axis: "outer word pairs" (WordPair), ordered by:
        (1) fake-word spellings alphabetically (derived from word_values.json),
        (2) WordType order: real_real_real, real_fake_real, fake_real_fake, fake_fake_fake,
        (3) WordPair label as tie-breaker.

Outputs:
  analysis/confusion_matrices/
    confusion_ALL.tsv, confusion_ALL.md, confusion_ALL.png
    confusion_subj_<id>.tsv/.md/.png  (one set per subject)

Notes:
  - Rows with all zeros are pruned (per matrix).
  - Requires the same inputs used by your ANOVA script:
      ./analysis/CVC_all_human_readable.tsv
      ./data/word_values.json
"""

import json
import itertools
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Paths ----------------
ROOT = Path("./")
CVC_PATH = ROOT / "analysis" / "CVC_all_human_readable.tsv"
WORD_VALUES_JSON = ROOT / "data" / "word_values.json"

OUT_DIR = ROOT / "analysis" / "confusion_matrices"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Utilities ----------------
def clean_str(x):
    return "" if x is None else str(x).strip()

def valid_ans(x):
    x = clean_str(x)
    if x.lower() in {"", "nan", "none"}:
        return False
    return x != "-1"

VOWELS = ["IH", "EH", "AE"]

def powerset_labels(vowels=VOWELS):
    """Powerset excluding empty set: size 1 -> 3, lexicographic within size."""
    out = []
    for r in range(1, len(vowels) + 1):
        for comb in itertools.combinations(vowels, r):
            out.append("+".join(comb))
    return out

ROW_ORDER = powerset_labels()

WORDTYPE_ORDER = {
    "real_real_real": 0,
    "real_fake_real": 1,
    "fake_real_fake": 2,
    "fake_fake_fake": 3,
}

def response_combo(row):
    """
    Build a set-combo label from Ans1V/Ans2V/Ans3V (restricted to IH/EH/AE).
    Example outputs: 'EH', 'IH+AE', 'IH+EH+AE'.
    Returns '' (empty) if no valid vowel response is available.
    """
    vals = []
    for col in ("Ans1V", "Ans2V", "Ans3V"):
        if col in row and valid_ans(row[col]):
            v = clean_str(row[col]).upper()
            if v in VOWELS:
                vals.append(v)
    if not vals:
        return ""
    # unique, but keep fixed vowel order (IH, EH, AE)
    s = set(vals)
    ordered = [v for v in VOWELS if v in s]
    return "+".join(ordered)

def write_table_png(df_tbl: pd.DataFrame, out_png: Path, title: str = ""):
    """
    Save an uncolored table as a PNG using matplotlib.
    """
    if df_tbl.empty:
        # write a tiny placeholder figure
        fig, ax = plt.subplots(figsize=(6, 1.5))
        ax.axis("off")
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        return

    nrows, ncols = df_tbl.shape
    # Heuristic sizing: scale with number of columns/rows
    fig_w = max(8, min(28, 0.8 * ncols + 4))
    fig_h = max(2.5, min(18, 0.45 * nrows + 1.8))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    if title:
        ax.set_title(title, pad=12)

    cell_text = df_tbl.values.tolist()
    col_labels = list(df_tbl.columns)
    row_labels = list(df_tbl.index)

    tbl = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.2)

    # Rotate column labels if there are many columns
    if ncols > 8:
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:  # header row in matplotlib table
                cell.get_text().set_rotation(45)
                cell.get_text().set_ha("left")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

# ---------------- Load data ----------------
df = pd.read_csv(CVC_PATH, sep="\t", dtype=str)

word_defs = json.loads(WORD_VALUES_JSON.read_text(encoding="utf-8"))

# Build WordPair/WordType maps per (C1, C2) and per WordPair label
pair_map = {}
type_map = {}
pair_meta = {}  # WordPair -> {wordtype, fake_key}

for w in word_defs:
    c1, c2 = w["C1"], w["C2"]
    s = w["Spelling1"]  # [AE, EH, IH] spellings
    rf = w["Real_Fake_word"]  # [AE, EH, IH] as "real"/"fake"

    # WordPair label uses outer spellings [AE]_[IH]
    if isinstance(s, list) and len(s) >= 3:
        wp = f"{s[0]}_{s[2]}"
    else:
        wp = None

    wt = "_".join(rf) if isinstance(rf, list) else None

    pair_map[(c1, c2)] = wp
    type_map[(c1, c2)] = wt

    # Fake-word key: collect spellings marked fake and sort alphabetically
    fake_spellings = []
    if isinstance(s, list) and isinstance(rf, list) and len(s) >= 3 and len(rf) >= 3:
        for spelling, status in zip(s, rf):
            if str(status).lower() == "fake":
                fake_spellings.append(str(spelling))
    fake_key = "+".join(sorted(fake_spellings, key=lambda x: x.lower()))

    if wp:
        pair_meta[wp] = {
            "wordtype": wt or "",
            "fake_key": fake_key,
        }

df["C1"] = df["C1"].astype(str).str.strip()
df["C2"] = df["C2"].astype(str).str.strip()
df["WordPair"] = df.apply(lambda r: pair_map.get((r["C1"], r["C2"])), axis=1)
df["WordType"] = df.apply(lambda r: type_map.get((r["C1"], r["C2"])), axis=1)
df["RespCombo"] = df.apply(response_combo, axis=1)

# Filter to trials with a WordPair and a non-empty response combo
df_use = df[(df["WordPair"].notna()) & (df["RespCombo"].astype(str) != "")].copy()

# Determine ordered WordPair columns
wordpairs = sorted(df_use["WordPair"].dropna().unique().tolist())

def sort_key(wp: str):
    meta = pair_meta.get(wp, {})
    wt = meta.get("wordtype", "")
    fk = meta.get("fake_key", "")
    return (
        fk.lower(),
        WORDTYPE_ORDER.get(wt, 999),
        wp.lower(),
    )

ordered_wordpairs = sorted(wordpairs, key=sort_key)

def build_matrix(df_part: pd.DataFrame) -> pd.DataFrame:
    tbl = (
        df_part
        .pivot_table(index="RespCombo", columns="WordPair", values="id", aggfunc="count", fill_value=0)
        .reindex(columns=ordered_wordpairs, fill_value=0)
    )

    # Ensure row order (powerset), then append any unexpected combos at the end
    seen = set(tbl.index.tolist())
    ordered_rows = [r for r in ROW_ORDER if r in seen]
    extras = [r for r in tbl.index.tolist() if r not in ROW_ORDER]
    tbl = tbl.reindex(index=ordered_rows + extras, fill_value=0)

    # Prune all-zero rows
    tbl = tbl.loc[(tbl.sum(axis=1) != 0)]
    return tbl

def save_outputs(tbl: pd.DataFrame, stem: str, title: str):
    out_tsv = OUT_DIR / f"{stem}.tsv"
    out_md = OUT_DIR / f"{stem}.md"
    out_png = OUT_DIR / f"{stem}.png"

    tbl.to_csv(out_tsv, sep="\t")
    out_md.write_text(tbl.to_markdown(), encoding="utf-8")
    write_table_png(tbl, out_png, title=title)

# ---------------- Combined (all subjects) ----------------
tbl_all = build_matrix(df_use)
save_outputs(tbl_all, "confusion_ALL", "All subjects: vowel-combination responses by WordPair")

# ---------------- Per-subject ----------------
if "id" not in df_use.columns:
    raise ValueError("Column 'id' not found in the TSV. Expected a subject identifier column named 'id'.")

for sid in sorted(df_use["id"].dropna().unique().tolist(), key=lambda x: str(x)):
    df_s = df_use[df_use["id"] == sid]
    tbl_s = build_matrix(df_s)
    save_outputs(tbl_s, f"confusion_subj_{sid}", f"Subject {sid}: vowel-combination responses by WordPair")

print(f"Done. Wrote outputs to: {OUT_DIR.resolve()}")
