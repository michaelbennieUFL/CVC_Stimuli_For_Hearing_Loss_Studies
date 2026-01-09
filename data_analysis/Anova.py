#!/usr/bin/env python3
import json
import itertools
import pandas as pd
from pathlib import Path
from statsmodels.stats.anova import AnovaRM

from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests

# ---------------- Paths ----------------
ROOT = Path("./")
CVC_PATH = Path("./analysis") / "CVC_all_human_readable.tsv"
WORD_VALUES_JSON = Path("./data") / "word_values.json"
ANALYSIS_DIR = ROOT / "analysis"
ANALYSIS_DIR.mkdir(exist_ok=True)

# ---------------- Load data ----------------
df = pd.read_csv(CVC_PATH, sep="\t", dtype=str)
word_defs = json.loads(WORD_VALUES_JSON.read_text(encoding="utf-8"))

def clean_str(x):
    return "" if x is None else str(x).strip()

def valid_ans(x):
    x = clean_str(x)
    if x.lower() in {"", "nan", "none"}:
        return False
    return x != "-1"

# ---------------- Build condition maps ----------------
pair_map = {}
type_map = {}

for w in word_defs:
    c1, c2 = w["C1"], w["C2"]

    # WordPair label: AE vs IH (Spelling1 is [AE, EH, IH])
    s = w["Spelling1"]
    pair_map[(c1, c2)] = f"{s[0]}_{s[2]}" if isinstance(s, list) and len(s) >= 3 else None

    # WordType label: real_fake_real etc.
    type_map[(c1, c2)] = "_".join(w["Real_Fake_word"])

df["C1"] = df["C1"].astype(str).str.strip()
df["C2"] = df["C2"].astype(str).str.strip()
df["WordPair"] = df.apply(lambda r: pair_map.get((r["C1"], r["C2"])), axis=1)
df["WordType"] = df.apply(lambda r: type_map.get((r["C1"], r["C2"])), axis=1)

# ====================================================
# Helper: write markdown tables
# ====================================================
def write_md_table(df_, path, title):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(df_.to_markdown(index=True))
        f.write("\n")

def write_anova_md(anova, path, factor, dv_label):
    a = anova.anova_table.reset_index()
    a = a.rename(columns={"Num DF": "df_num", "Den DF": "df_den", "Pr > F": "p"})
    a = a[["index", "F Value", "df_num", "df_den", "p"]]
    a.columns = ["Effect", "F", "df_num", "df_den", "p"]

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Repeated-measures ANOVA: {factor}\n\n")
        f.write(f"DV: **{dv_label}**\n\n")
        f.write(a.to_markdown(index=False))
        f.write("\n")

# ====================================================
# Helper: post-hoc paired t-tests (all pairs) + correction
# ====================================================
def posthoc_paired_ttests(
    wide: pd.DataFrame,
    *,
    correction: str = "holm",
    alpha: float = 0.05,
    drop_zero_variance: bool = True,
) -> pd.DataFrame:
    cols = list(wide.columns)
    results = []

    for a, b in itertools.combinations(cols, 2):
        x = pd.to_numeric(wide[a], errors="coerce")
        y = pd.to_numeric(wide[b], errors="coerce")

        pair = pd.concat([x, y], axis=1).dropna()
        if len(pair) < 2:
            continue

        x2 = pair.iloc[:, 0]
        y2 = pair.iloc[:, 1]
        d = x2 - y2

        if drop_zero_variance and float(d.var(ddof=1)) == 0.0:
            tval = float("nan")
            pval = float("nan")
        else:
            tval, pval = ttest_rel(x2, y2, nan_policy="omit")

        results.append({
            "A": a,
            "B": b,
            "n": int(len(pair)),
            "mean_A": float(x2.mean()),
            "mean_B": float(y2.mean()),
            "mean_diff_A_minus_B": float(d.mean()),
            "t": float(tval) if pd.notna(tval) else float("nan"),
            "p_uncorrected": float(pval) if pd.notna(pval) else float("nan"),
        })

    res = pd.DataFrame(results)
    if res.empty:
        return res

    mask = res["p_uncorrected"].notna()
    pvals = res.loc[mask, "p_uncorrected"].to_numpy()
    reject, p_corr, _, _ = multipletests(pvals, alpha=alpha, method=correction)

    res.loc[mask, "p_corrected"] = p_corr
    res.loc[mask, "reject_H0"] = reject
    res = res.sort_values(["p_corrected", "p_uncorrected"], na_position="last").reset_index(drop=True)
    return res

def write_posthoc_md(res: pd.DataFrame, path: Path, title: str, correction: str, dv_label: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"DV: **{dv_label}**\n\n")
        f.write(f"Multiple-comparisons correction: **{correction}**\n\n")
        if res.empty:
            f.write("_No valid pairwise tests were computed._\n")
            return

        out = res.copy()
        for c in ["mean_A", "mean_B", "mean_diff_A_minus_B", "t", "p_uncorrected", "p_corrected"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").round(6)
        f.write(out.to_markdown(index=False))
        f.write("\n")

# ====================================================
# Core runner: do ANOVA1/ANOVA2 + posthoc for a DV
# ====================================================
def run_all(dv_col: str, dv_label: str, suffix: str = ""):
    # ---------- ANOVA 1: WordPair ----------
    df1 = df[df["WordPair"].notna()].copy()

    anova1_tbl = (
        df1.groupby(["id", "WordPair"])[dv_col]
           .sum()
           .unstack(fill_value=0)
           .sort_index()
    )

    anova1_long = anova1_tbl.reset_index().melt(
        id_vars="id", var_name="WordPair", value_name="dv"
    )

    anova1 = AnovaRM(anova1_long, depvar="dv", subject="id", within=["WordPair"]).fit()

    write_md_table(
        anova1_tbl,
        ANALYSIS_DIR / f"anova1_wordpair_data{suffix}.md",
        f"ANOVA 1 data: Subjects × WordPair ({dv_label})"
    )
    write_anova_md(
        anova1,
        ANALYSIS_DIR / f"anova1_wordpair_results{suffix}.md",
        "WordPair",
        dv_label
    )

    posthoc1 = posthoc_paired_ttests(anova1_tbl, correction="holm")
    write_posthoc_md(
        posthoc1,
        ANALYSIS_DIR / f"posthoc_wordpair_ttests{suffix}.md",
        "Post-hoc paired t-tests: WordPair",
        correction="holm",
        dv_label=dv_label
    )

    # ---------- ANOVA 2: WordType ----------
    df2 = df[df["WordType"].notna()].copy()

    anova2_tbl = (
        df2.groupby(["id", "WordType"])[dv_col]
           .sum()
           .unstack(fill_value=0)
           .sort_index()
    )

    anova2_long = anova2_tbl.reset_index().melt(
        id_vars="id", var_name="WordType", value_name="dv"
    )

    anova2 = AnovaRM(anova2_long, depvar="dv", subject="id", within=["WordType"]).fit()

    write_md_table(
        anova2_tbl,
        ANALYSIS_DIR / f"anova2_wordtype_data{suffix}.md",
        f"ANOVA 2 data: Subjects × WordType ({dv_label})"
    )
    write_anova_md(
        anova2,
        ANALYSIS_DIR / f"anova2_wordtype_results{suffix}.md",
        "WordType",
        dv_label
    )

    posthoc2 = posthoc_paired_ttests(anova2_tbl, correction="holm")
    write_posthoc_md(
        posthoc2,
        ANALYSIS_DIR / f"posthoc_wordtype_ttests{suffix}.md",
        "Post-hoc paired t-tests: WordType",
        correction="holm",
        dv_label=dv_label
    )

# ====================================================
# DV 1: Ans1_only (your existing case)
# ====================================================
df["Ans1_only"] = (
    df["Ans1V"].apply(valid_ans)
    & ~df["Ans2V"].apply(valid_ans)
    & ~df["Ans3V"].apply(valid_ans)
).astype(int)

# ====================================================
# DV 2: Full fusion (EH): Ans1_only AND Ans1V == "EH"
# ====================================================
df["FullFusion_EH"] = (
    (df["Ans1_only"] == 1)
    & (df["Ans1V"].apply(clean_str) == "EH")
).astype(int)

# Run both analyses
run_all("Ans1_only", "Ans1V-only (Ans2V & Ans3V missing)", suffix="")
run_all("FullFusion_EH", "Full fusion (EH): Ans1V-only AND Ans1V == EH", suffix="_fullfusion_EH")

print("Markdown ANOVA + post-hoc tables written to ./analysis/ (including fullfusion_EH)")
