# Define the requested functions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Sequence

VOWEL_ORDER = ["IH", "EH", "AE"]


def make_confusion_matrix(
        dfs: List[pd.DataFrame],
        relative: bool = False,
        c1_include: Optional[Sequence[str]] = None,
        c2_include: Optional[Sequence[str]] = None,
        f0_include: Optional[Sequence[str]] = None,
        show_combined_average: bool = False,
        separate_by_f0: bool = False,
        vowel_col: str = "Vowel",
        answer_vowel_col: str = "Answer_vowel",
        c1_col: str = "C1",
        c2_col: str = "C2",
        f0_col: str = "F0",
) -> pd.DataFrame:
    """
    Build a confusion matrix DataFrame for actual vs. answered vowels.

    Parameters
    ----------
    dfs : list of pd.DataFrame
        Input data frames that will be concatenated.
    relative : bool
        If True, return relative frequencies (proportion of all items). If False, return absolute counts.
    c1_include, c2_include, f0_include : Optional[sequence of str]
        If provided, filter rows to only those with values in these lists.
    show_combined_average : bool
        If True, return a single-cell DataFrame with the overall accuracy
        (sum of the diagonal divided by total), with column name 'combined average score'.
    separate_by_f0 : bool
        If True, split rows by low/high F0 (expects values like 'low_f0'/'high_f0' in f0_col).
        Row labels become IH_low, IH_high, ..., AE_low, AE_high (only the ones present in data).
    vowel_col, answer_vowel_col, c1_col, c2_col, f0_col : str
        Column names for the relevant fields.
    """
    if not dfs:
        raise ValueError("No dataframes provided.")

    # Concatenate and standardize relevant columns
    df = pd.concat(dfs, ignore_index=True)

    # Basic sanity: keep only rows where both vowel labels exist (non-null strings)
    df = df[[vowel_col, answer_vowel_col, c1_col, c2_col, f0_col]].dropna(subset=[vowel_col, answer_vowel_col])

    # Apply optional filters (case-insensitive)
    if c1_include is not None:
        df = df[df[c1_col].str.lower().isin([v.lower() for v in c1_include])]
    if c2_include is not None:
        df = df[df[c2_col].str.lower().isin([v.lower() for v in c2_include])]
    if f0_include is not None:
        df = df[df[f0_col].str.lower().isin([v.lower() for v in f0_include])]

    # Optionally split rows by F0 category, using what's already in the dataset (e.g., 'low_f0'/'high_f0')
    if separate_by_f0:
        row_index = df[vowel_col].astype(str) + "_" + df[f0_col].astype(str).str.replace("_f0", "", regex=False)
    else:
        row_index = df[vowel_col].astype(str)

    # Build the confusion matrix (counts)
    conf = pd.crosstab(
        index=row_index,
        columns=df[answer_vowel_col],
        dropna=False
    )

    # Ensure columns are ordered IH, EH, AE (even if some are missing)
    conf = conf.reindex(columns=VOWEL_ORDER, fill_value=0)

    # Ensure row order
    if separate_by_f0:
        # Build desired row order dynamically based on what's present
        desired_rows = []
        for base in VOWEL_ORDER:
            for lvl in ["low", "high"]:
                label = f"{base}_{lvl}"
                if label in conf.index:
                    desired_rows.append(label)
        if desired_rows:
            conf = conf.reindex(desired_rows)
    else:
        conf = conf.reindex([v for v in VOWEL_ORDER if v in conf.index])

    # If relative, convert to proportions over each row
    if relative:
        conf = conf.astype(float)
        row_sums = conf.sum(axis=1)
        conf = conf.div(row_sums, axis=0).fillna(0)

    # If only the combined average is requested, compute accuracy and return single-column DF
    if show_combined_average:
        # Compute on absolute counts for interpretability, regardless of `relative`
        total = conf.values.sum()
        # If relative was used, reverse it by recomputing from counts (so recompute from scratch)
        if relative:
            conf_counts = pd.crosstab(index=row_index, columns=df[answer_vowel_col], dropna=False)
            conf_counts = conf_counts.reindex(columns=VOWEL_ORDER, fill_value=0)
            if separate_by_f0:
                desired_rows = []
                for base in VOWEL_ORDER:
                    for lvl in ["low", "high"]:
                        label = f"{base}_{lvl}"
                        if label in conf_counts.index:
                            desired_rows.append(label)
                if desired_rows:
                    conf_counts = conf_counts.reindex(desired_rows)
            else:
                conf_counts = conf_counts.reindex([v for v in VOWEL_ORDER if v in conf_counts.index])
            conf_used = conf_counts
            total = conf_used.values.sum()
        else:
            conf_used = conf

        # Accuracy: sum of "matching labels" cells / total
        # For F0-split, a "correct" is when the predicted vowel matches the base vowel (ignore low/high)
        diag_sum = 0
        if separate_by_f0:
            for idx in conf_used.index:
                base = idx.split("_")[0]
                if base in conf_used.columns:
                    diag_sum += conf_used.loc[idx, base]
        else:
            # standard diagonal
            common = [v for v in VOWEL_ORDER if v in conf_used.index and v in conf_used.columns]
            diag_sum = sum(conf_used.loc[v, v] for v in common)

        score = (diag_sum / total) if total > 0 else np.nan
        # Return as a 1x1 DataFrame with requested column name
        return pd.DataFrame({"combined average score": [score]})

    return conf


def plot_confusion_heatmap(conf_df: pd.DataFrame, title: Optional[str] = None):
    """
    Render a heatmap for a confusion matrix DataFrame (rows/columns as labels).
    - Uses matplotlib only.
    - One chart per call.
    - No explicit colors are set.
    """
    if conf_df.empty:
        raise ValueError("The confusion matrix DataFrame is empty.")

    fig = plt.figure()
    ax = plt.gca()
    im = ax.imshow(conf_df.values, aspect="equal")
    ax.set_xticks(range(conf_df.shape[1]))
    ax.set_xticklabels(list(conf_df.columns))
    ax.set_yticks(range(conf_df.shape[0]))
    ax.set_yticklabels(list(conf_df.index))
    ax.set_xlabel("Answer Vowel")
    ax.set_ylabel("Actual Vowel")
    if title:
        ax.set_title(title)
    # Annotate cells
    for i in range(conf_df.shape[0]):
        for j in range(conf_df.shape[1]):
            val = conf_df.iloc[i, j]
            if isinstance(val, (float, np.floating)):
                text = f"{val:.2f}"
            else:
                text = f"{int(val)}"
            ax.text(j, i, text, ha="center", va="center")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()



def plot_Monoral_Both_Ear_data():
    paths = [
        "./data/NH290/NH290_CVCmono_human_readable_1.tsv",
        "./data/NH290/NH290_CVCmono_human_readable_2.tsv",

        "./data/NH291/NH291_CVCmono_human_readable_1.tsv",
        "./data/NH291/NH291_CVCmono_human_readable_2.tsv",

        "./data/NH292/NH292_CVCmono_human_readable_1.tsv",
        "./data/NH292/NH292_CVCmono_human_readable_2.tsv",

        "./data/NH293/NH293_CVCmono_human_readable_1.tsv",
        "./data/NH293/NH293_CVCmono_human_readable_2.tsv",

        "./data/NH295/NH294_CVCmono_human_readable_1.tsv",
        "./data/NH295/NH294_CVCmono_human_readable_2.tsv",
    ]

    dfs = [pd.read_csv(path, sep="\t") for path in paths]

    demo_conf_abs = make_confusion_matrix(dfs, relative=False)
    demo_conf_rel = make_confusion_matrix(dfs, relative=True)
    demo_conf_f0 = make_confusion_matrix(dfs, relative=False, separate_by_f0=True)
    demo_conf_f0_rel = make_confusion_matrix(dfs, relative=True, separate_by_f0=True)

    demo_combined_avg = make_confusion_matrix(dfs, show_combined_average=True)
    # plot_confusion_heatmap(demo_conf_abs, title="Combined Monaural Both Ears\n Vowel Confusion Matrix (Absolute)")
    plot_confusion_heatmap(demo_conf_rel, title="Combined Monaural Both Ears\n Vowel Confusion Matrix (Relative)")
    # plot_confusion_heatmap(demo_conf_f0, title="Combined Monaural Both Ears\n Vowel/F0 Confusion Matrix (Absolute)")
    plot_confusion_heatmap(demo_conf_f0_rel, title="Combined Monaural Both Ears\n Vowel/F0 Confusion Matrix (Relative)")
    print(demo_combined_avg)

def plot_Monoral_Both_Ear_by_initials(initials=("D", "L", "G")):
    paths = [
        "./data/NH290/NH290_CVCmono_human_readable_1.tsv",
        "./data/NH290/NH290_CVCmono_human_readable_2.tsv",

        "./data/NH291/NH291_CVCmono_human_readable_1.tsv",
        "./data/NH291/NH291_CVCmono_human_readable_2.tsv",

        "./data/NH292/NH292_CVCmono_human_readable_1.tsv",
        "./data/NH292/NH292_CVCmono_human_readable_2.tsv",

        "./data/NH293/NH293_CVCmono_human_readable_1.tsv",
        "./data/NH293/NH293_CVCmono_human_readable_2.tsv",

        "./data/NH295/NH294_CVCmono_human_readable_1.tsv",
        "./data/NH295/NH294_CVCmono_human_readable_2.tsv",
    ]
    dfs = [pd.read_csv(path, sep="\t") for path in paths]

    for ini in initials:
        # # Absolute counts
        # conf_abs = make_confusion_matrix(
        #     dfs, relative=False, c1_include=[ini]
        # )
        # plot_confusion_heatmap(
        #     conf_abs,
        #     title=f"Combined Monaural Both Ears — Initial {ini}\nVowel Confusion Matrix (Absolute)"
        # )

        # Row-normalized (each row sums to 1.00)
        conf_rel = make_confusion_matrix(
            dfs, relative=True, c1_include=[ini]
        )
        plot_confusion_heatmap(
            conf_rel,
            title=f"Combined Monaural Both Ears — Initial {ini}\nVowel Confusion Matrix (Relative)"
        )

        # # With F0 split (absolute)
        # conf_f0_abs = make_confusion_matrix(
        #     dfs, relative=False, separate_by_f0=True, c1_include=[ini]
        # )
        # plot_confusion_heatmap(
        #     conf_f0_abs,
        #     title=f"Combined Monaural Both Ears — Initial {ini}\nVowel/F0 Confusion Matrix (Absolute)"
        # )

        # With F0 split (row-normalized)
        conf_f0_rel = make_confusion_matrix(
            dfs, relative=True, separate_by_f0=True, c1_include=[ini]
        )
        plot_confusion_heatmap(
            conf_f0_rel,
            title=f"Combined Monaural Both Ears — Initial {ini}\nVowel/F0 Confusion Matrix (Relative)"
        )

        # Combined average accuracy for this initial (computed on counts)
        avg_df = make_confusion_matrix(
            dfs, c1_include=[ini], show_combined_average=True
        )
        print(f"Initial {ini} —", avg_df.to_string(index=False))


def plot_Monoral_Left_Ear_data():
    paths = [
        "./data/NH290/NH290_CVCmono_human_readable_3.tsv",
        "./data/NH290/NH290_CVCmono_human_readable_4.tsv",

        "./data/NH291/NH291_CVCmono_human_readable_3.tsv",
        "./data/NH291/NH291_CVCmono_human_readable_4.tsv",

        "./data/NH292/NH292_CVCmono_human_readable_3.tsv",
        "./data/NH292/NH292_CVCmono_human_readable_5.tsv",

        "./data/NH293/NH293_CVCmono_human_readable_2.tsv",
        "./data/NH293/NH293_CVCmono_human_readable_5.tsv",

        "./data/NH295/NH294_CVCmono_human_readable_3.tsv",
        "./data/NH295/NH294_CVCmono_human_readable_4.tsv",
    ]

    dfs = [pd.read_csv(path, sep="\t") for path in paths]

    demo_conf_abs = make_confusion_matrix(dfs, relative=False)
    demo_conf_rel = make_confusion_matrix(dfs, relative=True)
    demo_conf_f0 = make_confusion_matrix(dfs, relative=False, separate_by_f0=True)
    demo_conf_f0_rel = make_confusion_matrix(dfs, relative=True, separate_by_f0=True)

    demo_combined_avg = make_confusion_matrix(dfs, show_combined_average=True)
    # plot_confusion_heatmap(demo_conf_abs, title="Combined Monaural Both Ears\n Vowel Confusion Matrix (Absolute)")
    plot_confusion_heatmap(demo_conf_rel, title="Combined Monaural Left Ears\n Vowel Confusion Matrix (Relative)")
    # plot_confusion_heatmap(demo_conf_f0, title="Combined Monaural Both Ears\n Vowel/F0 Confusion Matrix (Absolute)")
    plot_confusion_heatmap(demo_conf_f0_rel, title="Combined Monaural Left Ears\n Vowel/F0 Confusion Matrix (Relative)")
    print(demo_combined_avg)



def plot_Monoral_Right_Ear_data():
    paths = [
        "./data/NH290/NH290_CVCmono_human_readable_5.tsv",
        "./data/NH290/NH290_CVCmono_human_readable_6.tsv",

        "./data/NH291/NH291_CVCmono_human_readable_5.tsv",
        "./data/NH291/NH291_CVCmono_human_readable_6.tsv",

        "./data/NH292/NH292_CVCmono_human_readable_6.tsv",
        "./data/NH292/NH292_CVCmono_human_readable_8.tsv",

        "./data/NH293/NH293_CVCmono_human_readable_3.tsv",
        "./data/NH293/NH293_CVCmono_human_readable_4.tsv",

        "./data/NH295/NH294_CVCmono_human_readable_5.tsv",
        "./data/NH295/NH294_CVCmono_human_readable_6.tsv",
    ]

    dfs = [pd.read_csv(path, sep="\t") for path in paths]

    demo_conf_abs = make_confusion_matrix(dfs, relative=False)
    demo_conf_rel = make_confusion_matrix(dfs, relative=True)
    demo_conf_f0 = make_confusion_matrix(dfs, relative=False, separate_by_f0=True)
    demo_conf_f0_rel = make_confusion_matrix(dfs, relative=True, separate_by_f0=True)

    demo_combined_avg = make_confusion_matrix(dfs, show_combined_average=True)
    # plot_confusion_heatmap(demo_conf_abs, title="Combined Monaural Both Ears\n Vowel Confusion Matrix (Absolute)")
    plot_confusion_heatmap(demo_conf_rel, title="Combined Monaural Right Ears\n Vowel Confusion Matrix (Relative)")
    # plot_confusion_heatmap(demo_conf_f0, title="Combined Monaural Both Ears\n Vowel/F0 Confusion Matrix (Absolute)")
    plot_confusion_heatmap(demo_conf_f0_rel, title="Combined Monaural Right Ears\n Vowel/F0 Confusion Matrix (Relative)")
    print(demo_combined_avg)

if __name__ == "__main__":
    # plot_Monoral_Both_Ear_data()
    plot_Monoral_Both_Ear_by_initials(initials=("D", "L", "G"))
    plot_Monoral_Left_Ear_data()
    plot_Monoral_Right_Ear_data()

