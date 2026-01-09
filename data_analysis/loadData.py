#!/usr/bin/env python3
"""
Prepare 2 master pandas tables (CVCmono + CVC) from *human_readable* TSVs,
add `id` + `instance`, optionally filter to a subject/instance allow-list,
and optionally save the combined + filtered outputs for later analysis.

Supports selection JSON formats like:
  Old:
    {"NH273": {"mono":[0,1], "dichotic":[0,2]}}
  New (yours):
    {"NH273": {"mono_b":[0,1,2], "mono_l":[3,4], "mono_r":[5,6], "dichotic":[0,1,2]}}
"""

import re
import json
import argparse
from pathlib import Path

import pandas as pd


# ------------------------------------------------------------
# 1) Robust TSV loader (skips "Ear: ..." style metadata lines)
# ------------------------------------------------------------
def smart_read_tsv(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    header_idx = None
    for i, line in enumerate(lines[:300]):
        if "\t" in line and re.search(r"\bTrial\b", line):
            header_idx = i
            break
    if header_idx is None:
        for i, line in enumerate(lines[:300]):
            if "\t" in line:
                header_idx = i
                break
    if header_idx is None:
        raise ValueError(f"No tab-delimited header found in {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        header=0,
        skiprows=header_idx,
        engine="python",
        dtype=str,
    )
    df.columns = [str(c).strip() for c in df.columns]
    return df


# ------------------------------------------------------------
# 2) File discovery
# ------------------------------------------------------------
def discover_human_readable_files(root: Path):
    cvcmono = sorted(root.rglob("*CVCmono_human_readable_*.tsv"))

    cvc_all = sorted(root.rglob("*CVC_human_readable_*.tsv"))
    cvc = [p for p in cvc_all if "CVCmono_human_readable_" not in p.name]

    return cvcmono, cvc


# ------------------------------------------------------------
# 3) Tagging helpers: id + instance
# ------------------------------------------------------------
def extract_id_from_path(p: Path) -> str:
    # Prefer folder segment like .../NH273/...
    for part in p.parts[::-1]:
        if re.fullmatch(r"NH\d+", part):
            return part

    # Fallback: somewhere in filename
    m = re.search(r"(NH\d+)", p.name)
    return m.group(1) if m else "UNKNOWN"


def extract_instance(p: Path):
    m = re.search(r"_(\d+)\.(?:tsv|txt)$", p.name)
    if m:
        return int(m.group(1))
    last = p.stem.split("_")[-1]
    return int(last) if last.isdigit() else pd.NA


def _strip_string_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].fillna("").astype(str).str.strip()
    return out


def load_stack(files, kind_label: str) -> pd.DataFrame:
    frames = []
    for f in files:
        df = smart_read_tsv(f)

        df["id"] = extract_id_from_path(f)
        df["instance"] = extract_instance(f)

        # provenance (handy for debugging)
        df["__source_file__"] = f.name
        df["__kind__"] = kind_label

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True, sort=False)
    out = _strip_string_cols(out)
    out["instance"] = pd.to_numeric(out["instance"], errors="coerce").astype("Int64")
    return out


# ------------------------------------------------------------
# 4) Build the two master tables
# ------------------------------------------------------------
def build_master_tables(root: str = "./data"):
    rootp = Path(root)
    cvcmono_files, cvc_files = discover_human_readable_files(rootp)

    df_mono = load_stack(cvcmono_files, "cvcmono")
    df_cvc = load_stack(cvc_files, "cvc")

    return df_mono, df_cvc


# ------------------------------------------------------------
# 5) Selection normalization + mono ear labeling
# ------------------------------------------------------------
def _as_int_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    return [int(x)]


def normalize_selection(selection: dict | None) -> dict:
    """
    Returns a normalized mapping:
      norm[subj]["mono"]     -> union of mono / mono_b / mono_l / mono_r
      norm[subj]["dichotic"] -> dichotic list (if present)
      norm[subj]["mono_ear_map"] -> dict {instance: "both"/"left"/"right"} if available
    """
    if not selection:
        return {}

    norm = {}
    for subj, spec in selection.items():
        spec = spec or {}

        mono = set(_as_int_list(spec.get("mono")))  # old format

        mono_b = set(_as_int_list(spec.get("mono_b")))
        mono_l = set(_as_int_list(spec.get("mono_l")))
        mono_r = set(_as_int_list(spec.get("mono_r")))

        mono |= (mono_b | mono_l | mono_r)

        ear_map = {}
        for inst in mono_b:
            ear_map[inst] = "both"
        for inst in mono_l:
            ear_map[inst] = "left"
        for inst in mono_r:
            ear_map[inst] = "right"

        norm[subj] = {
            "mono": sorted(mono),
            "dichotic": _as_int_list(spec.get("dichotic")),
            "mono_ear_map": ear_map,  # may be empty
        }

    return norm


def add_mono_ear_column(df_mono: pd.DataFrame, selection: dict | None) -> pd.DataFrame:
    """
    Adds df_mono["mono_ear"] = both/left/right based on selection's mono_b/mono_l/mono_r.
    If selection missing or no match for (id,instance), mono_ear is <NA>.
    """
    out = df_mono.copy()
    out["instance"] = pd.to_numeric(out["instance"], errors="coerce").astype("Int64")

    norm = normalize_selection(selection)
    if not norm or out.empty:
        out["mono_ear"] = pd.NA
        return out

    rows = []
    for subj, spec in norm.items():
        for inst, ear in (spec.get("mono_ear_map") or {}).items():
            rows.append((subj, int(inst), ear))

    if not rows:
        out["mono_ear"] = pd.NA
        return out

    ear_df = pd.DataFrame(rows, columns=["id", "instance", "mono_ear"])
    ear_df["instance"] = pd.to_numeric(ear_df["instance"], errors="coerce").astype("Int64")

    out = out.merge(ear_df, on=["id", "instance"], how="left")
    return out


# ------------------------------------------------------------
# 6) Filtering by selection
# ------------------------------------------------------------
def filter_by_selection(df: pd.DataFrame, selection: dict | None, mode: str) -> pd.DataFrame:
    """
    mode: "mono" or "dichotic"
    Uses normalized selection:
      - mono = union of mono / mono_b / mono_l / mono_r
      - dichotic = spec["dichotic"]
    """
    if df.empty or not selection:
        return df.copy()

    norm = normalize_selection(selection)

    allowed = []
    for subj, spec in norm.items():
        for inst in spec.get(mode, []):
            allowed.append((subj, int(inst)))

    if not allowed:
        return df.iloc[0:0].copy()

    allowed_df = pd.DataFrame(allowed, columns=["id", "instance"])
    allowed_df["instance"] = pd.to_numeric(allowed_df["instance"], errors="coerce").astype("Int64")

    tmp = df.copy()
    tmp["instance"] = pd.to_numeric(tmp["instance"], errors="coerce").astype("Int64")

    return tmp.merge(allowed_df, on=["id", "instance"], how="inner")


def apply_selection(df_mono, df_cvc, selection):
    df_mono_f = filter_by_selection(df_mono, selection, mode="mono")
    df_cvc_f = filter_by_selection(df_cvc, selection, mode="dichotic")
    return df_mono_f, df_cvc_f


# ------------------------------------------------------------
# 7) One setup function to use everywhere later
# ------------------------------------------------------------
def setup_human_readable_tables(
    root: str = "./data",
    selection: dict | None = None,
    outdir: str | None = None,
    fmt: str = "tsv",  # "tsv" or "csv"
    save_filtered: bool = True,
):
    """
    Returns:
        df_mono, df_cvc, df_mono_filtered, df_cvc_filtered

    Adds `mono_ear` to mono tables based on selection's mono_b/mono_l/mono_r (if provided).
    """
    df_mono, df_cvc = build_master_tables(root=root)

    # add ear label to mono (full table)
    df_mono = add_mono_ear_column(df_mono, selection)

    # filter (if selection provided)
    if selection:
        df_mono_f, df_cvc_f = apply_selection(df_mono, df_cvc, selection)
    else:
        df_mono_f, df_cvc_f = df_mono.copy(), df_cvc.copy()

    # ensure filtered mono keeps mono_ear (it will, since it came from df_mono)
    if outdir:
        outp = Path(outdir)
        outp.mkdir(parents=True, exist_ok=True)

        if fmt.lower() == "csv":
            mono_path = outp / "CVCmono_all_human_readable.csv"
            cvc_path = outp / "CVC_all_human_readable.csv"
            df_mono.to_csv(mono_path, index=False)
            df_cvc.to_csv(cvc_path, index=False)

            if save_filtered:
                (outp / "CVCmono_filtered.csv").write_text("")  # ensures path exists in some editors
                df_mono_f.to_csv(outp / "CVCmono_filtered.csv", index=False)
                df_cvc_f.to_csv(outp / "CVC_filtered.csv", index=False)
        else:
            mono_path = outp / "CVCmono_all_human_readable.tsv"
            cvc_path = outp / "CVC_all_human_readable.tsv"
            df_mono.to_csv(mono_path, sep="\t", index=False)
            df_cvc.to_csv(cvc_path, sep="\t", index=False)

            if save_filtered:
                df_mono_f.to_csv(outp / "CVCmono_filtered.tsv", sep="\t", index=False)
                df_cvc_f.to_csv(outp / "CVC_filtered.tsv", sep="\t", index=False)

    return df_mono, df_cvc, df_mono_f, df_cvc_f


# ------------------------------------------------------------
# CLI (optional)
# ------------------------------------------------------------
def _load_selection_json(path: str | None):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Selection JSON not found: {path}")
    return json.loads(p.read_text(encoding="utf-8"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./data", help="Root folder containing NH### subfolders")
    ap.add_argument("--outdir", default="./analysis", help="Where to write combined outputs")
    ap.add_argument("--fmt", choices=["tsv", "csv"], default="tsv", help="Output format")
    ap.add_argument(
        "--selection_json",
        default="./data/subject_data_selection.json",
        help="Optional JSON file with selection mapping",
    )
    ap.add_argument(
        "--no_save_filtered",
        action="store_true",
        help="If set, only save the full combined tables (not filtered tables)",
    )
    args = ap.parse_args()

    selection = _load_selection_json(args.selection_json)

    df_mono, df_cvc, df_mono_f, df_cvc_f = setup_human_readable_tables(
        root=args.root,
        selection=selection,
        outdir=args.outdir,
        fmt=args.fmt,
        save_filtered=not args.no_save_filtered,
    )

    print("Done.")
    print("Rows:")
    print("  CVCmono full:", len(df_mono))
    print("  CVC full:    ", len(df_cvc))
    print("  CVCmono filt:", len(df_mono_f))
    print("  CVC filt:    ", len(df_cvc_f))
    if "mono_ear" in df_mono.columns:
        print("  mono_ear counts (full):")
        print(df_mono["mono_ear"].value_counts(dropna=False))
