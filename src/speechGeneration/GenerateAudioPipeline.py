"""cvc_variant_generator.py

Batch‑generate CVC words with formant‑tuned vowels **and** create all
possible discordant stereo pairs.

This script imports helper modules you already have and exposes three
high‑level APIs:

    1. `generate_vowel_variants()` – given a single vowel segment, optimise
       its formants to several targets and rebuild full CVC tokens. **Now
       returns a list of `(Path, label)` tuples so downstream code has
       access to both the audio file and the label used in its filename.**

    2. `generate_discordant_pairs()` – utility that, given the list returned
       by (1), builds every ordered left/right permutation and exports them
       as stereo wavs in a dedicated `discordants/` sub‑folder.

    3. `generate_cvc_dataset()` – end‑to‑end pipeline that speaks the word
       with ElevenLabs, segments it, calls (1) for each requested formant
       spec, **then calls (2) to create stereo discordants.**

No changes are required in your existing helper modules – this script just
combines them.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union, Any

from pydub import AudioSegment

from CVCSplitter import generate_splitAudio, recombine_cvc_audio
from elevenAudioGeneration import PhonemeTTSEngine
from modifyFormants import tune_formants
from src.analyzeAmEVowels.parseVowelDistFile import parse_vowel_dist_data
import os
from dotenv import load_dotenv, find_dotenv
from CVCSplitter import CROSSFADE_TIME


# ────────────────────────────────────────────────────────────────────────────
# constants & small helpers
# ────────────────────────────────────────────────────────────────────────────
VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY",
    "EH", "ER", "EY",
    "IH", "IY",
    "OW", "OY",
    "UH", "UW",
}  # stress‑stripped

def _norm(phone: str) -> str:
    """Upper‑case, strip any stress digit ("AE1" → "AE")."""
    return phone.upper().rstrip("0123456789")


def _extract_vowel(arpabet: str) -> str:
    """Return the *first* vowel encountered in an ARPAbet string (CVC)."""
    for token in arpabet.split():
        tok = _norm(token)
        if tok in VOWELS:
            return tok
    raise ValueError(f"No vowel found in '{arpabet}' – expected CVC phonemes")


def _sanitize(txt: str) -> str:
    """File‑system‑safe slug (spaces → '_', drop non‑alnum)."""
    import re

    txt = re.sub(r"[\s]+", "_", txt.strip())
    return re.sub(r"[^A-Za-z0-9_\-]", "", txt)


def _latest_wav(dir_: Path) -> Path:
    wavs = list(dir_.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No *.wav produced in {dir_}")
    return max(wavs, key=lambda p: p.stat().st_mtime)

# ────────────────────────────────────────────────────────────────────────────
# 1.  generate_vowel_variants
# ────────────────────────────────────────────────────────────────────────────

def generate_vowel_variants(
    vowel_wav: Path | str,
    c1_path: Path | str,
    c2_path: Path | str,
    vowel_phoneme: str,
    targets: Sequence[Dict[str, float]],
    dest_dir: Path | str,
    tolerance: float = 50.0,
    max_iters: int = 40,
    crossfade_time=3,
    vowel_length=None,
    max_percent_distance=0.002
) -> List[Tuple[Path, str]]:
    """Create *n* vowel‑tuned versions of an existing vowel chunk.

    Parameters
    ----------
    vowel_wav     : Path to the *original* vowel segment.
    c1_path, c2_path
                  : Paths to the onset and coda consonant chunks.
    vowel_phoneme : ARPAbet code for the vowel (stress‑stripped).
    targets       : Iterable of dicts – each dict can specify any subset of
                    {"f0", "f1", "f2", "f3", "f4"} in *Hz* and may optionally
                    include a "label" key that will be used in filenames.
    dest_dir      : Where to write the rebuilt CVC wavs and the tuned vowels.
    tolerance     : Pass‑through to `tune_formants()` (in Hz).
    max_iters     : Max optimisation steps (per target).

    Returns
    -------
    List[Tuple[Path, str]] – (audio_path, label) pairs in the same order as
    *targets*.  The *label* is whatever was used in the filename (either the
    provided `label` field or the fallback "var{i}").
    """
    vowel_wav = Path(vowel_wav).resolve()
    c1_path, c2_path = Path(c1_path).resolve(), Path(c2_path).resolve()
    dest_dir = Path(dest_dir).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    rebuilt: List[Tuple[Path, str]] = []

    for i, spec in enumerate(targets):
        # 1) run HiFi‑Glot optimisation --------------------------------------
        tgt_out = dest_dir / f"tuned_{i}"
        tgt_out.mkdir(parents=True, exist_ok=True)

        try:
            _, achieved = tune_formants(
                wav_file=vowel_wav,
                output_dir=tgt_out,
                target=spec,
                tolerance=tolerance,
                max_iters=max_iters,
                max_percent_distance=max_percent_distance
            )
        except RuntimeError as e:
            print(f"[WARN] Formant tuning #{i} failed: {e}")
            continue

        mod_vowel = _latest_wav(tgt_out)

        # 2) stitch back into C1 + C2 ----------------------------------------
        label = spec.get("label", f"var{i}")
        out_path = dest_dir / f"cvc_variant_{vowel_phoneme}_{label}.wav"
        recombine_cvc_audio(
            mod_vowel_path=mod_vowel,
            c1_path=c1_path,
            c2_path=c2_path,
            vowel_phoneme=vowel_phoneme,
            out_path=out_path,
            vowel_length=vowel_length,
            crossfade_time=crossfade_time
        )
        rebuilt.append((out_path, label))

    return rebuilt

# ────────────────────────────────────────────────────────────────────────────
# 2.  generate_discordant_pairs
# ────────────────────────────────────────────────────────────────────────────

def generate_discordant_pairs(
    vowel_phoneme: str,
    variants: List[Tuple[Path, str]],
    dest_dir: Path | str,
) -> List[Path]:
    """Create every *ordered* left/right permutation of the supplied variants.

    The resulting stereo files are written into `dest_dir/discordants/` and
    named `cvc_variant_{vowel}_discordant_{labelL}_{labelR}.wav`.

    Parameters
    ----------
    vowel_phoneme : The vowel symbol, used in filenames.
    variants      : Output of `generate_vowel_variants()` – (path, label).
    dest_dir      : Parent folder that already holds the single‑channel
                    variants.  A `discordants/` sub‑folder will be created
                    next to them.

    Returns
    -------
    List[Path] – paths to the newly created stereo discordant wavs.
    """
    variants = [v for v in variants if v]  # filter out failed ones
    if len(variants) < 2:
        # nothing to permute
        return []

    disc_dir = Path(dest_dir).resolve() / "discordants"
    disc_dir.mkdir(parents=True, exist_ok=True)

    out_paths: List[Path] = []

    for (left_path, left_lbl), (right_path, right_lbl) in permutations(variants, 2):
        left_seg = AudioSegment.from_wav(left_path)
        right_seg = AudioSegment.from_wav(right_path)

        # ensure equal length
        if len(left_seg) != len(right_seg):
            min_len = min(len(left_seg), len(right_seg))
            left_seg = left_seg[:min_len]
            right_seg = right_seg[:min_len]

        stereo = AudioSegment.from_mono_audiosegments(left_seg, right_seg)
        out_name = f"cvc_variant_{vowel_phoneme}_discordant_{left_lbl}_{right_lbl}.wav"
        out_path = disc_dir / out_name
        stereo.export(out_path, format="wav")
        out_paths.append(out_path)

    return out_paths

# ────────────────────────────────────────────────────────────────────────────
# 3.  generate_cvc_dataset – complete pipeline
# ────────────────────────────────────────────────────────────────────────────

def generate_cvc_dataset(
    cvc_items: Sequence[Tuple[str, str]],
    vowel_targets: Dict[str, Sequence[Dict[str, float]]],
    vowel_length=None,
    *,
    tts: PhonemeTTSEngine,
    base_tmp_dir: Path | str = "./_tmp_cvc",
    output_root: Path | str = "./cvc_dataset",
    pad_ms: Tuple[float, float] | float | None = None,
    tolerance: float = 50.0,
    max_iters: int = 40,
    crossfade_time: int = 3,
    max_percent_distance=0.03,
    **generate_split_kwargs,
) -> None:
    """End‑to‑end generation of a labelled CVC dataset **plus discordants**.

    Adds parameters for formant tuning precision (tolerance), iteration cap,
    and audio recombination crossfade duration.
    """
    base_tmp_dir = Path(base_tmp_dir).resolve()
    output_root = Path(output_root).resolve()
    base_tmp_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    for phoneme_str, word in cvc_items:
        slug = _sanitize(word)
        work_dir = base_tmp_dir / slug
        work_dir.mkdir(parents=True, exist_ok=True)

        # 1. synthesize base word
        raw_cvc_path = work_dir / f"{slug}_base.wav"
        tts.speak_phonemes(phoneme_str, output_path=str(raw_cvc_path))

        # 2. split into C1 / V / C2
        split_args = dict(
            wav_path=str(raw_cvc_path),
            transcript=word,
            out_dir=str(work_dir),
            **generate_split_kwargs,
        )
        if pad_ms is not None:
            split_args["pad_ms"] = pad_ms

        c1_path, v_path, c2_path = generate_splitAudio(**split_args)

        # 3. get vowel and variants
        vowel = _extract_vowel(phoneme_str)
        specs = vowel_targets.get(vowel, [])
        if not specs:
            print(f"[INFO] No formant targets provided for vowel '{vowel}' – skipping variants.")
            continue

        dest_dir = output_root / slug
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Copy reference chunks
        for p in (c1_path, v_path, c2_path):
            shutil.copy2(p, dest_dir / p.name)

        variant_pairs = generate_vowel_variants(
            vowel_wav=v_path,
            c1_path=c1_path,
            c2_path=c2_path,
            vowel_phoneme=vowel,
            targets=specs,
            dest_dir=dest_dir,
            tolerance=tolerance,
            max_iters=max_iters,
            vowel_length=vowel_length,
            crossfade_time=crossfade_time,
            max_percent_distance=max_percent_distance
        )

        # 4. Create discordant stereo pairs
        generate_discordant_pairs(
            vowel_phoneme=vowel,
            variants=variant_pairs,
            dest_dir=dest_dir,
        )


# ────────────────────────────────────────────────────────────────────────────
# CLI quick‑start (optional) — unchanged except for sample targets
# ────────────────────────────────────────────────────────────────────────────


def generate_vowel_target_list(vowel_df, target_vowels, f0=None, f3=None, f4=None, group='overall'):
    """
    Generate a list of vowel targets using f1 and f2 values from vowel_df
    and optional fixed values for f0, f3, and f4.

    Args:
        vowel_df (pd.DataFrame): DataFrame with vowel formant statistics.
        target_vowels (list): List of vowel labels (e.g., ["IY", "EH"]).
        f0 (float, optional): Fixed F0 value.
        f3 (float, optional): Fixed F3 value.
        f4 (float, optional): Fixed F4 value.
        group (str): Group name (default = 'overall').

    Returns:
        List[Dict]: List of target dictionaries with label, f1, f2,
                    and optionally f0, f3, f4.
    """
    target_list = []

    for vowel in target_vowels:
        row = vowel_df[(vowel_df['vowel'] == vowel) & (vowel_df['group'] == group)]
        if not row.empty:
            f1 = row['f1_mean'].values[0]
            f2 = row['f2_mean'].values[0]
            target = {
                "label": vowel,
                "f1": f1,
                "f2": f2
            }
            if f0 is not None:
                target["f0"] = f0
            if f3 is not None:
                target["f3"] = f3
            if f4 is not None:
                target["f4"] = f4
            target_list.append(target)
        else:
            print(f"Warning: Vowel '{vowel}' not found in group '{group}'.")

    return target_list



def interpolate_formant_targets(point_a, point_b, n_subdivisions):
    """
    Linearly interpolate between two formant-space points.

    Args:
        point_a (Dict): {"label": str, "f1": float, "f2": float, ...}
        point_b (Dict): same keys as point_a.
        n_subdivisions (int): Number of equally-spaced interior points to create.

    Returns:
        List[Dict]: [{label, f1, f2, ...}, …] ordered A → B.
    """
    if n_subdivisions < 0:
        raise ValueError("n_subdivisions must be >= 0")

    # Determine which keys to interpolate
    common_keys = set(point_a.keys()) & set(point_b.keys()) - {"label"}
    targets = [point_a.copy()]

    for k in range(1, n_subdivisions + 1):
        t = k / (n_subdivisions + 1)
        frac_a, frac_b = 1 - t, t

        label = (
            f"|{point_a['label']}_{int(round(frac_a * 100))}_"
            f"{point_b['label']}_{int(round(frac_b * 100))}|"
        )

        interpolated = {"label": label}
        for key in common_keys:
            interpolated[key] = frac_a * point_a[key] + frac_b * point_b[key]

        targets.append(interpolated)

    targets.append(point_b.copy())
    return targets






def _coerce_pad_ms(val: Any) -> Tuple[float, float] | float | None:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, (list, tuple)) and len(val) == 2:
        return (float(val[0]), float(val[1]))
    raise TypeError("pad_ms must be number, 2-element list/tuple, or null.")


def _expand_vowel_targets_if_needed(cfg: dict) -> Dict[str, Sequence[Dict[str, float]]]:
    """
    Create full vowel-to-target mapping from `targets` + `vowel_targets`.

    - `cfg["targets"]` should be a list like ["EH", "AE"]
    - `cfg["vowel_targets"]` contains f1/f2/formant specs with labels (e.g., IH, AE)
    - `cfg["f0f3f4"]` is merged into any target that lacks those fields

    Returns:
        Dict[str, List[Dict]] – mapping like { "EH": [ {...}, {...} ], ... }
    """
    requested_targets = cfg.get("targets")
    vowel_specs = cfg.get("vowel_targets") or []
    global_ff = cfg.get("f0f3f4") or {}

    if not requested_targets or not isinstance(requested_targets, list):
        return {}

    # Index vowel_targets by label for lookup
    label_map = {row["label"]: row for row in vowel_specs if "label" in row}

    out = {}
    for vowel in requested_targets:
        specs = []

        for label, row in label_map.items():
            # copy row so we don't modify original
            spec = {
                "label": label,
                "f1": row["f1"],
                "f2": row["f2"],
            }

            for k in ("f0", "f3", "f4"):
                if k in row:
                    spec[k] = row[k]
                elif k in global_ff:
                    spec[k] = global_ff[k]

            specs.append(spec)

        out[vowel.upper()] = specs

    return out



def _coerce_items(val: Any) -> Sequence[Tuple[str, str]]:
    if not isinstance(val, (list, tuple)):
        raise TypeError("items must be a list of [phoneme_str, word] pairs.")
    out = []
    for row in val:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ValueError(f"Bad item row: {row!r}")
        ph, wd = row
        out.append((str(ph), str(wd)))
    return out


def load_cvc_runs(json_data_or_path: Union[str, Path, list]) -> List[dict]:
    """
    Load and normalize a list of run configs.
    """
    if isinstance(json_data_or_path, (str, Path)):
        with open(json_data_or_path, "r", encoding="utf8") as f:
            raw = json.load(f)
    else:
        raw = json_data_or_path

    if not isinstance(raw, list):
        raise ValueError("Top-level JSON must be a list of run objects.")

    runs = []
    for i, cfg in enumerate(raw):
        if not isinstance(cfg, dict):
            raise ValueError(f"Run #{i} is not an object.")

        if "output_root" not in cfg:
            raise ValueError(f"Run #{i} missing required 'output_root'.")

        if "items" not in cfg:
            raise ValueError(f"Run #{i} missing required 'items'.")

        cfg_norm = dict(cfg)  # shallow copy

        cfg_norm["items"] = _coerce_items(cfg["items"])
        cfg_norm["pad_ms"] = _coerce_pad_ms(cfg.get("pad_ms"))
        cfg_norm["crossfade_time"] = int(cfg.get("crossfade_time", 3))
        cfg_norm["tolerance"] = float(cfg.get("tolerance", 50.0))
        cfg_norm["max_percent_distance"] = float(cfg.get("max_percent_distance", 0.03))
        if "vowel_length" in cfg_norm and cfg_norm["vowel_length"] is not None:
            cfg_norm["vowel_length"] = float(cfg_norm["vowel_length"])
        else:
            cfg_norm["vowel_length"] = None

        # build targets (handles precedence)
        cfg_norm["targets"] = _expand_vowel_targets_if_needed(cfg_norm)

        runs.append(cfg_norm)

    return runs


def run_cvc_from_configs(
    runs: List[dict],
    *,
    api_key: str,
    default_voice_id: str,
    base_tmp_dir: Path | str = "./_tmp_cvc",
    **generate_split_kwargs,
) -> None:
    """
    Execute all runs. Each run may override `voice_id`.
    """
    for r in runs:
        voice_id = r.get("voice_id", default_voice_id)
        tts_engine = PhonemeTTSEngine(api_key=api_key, voice_id=voice_id)

        generate_cvc_dataset(
            cvc_items=r["items"],
            vowel_targets=r["targets"],        # already normalized mapping
            vowel_length=r["vowel_length"],
            tts=tts_engine,
            base_tmp_dir=base_tmp_dir,
            output_root=r["output_root"],
            pad_ms=r["pad_ms"],
            tolerance=r["tolerance"],
            max_iters=int(r.get("max_iters", 850)),  # allow override though not numbered; fallback 40
            crossfade_time=r["crossfade_time"],
            max_percent_distance=r["max_percent_distance"],
            **generate_split_kwargs,
        )


# ────────────────────────────────────────────────
# 2.  Main generation loop
# ────────────────────────────────────────────────
if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())

    import os

    # HARD-CODED PATHS – adjust as needed
    CONFIG_PATH = "targets.json"         # Path to your JSON config
    TMP_DIR = "./_tmp_cvc"                # Temp folder for intermediate files

    # Get API key and fallback/default voice
    api_key = os.environ["ELEVENLABS_API_KEY"]
    default_voice_id = os.environ["ELEVENLABS_VOICE_ID"]

    # Load all run configs from JSON file
    runs = load_cvc_runs(CONFIG_PATH)

    # Run each config using the shared API key
    run_cvc_from_configs(
        runs,
        api_key=api_key,
        default_voice_id=default_voice_id,
        base_tmp_dir=TMP_DIR,
    )

    print("All runs complete.")
