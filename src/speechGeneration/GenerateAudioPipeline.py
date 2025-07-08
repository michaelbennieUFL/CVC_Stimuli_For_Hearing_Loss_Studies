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

import os
import shutil
import tempfile
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from pydub import AudioSegment

from CVCSplitter import generate_splitAudio, recombine_cvc_audio
from elevenAudioGeneration import PhonemeTTSEngine
from modifyFormants import tune_formants
from src.analyzeAmEVowels.parseVowelDistFile import parse_vowel_dist_data



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
    *,
    tts: PhonemeTTSEngine,
    base_tmp_dir: Path | str = "./_tmp_cvc",
    output_root: Path | str = "./cvc_dataset",
    pad_ms: Tuple[float, float] | float | None = None,
    **generate_split_kwargs,
) -> None:
    """End‑to‑end generation of a labelled CVC dataset **plus discordants**.

    Everything works exactly as before, but now – after producing the tuned
    vowel variants – we automatically call `generate_discordant_pairs()` so
    that every left‑/right‑channel permutation is exported.
    """
    base_tmp_dir = Path(base_tmp_dir).resolve()
    output_root = Path(output_root).resolve()
    base_tmp_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    for phoneme_str, word in cvc_items:
        slug = _sanitize(word)
        work_dir = base_tmp_dir / slug
        work_dir.mkdir(parents=True, exist_ok=True)

        # ————————————————— 1. synthesise base CVC via ElevenLabs —————————————————
        raw_cvc_path = work_dir / f"{slug}_base.wav"
        tts.speak_phonemes(phoneme_str, output_path=str(raw_cvc_path))

        # ————————————————— 2. segment into C1 / V / C2 ————————————————————
        if pad_ms is None:
            c1_path, v_path, c2_path = generate_splitAudio(
                wav_path=str(raw_cvc_path),
                transcript=word,
                out_dir=str(work_dir),
                **generate_split_kwargs,
            )
        else:
            c1_path, v_path, c2_path = generate_splitAudio(
                wav_path=str(raw_cvc_path),
                transcript=word,
                out_dir=str(work_dir),
                pad_ms=pad_ms,
                **generate_split_kwargs,
            )

        vowel = _extract_vowel(phoneme_str)
        specs = vowel_targets.get(vowel, [])
        if not specs:
            print(f"[INFO] No formant targets provided for vowel '{vowel}' – skipping variants.")
            continue

        # ————————————————— 3. generate tuned variants —————————————————————
        dest_dir = output_root / slug
        dest_dir.mkdir(parents=True, exist_ok=True)

        # copy original segments for reference
        for p in (c1_path, v_path, c2_path):
            shutil.copy2(p, dest_dir / p.name)

        variant_pairs = generate_vowel_variants(
            vowel_wav=v_path,
            c1_path=c1_path,
            c2_path=c2_path,
            vowel_phoneme=vowel,
            targets=specs,
            dest_dir=dest_dir,
            tolerance=10,
            max_iters=800,
        )

        # ————————————————— 4. create discordant stereo pairs —————————————
        generate_discordant_pairs(
            vowel_phoneme=vowel,
            variants=variant_pairs,
            dest_dir=dest_dir,
        )

# ────────────────────────────────────────────────────────────────────────────
# CLI quick‑start (optional) — unchanged except for sample targets
# ────────────────────────────────────────────────────────────────────────────


def generate_vowel_target_list(vowel_df, target_vowels, f0, f3, f4, group='overall'):
    """
    Generate a list of vowel targets using f1 and f2 values from vowel_df
    and fixed values for f0, f3, and f4.

    Args:
        vowel_df (pd.DataFrame): DataFrame with vowel formant statistics.
        target_vowels (list): List of vowel labels (e.g., ["IY", "EH"]).
        f0 (float): Fixed F0 value.
        f3 (float): Fixed F3 value.
        f4 (float): Fixed F4 value.
        group (str): Group name (default = 'overall').

    Returns:
        List[Dict]: List of target dictionaries with label, f0, f1, f2, f3, f4.
    """
    target_list = []

    for vowel in target_vowels:
        row = vowel_df[(vowel_df['vowel'] == vowel) & (vowel_df['group'] == group)]
        if not row.empty:
            f1 = row['f1_mean'].values[0]
            f2 = row['f2_mean'].values[0]
            target_list.append({
                "label": vowel,
                "f0": f0,
                "f1": f1,
                "f2": f2,
                "f3": f3,
                "f4": f4
            })
        else:
            print(f"Warning: Vowel '{vowel}' not found in group '{group}'.")

    return target_list


if __name__ == "__main__":
    import json
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv())

    # ★ 1. configure ElevenLabs ★
    tts_engine = PhonemeTTSEngine(
        api_key=os.environ["ELEVENLABS_API_KEY"],
        voice_id=os.environ["ELEVENLABS_VOICE_ID"],
    )

    # ★ 2. define your input set + targets ★
    items = [
        ("B AE G", "bag"),
    ]
    vowel_df=parse_vowel_dist_data(file_path='../../input_data/vowel_stats.txt')

    targets = {
        "AE": generate_vowel_target_list(
            vowel_df,
            target_vowels=["IH", "EY", "IY", "EH", "AE", "AH"],
            f0=173,
            f3=3057,
            f4=3565,
            group="f"
        )
    }

    # ★ 3. run generation ★
    generate_cvc_dataset(
        cvc_items=items,
        vowel_targets=targets,
        tts=tts_engine,
        output_root="./generated_cvc",
    )

    print("Dataset written to ./generated_cvc – happy modelling! ☺")
