#!/usr/bin/env python3
"""
cvc_variant_search.py  –  multithreaded TTS sampling (8 threads)
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path
import concurrent.futures, math, os, shutil, tempfile, uuid
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Iterable
import numpy as np
import parselmouth
import soundfile as sf
from elevenlabs.core import ApiError
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
# local helper modules
from src.speechGeneration.CVCSplitter import generate_splitAudio, generate_convex_segment_set, split_cvc_audio, \
    segment_vowels
from src.speechGeneration.elevenAudioGeneration import PhonemeTTSEngine
import os
import concurrent.futures as _cf

os.environ["MFA_DISABLE_HISTORY"] = "true"
# ───────── utility helpers ───────────────────────────────────────────
def wav_duration(path: Path) -> float:
    with sf.SoundFile(path) as f:
        return len(f) / f.samplerate

def compute_trimmed_f0(f0_values, lower=20, upper=80, max_f0=280):
    f0_values = np.array(f0_values)
    f0_values = f0_values[~np.isnan(f0_values)]
    f0_values = f0_values[f0_values <= max_f0]
    if len(f0_values) == 0:
        return None, None
    q1, q3 = np.percentile(f0_values, [lower, upper])
    core = f0_values[(f0_values >= q1) & (f0_values <= q3)]

    if len(core) == 0:
        core = np.sort(f0_values)[:2]
    return float(core.mean()), float(core.std(ddof=0))

def mean_middle_percent(values: np.ndarray, middle_percent=0.80) -> float:
    if len(values) == 0:
        return math.nan
    trim = int(round((1 - middle_percent) / 2 * len(values)))
    return float(values[trim : len(values) - trim].mean())

def analyse_formants(wav_path: Path):
    snd   = parselmouth.Sound(str(wav_path))
    pitch = snd.to_pitch(time_step=0.001, pitch_floor=75, pitch_ceiling=280)
    f0_track = pitch.selected_array["frequency"]; f0_track = f0_track[f0_track > 0]
    mu_f0, sd_f0 = compute_trimmed_f0(f0_track)

    # Replace None with NaN (or a neutral default)
    if mu_f0 is None:
        mu_f0 = float("nan")
    if sd_f0 is None:
        sd_f0 = float("nan")

    form = snd.to_formant_burg(0.0025, 5, 5500, 0.025, 50)
    tracks = [[] for _ in range(5)]
    for t in np.linspace(0, snd.duration, int(snd.duration / 0.01) + 1):
        for ch in range(5):
            val = form.get_value_at_time(ch + 1, t)
            if not math.isnan(val):
                tracks[ch].append(val)
    mu_f1 = mean_middle_percent(np.array(tracks[0]))
    mu_f2 = mean_middle_percent(np.array(tracks[1]))

    return {"mean_f0": mu_f0, "sd_f0": sd_f0, "mean_f1": mu_f1, "mean_f2": mu_f2}


def cost(stats, tgt_f0, tgt_f1, tgt_f2, wav_len, tgt_len, len_weight):
    # Handle missing F0
    if stats["mean_f0"] is None or math.isnan(stats["mean_f0"]):
        return float("inf")  # Worst possible score
    if wav_len<tgt_len*0.8:
        return float("inf")
    if tgt_f0<120 or tgt_f0>150:
        return float("inf")
    return (abs(tgt_f0 - stats["mean_f0"]) ** 2
          + (1 + (stats["sd_f0"] or 0)) ** 2 * 4
          + abs(tgt_f1 - stats["mean_f1"]) * 2
          + abs(tgt_f2 - stats["mean_f2"]) * 2
          + abs(tgt_len - wav_len) ** 2 * len_weight)


# ───────── one trial (worker) ────────────────────────────────────────
def run_trial(args: tuple) -> Tuple[str, float, dict, Path]:
    """Return (label, score, stats, wav_path) for this trial."""
    # unique filename avoids clashes
    (idx, phonemes, plaintext, api_key, voice_id, speed_factor,
     tmp_root, tgt_f0, tgt_f1, tgt_f2, tgt_len, len_weight) = args
    raw_wav = tmp_root / f"take_{idx}_{uuid.uuid4().hex}.wav"

    # new client per thread → thread‑safe
    tts = PhonemeTTSEngine(api_key=api_key, voice_id=voice_id, stability=0.4,speed=random.uniform(0.7,0.9) )
    tts.speak_phonemes(phonemes, output_path=str(raw_wav), speed_factor=speed_factor)

    # split to get vowel only
    _, v_path, _ = generate_splitAudio(wav_path=raw_wav, transcript=plaintext, out_dir=tmp_root,noise_buffer_duration=0)
    v_len = wav_duration(v_path)

    stats = analyse_formants(v_path)
    score = cost(stats, tgt_f0, tgt_f1, tgt_f2, v_len, tgt_len, len_weight)
    return (f"take_{idx}", score, stats, v_len, raw_wav)

# ───────── master routine ────────────────────────────────────────────
def run_search(
    *,
    phonemes: str,
    plaintext: str,
    api_key: str,
    voice_id: str,
    target_f0: float,
    target_f1: float,
    target_f2: float,
    target_vlen: float = 0.22,
    len_weight : float = 20.0,
    n_trials: int = 32,
    speed_factor: float = 1.0,
    n_threads: int = 8,
) -> None:

    console = Console()
    console.rule("[bold]CVC Variant Search – multithreaded")

    tmp_root = Path(tempfile.mkdtemp(prefix="cvc_search_"))
    console.print(f"Generating {n_trials} trials with {n_threads} threads…")

    # multithread pool
    with ProcessPoolExecutor(max_workers=n_threads) as pool:
        arg_tuples = [
            (i, phonemes, plaintext, api_key, voice_id, speed_factor,
             tmp_root, target_f0, target_f1, target_f2,
             target_vlen, len_weight)
            for i in range(1, n_trials + 1)
        ]
        futures = [pool.submit(run_trial, a) for a in arg_tuples]


        results: List[Tuple[str, float, dict, Path]] = []
        for fut in tqdm(as_completed(futures), total=n_trials):
            try:
                res = fut.result()
                results.append(res)
                console.print(
                                        f"{res[0]} finished → cost {res[1]:.1f} | "
                     f"V‑len {res[3]:.3f}s  μF0 {res[2]['mean_f0']:.1f}"
                    )
            except Exception as e:
                console.print(f"[red]Worker failed:[/] {e}")

    if not results:
        console.print("[red]All trials failed.")
        return

    results.sort(key=lambda r: r[1])
    best_label, best_score, best_stats, best_vlen, best_path = results[0]

    # table
    tbl = Table(title="Top 10 trials")
    tbl.add_column("#", justify="right")
    tbl.add_column("Cost", justify="right")
    tbl.add_column("V‑len (s)", justify="right")
    tbl.add_column("μF0 / σF0")
    tbl.add_column("μF1")
    tbl.add_column("μF2")
    for rank, (lbl, sc, st, vlen, _) in enumerate(results[:10], start=1):
        style = "bold green" if rank == 1 else ""
        tbl.add_row(
            str(rank),
            f"{sc:6.1f}",
            f"{vlen:.3f}",
            f"{st['mean_f0']:.1f} / {st['sd_f0']:.1f}",
            f"{st['mean_f1']:.1f}",
            f"{st['mean_f2']:.1f}",
            style=style
        )
    console.print(tbl)
    shutil.copy2(best_path, Path("best_variant.wav"))
    console.print(
            f"[bold green]Best take:[/] cost {best_score:.1f}, "
            f"V‑len {best_vlen:.3f}s  →  best_variant.wav"
    )


# ---------------------------------------------------------------------
# 0. tiny utility
# ---------------------------------------------------------------------

def wav_duration(path: Path) -> float:
    """Quick duration in seconds, no whole file load."""
    with sf.SoundFile(path) as f:
        return len(f) / f.samplerate


# ---------------------------------------------------------------------
# 1. Phase 1 – TTS synthesis only (no MFA)
# ---------------------------------------------------------------------
MAX_RETRIES = 3
def _worker_generate(
    idx: int,
    phonemes: str,
    plaintext: str,
    api_key: str,
    voice_id: str,
    speed: float,
    tmp_root: Path,
) -> tuple[Path, str]:
    wav_path = tmp_root / f"take_{idx}_{uuid.uuid4().hex}.wav"
    for attempt in range(MAX_RETRIES):
        try:
            PhonemeTTSEngine(api_key, voice_id,
                        stability=0.4, speed=speed
            ).speak_phonemes(phonemes, output_path=str(wav_path))
            return wav_path, plaintext         # ✅ success
        except Exception as e:
            print(e)
            if attempt < MAX_RETRIES - 1:
                backoff = 2 ** attempt
                time.sleep(backoff)            # exponential back‑off
    return None,plaintext


def _worker_generate_star(args):
    """Unpack the tuple so we can use executor.map without a lambda."""
    return _worker_generate(*args)

def synthesize_trials(
    *,
    n_trials: int,
    n_threads: int,
    phonemes: str,
    plaintext: str,
    api_key: str,
    voice_id: str,
    speed_range: tuple[float, float] = (0.7, 0.9),
    tmp_root: Path | None = None,
) -> tuple[list[Path], list[str], Path]:
    """Return lists of WAV paths and their transcripts (same order)."""
    tmp_root = Path(tmp_root or tempfile.mkdtemp(prefix="cvc_search_"))
    args = [
        (
            i,
            phonemes,
            plaintext,
            api_key,
            voice_id,
            random.uniform(*speed_range),
            tmp_root,
        )
        for i in range(1, n_trials + 1)
    ]
    wav_paths, transcripts = [], []
    with _cf.ProcessPoolExecutor(max_workers=n_threads) as pool:
        for p, t in tqdm(
            pool.map(_worker_generate_star, args),
            total=n_trials,
            desc=f"生成音素序列 {phonemes}",
            unit="音當",
            position=0,
            leave=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        ):
            if p:
                wav_paths.append(p)
                transcripts.append(t)
    return wav_paths, transcripts, tmp_root



# ---------------------------------------------------------------------
# 2. Phase 2 – *single* batch MFA alignment
# ---------------------------------------------------------------------


def run_mfa_align(
        corpus_path: str,
        output_path: str,
        dictionary: str = "english_us_arpa",
        acoustic_model: str = "english_us_arpa",
        num_jobs: int = 5
):
    """
    运行 MFA 对齐，带有错误处理
    """
    # 确保目录存在
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 构建 MFA 命令
        cmd = [
            "mfa",
            "align",
            corpus_path,
            dictionary,
            acoustic_model,
            str(output_dir),
            "--clean",
            "--output_format", "short_textgrid",
            "--single_speaker",
            "--beam", "15",
            "--retry_beam", "80",
            "--num_jobs", str(num_jobs)
        ]

        # 运行命令并捕获输出
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        return True, result.stdout

    except subprocess.CalledProcessError as e:
        return False, f"MFA 运行错误: {e.stderr}"
    except Exception as e:
        return False, f"发生错误: {str(e)}"


def batch_align(
    wav_paths: list[Path],
    transcripts: list[str],
    *,
    dictionary: str = "english_us_arpa",
    acoustic_model: str = "english_us_arpa",
    num_jobs: int = 8,
) -> Path:
    """Run one MFA alignment over *all* wav/text pairs.

    Returns the directory containing one TextGrid per wav (same basename).
    """
    if len(wav_paths) != len(transcripts):
        raise ValueError("wav_paths and transcripts length mismatch")

    corpus_dir = Path(tempfile.mkdtemp(prefix="mfa_corpus_"))
    output_dir = Path(tempfile.mkdtemp(prefix="mfa_out_"))

    for wav, txt in zip(wav_paths, transcripts):
        shutil.copy2(wav, corpus_dir / wav.name)
        (corpus_dir / wav.with_suffix(".lab").name).write_text(txt)

    run_mfa_align(corpus_path=corpus_dir,output_path=output_dir,dictionary=dictionary,acoustic_model=acoustic_model,num_jobs=num_jobs)


    return output_dir


# ---------------------------------------------------------------------
# 3. Per‑file vowel extraction + scoring helper
# ---------------------------------------------------------------------

def analyse_and_score(
    wav_path: Path,
    tg_path: Path,
    *,
    vowels: Iterable[str],
    target_f0: float,
    target_f1: float,
    target_f2: float,
    target_vlen: float,
    len_weight: float,
    work_dir: Path,
) -> tuple[str, float, dict, float, Path]:
    """Return (label, score, stats, vowel_len, wav_path)."""

    # 1. extract vowel span ------------------------------------------------
    spans = segment_vowels(wav_path, vowels, tg_path=tg_path)
    v_span = generate_convex_segment_set(spans)

    # 2️⃣ slice the wav
    _, v_path, _ = split_cvc_audio(
        wav_path,
        v_span,  # <-- the span we just built
        out_dir=work_dir,
        prefix=wav_path.stem + "_tmp",
        noise_buffer_duration=0.0,
        add_noise_buffer=False,
    )
    try:
        v_len = wav_duration(v_path)
    except RuntimeError:
        # wav file is unreadable – treat as a failed take
        return wav_path.stem, float("inf"), {}, 0.0, wav_path

    # Bail out if the slice is silent/empty
    if v_len == 0:
        return wav_path.stem, float("inf"), {}, 0.0, wav_path

    # 3️⃣ acoustic measurements
    try:
        stats = analyse_formants(v_path)
    except Exception as e:      # ← catches “Audio file contains 0 samples”
        return wav_path.stem, float("inf"), {}, v_len, wav_path

    # 4. cost --------------------------------------------------------------
    sc = cost(stats, target_f0, target_f1, target_f2, v_len, target_vlen, len_weight)
    return wav_path.stem, sc, stats, v_len, wav_path


# ---------------------------------------------------------------------
# 4. High‑level batch search (drop‑in replacement)
# ---------------------------------------------------------------------

def run_search_batch(
    *,
    phonemes: str,
    plaintext: str,
    api_key: str,
    voice_id: str,
    target_f0: float,
    target_f1: float,
    target_f2: float,
    target_vlen: float = 0.22,
    len_weight: float = 20.0,
    n_trials: int = 32,
    n_threads: int = 8,
    speed_range: tuple[float, float] = (0.7, 0.9),
    vowels: List[str] | None = None,
):
    """Run CVC variant search with *one* MFA call for all trials."""
    if vowels is None:
        vowels = [
            "AA",
            "AE",
            "AH",
            "AO",
            "AW",
            "AY",
            "EH",
            "ER",
            "EY",
            "IH",
            "IY",
            "OW",
            "OY",
            "UH",
            "UW",
        ]

    # 1. synthesis ---------------------------------------------------------
    wav_paths, transcripts, tmp_root = synthesize_trials(
        n_trials=n_trials,
        n_threads=n_threads,
        phonemes=phonemes,
        plaintext=plaintext,
        api_key=api_key,
        voice_id=voice_id,
        speed_range=speed_range,
    )

    # 2. batch alignment ---------------------------------------------------
    aligned_dir = batch_align(
        wav_paths,
        transcripts,
        dictionary="english_us_arpa",
        acoustic_model="english_us_arpa",
        num_jobs=n_threads,
    )

    # 3. analysis & scoring ------------------------------------------------
    results = []
    args = [
        (
            wav,
            aligned_dir / f"{wav.stem}.TextGrid",
            vowels,
            target_f0,
            target_f1,
            target_f2,
            target_vlen,
            len_weight,
            tmp_root,
        )
        for wav in wav_paths
    ]

    with _cf.ProcessPoolExecutor(max_workers=n_threads) as pool:
        futures = [
            pool.submit(
                analyse_and_score,
                a[0],
                a[1],
                vowels=a[2],
                target_f0=a[3],
                target_f1=a[4],
                target_f2=a[5],
                target_vlen=a[6],
                len_weight=a[7],
                work_dir=a[8],
            )
            for a in args
        ]
        for fut in as_completed(futures):
            try:
                res = fut.result()
                results.append(res)
            except Exception as e:
                print(f"[WARN] worker failed: {e}")

    # drop any ∞‑loss trials
    results = [r for r in results if math.isfinite(r[1])]
    if not results:
        print("[ERROR] All trials failed.")
        return

    # sort by cost --------------------------------------------------------
    results.sort(key=lambda r: r[1])

    # sort by cost --------------------------------------------------------
    results.sort(key=lambda r: r[1])

    # build rich table for top 10 ----------------------------------------
    console = Console()
    console.rule("[bold]Top 10 Trials")

    tbl = Table(show_header=True, header_style="bold magenta")
    tbl.add_column("Rank", justify="right")
    tbl.add_column("Label", justify="left")
    tbl.add_column("Loss", justify="right")
    tbl.add_column("Vowel Len (s)", justify="right")
    tbl.add_column("Mean F0", justify="right")
    tbl.add_column("SD F0", justify="right")
    tbl.add_column("F1", justify="right")
    tbl.add_column("F2", justify="right")

    for rank, (lbl, sc, st, vlen, _) in enumerate(results[:10], 1):
        tbl.add_row(
            str(rank),
            lbl,
            f"{sc:6.1f}",
            f"{vlen:.3f}",
            f"{st['mean_f0']:.1f}",
            f"{st['sd_f0']:.1f}",
            f"{st['mean_f1']:.1f}",
            f"{st['mean_f2']:.1f}",
        )

    console.print(tbl)

    # copy best wav -------------------------------------------------------
    best_path = results[0][-1]
    shutil.copy2(best_path, Path("best_variant.wav"))
    console.print(f"\n[bold green]Best take copied to best_variant.wav[/]")
    # copy best wav -------------------------------------------------------
    best_path = results[0][-1]
    shutil.copy2(best_path, Path("best_variant.wav"))
    print("\nBest take copied to best_variant.wav")
def _run_search_batch_patched(*, output_path: str | Path | None = None, **kwargs):
    """Wrapper around the original *run_search_batch*.

    Accepts an **output_path** argument; after the search finishes, copies the
    best file to that destination.  If **output_path** is *None* the original
    behaviour (writing *best_variant.wav* next to the working dir) is kept.
    Returns the *Path* of the best take.
    """
    # Call the original function. It will already copy the wav to
    # *best_variant.wav* in the current working directory – we'll respect that
    # and move/duplicate afterwards.
    run_search_batch(**kwargs)

    best_local = Path("best_variant.wav")
    if not best_local.exists():
        raise FileNotFoundError("best_variant.wav not produced by run_search_batch")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_local, output_path)
        print(f"[INFO] Copied best take to {output_path}")

    return best_local if output_path is None else Path(output_path)



# If someone "from cvc_variant_search import run_search_batch" after this file
# is imported **it will receive the patched version** because the symbol in the
# module has been replaced.

# ──────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ──────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────
# 2. HELPER – build an orthographic label from phonemes
# ──────────────────────────────────────────────────────────────────────

def spell_out(c1: str, vowel: str, c2: str) -> str:
    """Very naïve phoneme→grapheme mapping used purely for file names."""
    # Map the three vowels to a rough English vowel letter
    vowel_letter = {
        "AE": "a",  # e.g. *cat*
        "EH": "e",  # e.g. *bed*
        "IH": "i",  # e.g. *sit*
    }[vowel]

    # Join phoneme symbols in lower‑case – you can replace this with a more
    # sophisticated mapping if needed.
    return f"{c1.lower()}{vowel_letter}{c2.lower()}"

# ──────────────────────────────────────────────────────────────────────
# 3. MAIN BATCH LOOP
# ──────────────────────────────────────────────────────────────────────

def main(SEARCH_KWARGS):
    for c1, c2_list in tqdm(C1_C2_MAP.items(),desc=f"所有的資料"):
        for c2 in tqdm(c2_list,desc=f"篩選答案"):
            for vowel, formants in VOWEL_TARGETS.items():
                phonemes = f"{c1} {vowel} {c2}"
                word     = spell_out(c1, vowel, c2)

                # Build output path ./AI_Words/<C1>/<C2>/<word>_base.wav
                out_dir  = Path("AI_Words") / c1 / c2
                out_dir.mkdir(parents=True, exist_ok=True)
                dst_wav  = out_dir / f"{word}_base.wav"

                print("\n────────────────────────────────────────────")
                print(f"Synthesising {phonemes} → {dst_wav}")

                _run_search_batch_patched(
                    phonemes   = phonemes,
                    plaintext  = "dad",
                    target_f1  = formants["f1"],
                    target_f2  = formants["f2"],
                    output_path= dst_wav,
                    **SEARCH_KWARGS,
                )

if __name__ == "__main__":
    # Target formant values for each vowel
    VOWEL_TARGETS = {
        "AE": {"f1": 741.2, "f2": 1664.3},
        "EH": {"f1": 589.7, "f2": 1792.8},
        "IH": {"f1": 438.1, "f2": 1921.4},
    }

    # C₁→C₂ mapping (spaces optional after commas in the original note)
    C1_C2_MAP = {
        "JH": ["G", "T", "SH", "TH", "JH"],
        "G": ["G", "T", "TH"],
        "L": ["P", "JH", "D", "DH", "TH", "Z"],
        "D": ["SH", "F", "JH", "D", "Z"],
    }

    API_KEY = "sk_021896141aee88cc1eca629bdcc92043066224cb9a5f98d5"
    VOICE_ID = "b3tuFWghbXYRa9Cs9MJf"

    # Global search hyper‑parameters
    SEARCH_KWARGS = dict(
        api_key=API_KEY,
        voice_id=VOICE_ID,
        target_f0=130.0,
        target_vlen=0.2,
        len_weight=100.0,
        n_trials=450,
        n_threads=60,
        speed_range=(0.7, 1.15),
    )
    main(SEARCH_KWARGS)


