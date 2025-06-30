"""
minimal_mfa_helpers.py – MFA helpers, now compatible with BOTH
praatio 4.x (tgio) and praatio 5.x (textgrid).
"""

from __future__ import annotations

import math
from pathlib import Path
import re, subprocess
from typing import Iterable, Dict, List, Tuple

from praatio import textgrid as _tg_mod
import soundfile as sf
import numpy as np
from datetime import datetime
# --------------------------------------------------------------------
# 1. generate_textgrid  (unchanged)
# --------------------------------------------------------------------
# ---------- only the body of generate_textgrid() changed --------------
import os, tempfile, shutil   # add at top of file

def generate_textgrid(
    wav_path: str | Path,
    transcript: str | Path | None = None,
    *,
    dictionary: str = "english_us_arpa",
    acoustic_model: str = "english_mfa",
    tg_format: str = "short_textgrid",
    overwrite: bool = False,
) -> Path:
    wav_path = Path(wav_path).expanduser().resolve()
    tg_path  = wav_path.with_suffix(".TextGrid")
    if tg_path.exists() and not overwrite:
        return tg_path

    # 1) Pick or create a text-file that contains the transcript -------------
    tmp_file: Path | None = None
    if transcript is None:                       # look for sibling *.lab
        lab = wav_path.with_suffix(".lab")
        if not lab.exists():
            raise ValueError("No transcript string/path and no *.lab file")
        transcript_path = lab

    else:
        transcript_path = Path(transcript).expanduser()
        if not transcript_path.exists():         # you've passed raw text
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, dir=wav_path.parent
            )
            tmp.write(str(transcript_path))      # transcript_path holds the text
            tmp.close()
            tmp_file = Path(tmp.name)
            transcript_path = tmp_file           # now it *is* a path

    # 2) Call MFA ------------------------------------------------------------
    cmd = [
        "mfa", "align_one",
        str(wav_path),
        str(transcript_path),        # <-- always a real file now
        dictionary,
        acoustic_model,
        str(tg_path),                # positional OUTPUT_PATH
        "--output_format", tg_format,
        "--clean",
    ]
    subprocess.run(cmd, check=True)

    # 3) Clean up the temporary file (if we made one) ------------------------
    if tmp_file is not None and tmp_file.exists():
        tmp_file.unlink()

    return tg_path



# --------------------------------------------------------------------
# 2. segment_vowels  (tweaked for v4/v5 API differences)
# --------------------------------------------------------------------
def segment_vowels(
    wav_path: str | Path,
    phones: Iterable[str],
    *,
    tier_name: str = "phones",
    tg_path: str | Path | None = None,
    ignore_stress: bool = True,
) -> Dict[str, List[Tuple[float, float]]]:

    wav_path = Path(wav_path).expanduser().resolve()
    tg_path  = Path(tg_path).expanduser().resolve() if tg_path else wav_path.with_suffix(".TextGrid")
    if not tg_path.exists():
        raise FileNotFoundError(tg_path)

    norm = (lambda l: re.sub(r"\d$", "", l.upper()) if ignore_stress else l.upper())
    wanted = {norm(p) for p in phones}
    spans  = {p: [] for p in wanted}

    tg = _tg_mod.openTextgrid(str(tg_path), includeEmptyIntervals=False)  # same call in v4 & v5

    # ▶ NEW – phone-tier accessor that works in both versions
    if hasattr(tg, "getTier"):                     # praatio 5
        phone_tier = tg.getTier(tier_name)
        entries    = phone_tier.entries
    else:                                          # praatio 4
        phone_tier = tg.tierDict[tier_name]
        entries    = phone_tier.entryList

    for start, end, label in entries:
        lab = norm(label)
        if lab in spans:
            spans[lab].append((round(start, 6), round(end, 6)))

    return spans


# --------------------------------------------------------------------
# 3. wrapper – align *and* segment
# --------------------------------------------------------------------
def get_vowel_segments(
    wav_path: str | Path,
    vowels: Iterable[str],
    **align_kwargs,
) -> Dict[str, List[Tuple[float, float]]]:
    generate_textgrid(wav_path, **align_kwargs)
    return segment_vowels(wav_path, vowels)





# --------------------------------------------------------------------
# 4. split the aligned word into C – V – C audio files
# --------------------------------------------------------------------


def split_cvc_audio(
    wav_path: str | Path,
    vowel_span: Tuple[float, float],
    *,
    out_dir: str | Path | None = None,
    prefix: str | None = None,
    suffix: str = "",
    clip_silence: bool = False,
) -> Tuple[Path, Path, Path]:
    """
    Cut *wav_path* into three files: onset consonant(s), vowel, coda consonant(s).

    Parameters
    ----------
    wav_path      : Path to the original aligned .wav
    vowel_span    : (v_start, v_end) in **seconds**, e.g. (0.06, 0.15)
    out_dir       : Folder to save the chunks (default: same folder as wav)
    prefix        : Filename prefix (default: original stem)
    suffix        : Extra text to append before _C1/_V/_C2, e.g. '_slow'
    clip_silence  : If True, drops leading/trailing zeros from C1/C2

    Returns
    -------
    (c1_path, v_path, c2_path) – Paths to the three new WAV files.
    """
    wav_path = Path(wav_path).expanduser().resolve()
    out_dir  = Path(out_dir).expanduser().resolve() if out_dir else wav_path.parent
    prefix   = prefix or wav_path.stem

    data, sr = sf.read(wav_path)           # (samples, channels)
    v0_s, v1_s = vowel_span
    v0 = int(round(v0_s * sr))
    v1 = int(round(v1_s * sr))

    if v0 < 0 or v1 > len(data):
        raise ValueError("vowel_span outside audio bounds")

    # slice the numpy array -------------------------------------------
    c1 = data[:v0]
    v  = data[v0:v1]
    c2 = data[v1:]

    if clip_silence:
        def _trim(x):
            nz = np.nonzero(x)[0]
            return x[nz[0]:nz[-1]+1] if nz.size else x[:1]
        c1, c2 = _trim(c1), _trim(c2)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    c1_path = out_dir / f"{prefix}{suffix}_C1_{timestamp}.wav"
    v_path  = out_dir / f"{prefix}{suffix}_V_{timestamp}.wav"
    c2_path = out_dir / f"{prefix}{suffix}_C2_{timestamp}.wav"

    sf.write(c1_path, c1, sr)
    sf.write(v_path,  v,  sr)
    sf.write(c2_path, c2, sr)

    return c1_path, v_path, c2_path



def generate_convex_segment_set(segments):
    actualSpan=[math.inf,-math.inf]
    for span in segments.values():
        for segment in span:
            if segment[0]<actualSpan[0]:
                actualSpan[0]=segment[0]
            if segment[1]>actualSpan[1]:
                actualSpan[1]=segment[1]
    return actualSpan



def generate_splitAudio(wav_path="../../input_data/En-us-bag.wav",transcript="bag",out_dir="../../output_files/temp",
        dictionary="english_us_arpa",
        acoustic_model="english_us_arpa",
        pad_ms: float | Tuple[float, float] = [0.05,0],):


    if isinstance(pad_ms, (int, float)):
        pad_c1, pad_c2 = pad_ms, pad_ms
    else:
        pad_c1, pad_c2 = pad_ms            # (ms to add to C1 end, ms to add to C2 start)


    VOWELS = [
        "AA", "AE", "AH", "AO", "AW", "AY",
        "EH", "ER", "EY",
        "IH", "IY",
        "OW", "OY",
        "UH", "UW",
    ]
    spans = get_vowel_segments(
        wav_path,
        vowels=VOWELS,
        transcript=transcript,
        dictionary="english_us_arpa",
        acoustic_model="english_us_arpa",
    )

    actualSpan=generate_convex_segment_set(spans)

    print(spans)
    print(actualSpan)

    actualSpan[0]+=pad_c1
    actualSpan[1]+=pad_c2

    c1, v, c2 = split_cvc_audio(
        wav_path,
        actualSpan,
        out_dir=out_dir,
        prefix=transcript,
    )



    return c1, v, c2


# ──────────────────────────────────────────── recombine_cvc_audio ──────────────────────────────────────────
def recombine_cvc_audio(
    mod_vowel_path: str | Path,
    c1_path: str | Path,
    c2_path: str | Path,
    *,
    vowel_phoneme: str = "AE",  # ARPABET by default
    dictionary: str = "english_us_arpa",
    acoustic_model: str = None,
    tg_format: str = "short_textgrid",
    out_path: str | Path | None = None,
) -> Path:
    """
    Trim silence off a *modified* vowel file with MFA and stitch it back between C1 + C2.
    """
    mod_vowel_path = Path(mod_vowel_path).expanduser().resolve()
    c1_path = Path(c1_path).expanduser().resolve()
    c2_path = Path(c2_path).expanduser().resolve()

    if acoustic_model is None:
        acoustic_model = dictionary  # default to same phone set

    # Write the phoneme label into a temp transcript file
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False,
                                      dir=mod_vowel_path.parent)
    tmp.write(vowel_phoneme.strip())
    tmp.close()
    transcript_path = Path(tmp.name)

    tg_vowel = mod_vowel_path.with_suffix(".TextGrid")
    cmd = [
        "mfa", "align_one",
        str(mod_vowel_path),
        str(transcript_path),
        dictionary,
        acoustic_model,
        str(tg_vowel),
        "--output_format", tg_format,
        "--clean",
    ]

    try:
        subprocess.run(cmd, check=True)
    finally:
        if transcript_path.exists():
            transcript_path.unlink()

    # read alignment
    tg = _tg_mod.openTextgrid(str(tg_vowel), includeEmptyIntervals=False)
    phone_tier = tg.getTier("phones") if hasattr(tg, "getTier") else tg.tierDict["phones"]
    entry = phone_tier.entries[0] if hasattr(phone_tier, "entries") else phone_tier.entryList[0]
    v_start_s, v_end_s, _ = entry

    v_audio, sr_v = sf.read(mod_vowel_path)
    start_sample = int(round(v_start_s * sr_v))
    end_sample = int(round(v_end_s * sr_v))
    v_trimmed = v_audio[start_sample:end_sample]

    if v_trimmed.size == 0:
        raise RuntimeError("Forced aligner trimmed the vowel to zero length!")

    if c1_path.samefile(c2_path):
        raise ValueError("c1_path and c2_path are identical – did you pass the wrong file for C2?")

    c1_audio, sr_c1 = sf.read(c1_path)
    c2_audio, sr_c2 = sf.read(c2_path)

    # helper ──────────────────────────────────────────────────────────
    def _resample(audio: np.ndarray, sr_old: int, sr_new: int) -> np.ndarray:
        if sr_old == sr_new:
            return audio
        import librosa
        return librosa.resample(audio.T, orig_sr=sr_old, target_sr=sr_new).T

    c1_audio = _resample(c1_audio, sr_c1, sr_v)
    c2_audio = _resample(c2_audio, sr_c2, sr_v)

    # make every stream 2-D: (samples, channels)
    def _to_2d(a: np.ndarray) -> np.ndarray:
        return a if a.ndim == 2 else a[:, None]

    c1_audio = _to_2d(c1_audio)
    v_trimmed = _to_2d(v_trimmed)
    c2_audio = _to_2d(c2_audio)

    # if channel counts differ, pick the *largest* and replicate as needed
    n_channels = max(a.shape[1] for a in (c1_audio, v_trimmed, c2_audio))

    def _match_channels(a: np.ndarray, n_out: int) -> np.ndarray:
        if a.shape[1] == n_out:
            return a
        if a.shape[1] == 1:                 # mono → duplicate
            return np.repeat(a, n_out, axis=1)
        raise RuntimeError(f"Cannot reduce {a.shape[1]}-channel audio to {n_out} channels")

    c1_audio = _match_channels(c1_audio, n_channels)
    v_trimmed = _match_channels(v_trimmed, n_channels)
    c2_audio = _match_channels(c2_audio, n_channels)

    rebuilt = np.concatenate([c1_audio, v_trimmed, c2_audio], axis=0)

    out_path = Path(out_path).expanduser().resolve() if out_path else (
        mod_vowel_path.parent / f"{mod_vowel_path.stem}_recomb.wav"
    )
    sf.write(out_path, rebuilt, sr_v)
    return out_path




# --------------------------------------------------------------------
# example usage
# --------------------------------------------------------------------
if __name__ == "__main__":
    # os.system(f"mfa model download dictionary english_us_arpa")
    # os.system(f"mfa model download acoustic english_us_arpa ")
    # os.system(f"mfa model download dictionary english_mfa")
    # os.system(f"mfa model download acoustic english_mfa")
    c1, v, c2= generate_splitAudio(transcript="bog",wav_path="./_tmp_cvc/bog/bog_base.wav")
    print("saved:", c1, v, c2)

    c1, v_mod, c2 = Path("../../output_files/temp/bag_C1_20250627103235.wav"), Path("../../output_files/temp_result/bag_V_20250627103235_mod_wave_0.9901357065873746_1.0602099021363576_0.5357311238158247_1.0_1.0.wav"), Path("../../output_files/temp/bag_C2_20250627103235.wav")
    exit()
    final_wav = recombine_cvc_audio(
        mod_vowel_path=v_mod,
        c1_path=c1,
        c2_path=c2,
        vowel_phoneme="AW",  # match the actual vowel you’re using
        out_path="bag_rebuilt.wav",
    )
    print("Re-created CVC saved to:", final_wav)






