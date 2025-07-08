"""
minimal_mfa_helpers.py – MFA helpers, now compatible with BOTH
praatio 4.x (tgio) and praatio 5.x (textgrid).
"""

from __future__ import annotations

import math
from pathlib import Path
import re, subprocess
from typing import Iterable, Dict, List, Tuple
import pydub
from pydub import AudioSegment
import librosa
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
    add_noise_buffer: bool = True,
    noise_buffer_duration: float = 0.15,
    noise_level: float = 1e-6,
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

    if add_noise_buffer:
        # Replace v_path with a buffered version
        tmp_v_path = v_path.with_stem(v_path.stem + "_tmp")  # temporary rename
        v_path.rename(tmp_v_path)
        add_audio_buffer_with_noise(
            input_file=tmp_v_path,
            output_file=v_path,
            buffer_duration=noise_buffer_duration,
            noise_level=noise_level,
        )
        tmp_v_path.unlink()  # clean up temp

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



def add_audio_buffer_with_noise(input_file: str, output_file: str, buffer_duration: float = 0.15, noise_level: float = 1e-6):
    y, sr = librosa.load(input_file, sr=None)

    # Add low-level noise instead of zero silence
    silence = np.random.normal(scale=noise_level, size=int(buffer_duration * sr)).astype(np.float32)

    y_buffered = np.concatenate([silence, y, silence])
    sf.write(output_file, y_buffered, sr)


def generate_splitAudio(wav_path="../../input_data/En-us-bag.wav",transcript="bag",out_dir="../../output_files/temp",
        dictionary="english_us_arpa",
        acoustic_model="english_us_arpa",
        pad_ms: float | Tuple[float, float] = [0.015,-0.005],):


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
def match_rms(audio: AudioSegment, target_dBFS: float) -> AudioSegment:
    change_dBFS = target_dBFS - audio.dBFS
    return audio.apply_gain(change_dBFS)


def safe_append(audio1, audio2):
    min_duration = min(len(audio1), len(audio2))       # milliseconds
    if min_duration < 10:                 # either part shorter than 100 ms
        return audio1 + audio2             # no cross-fade
    crossfade_ms = max(6, min_duration // 10)  #  ≤100 ms,  never 0
    return audio1.append(audio2, crossfade=crossfade_ms)



def recombine_cvc_audio(
    mod_vowel_path: str | Path,
    c1_path: str | Path,
    c2_path: str | Path,
    *,
    vowel_phoneme: str = "AE",
    dictionary: str = "english_us_arpa",
    acoustic_model: str | None = None,
    tg_format: str = "short_textgrid",
    out_path: str | Path | None = None,
    remove_noise_buffer: bool = True,
    noise_buffer_duration: float = 0.15,
    vowel_length: float = 0.11,
) -> Path:
    """
    Trim (optionally) buffered noise from a modified V file and stitch it
    back between C1 and C2.

    Parameters
    ----------
    mod_vowel_path      : WAV file that contains the modified vowel
    c1_path, c2_path    : WAV files for onset and coda consonants
    vowel_phoneme       : ARPAbet label for the vowel (for MFA)
    remove_noise_buffer : If True, strips `noise_buffer_duration` seconds
                          from both start & end of the vowel before
                          recombining.
    noise_buffer_duration : Duration (seconds) of the buffer that was
                          added earlier in `split_cvc_audio()`.
    vowel_length   : length in seconds
    """
    # ---------------- paths & defaults ----------------
    p_mod  = Path(mod_vowel_path).expanduser().resolve()
    p_c1   = Path(c1_path).expanduser().resolve()
    p_c2   = Path(c2_path).expanduser().resolve()
    out_p  = Path(out_path).expanduser().resolve() if out_path else (
        p_mod.parent / f"{p_mod.stem}_recomb.wav"
    )

    audio_c1=AudioSegment.from_wav(p_c1)
    audio_v=AudioSegment.from_wav(p_mod)
    audio_c2=AudioSegment.from_wav(p_c2)


    if remove_noise_buffer:
        # Remove the noise buffer from the vowel
        length = audio_v.duration_seconds
        audio_v=audio_v[noise_buffer_duration*1000:length-noise_buffer_duration*1000]

    if vowel_length:
        actual_length = audio_v.duration_seconds
        if actual_length<vowel_length:
            raise ValueError(f"Vowel is shoter than target length: {actual_length} < {vowel_length}")

        offset = (actual_length-vowel_length)/2
        audio_v =audio_v[offset*1000:(offset+vowel_length)*1000]

    audio_cv = safe_append(audio_c1, audio_v)
    audio_cvc = safe_append(audio_cv, audio_c2)

    audio_cvc = match_rms(audio_cvc, target_dBFS=-20.0)
    audio_cvc.export(out_p, format="wav")





    return out_p





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

    v_mod=Path("./generated_cvc/bog/tuned_1/bog_V_20250701124115_wave_0.9866432752189519_0.6118647950981129_1.8780165791901258_1.0_1.0.wav")
    final_wav = recombine_cvc_audio(
        mod_vowel_path=v_mod,
        c1_path=c1,
        c2_path=c2,
        vowel_phoneme="AW",  # match the actual vowel you’re using
        out_path="bag_rebuilt.wav",
    )


    v_mod=Path("./generated_cvc/bog/tuned_0/bog_V_20250701124311_wave_0.9820402084699846_1.0633614131975866_1.0771034908956034_1.0_1.0.wav")
    final_wav = recombine_cvc_audio(
        mod_vowel_path=v_mod,
        c1_path=c1,
        c2_path=c2,
        vowel_phoneme="AW",  # match the actual vowel you’re using
        out_path="bag_rebuilt2.wav",
    )


    print("Re-created CVC saved to:", final_wav)






