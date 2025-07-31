"""
minimal_mfa_helpers.py – MFA helpers, now compatible with BOTH
praatio 4.x (tgio) and praatio 5.x (textgrid).
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
import re, subprocess
from typing import Iterable, Dict, List, Tuple, Union
import pydub
from pydub import AudioSegment
import librosa
from praatio import textgrid as _tg_mod
import soundfile as sf
import numpy as np
from datetime import datetime
import tempfile, uuid
CROSSFADE_TIME=3

# --------------------------------------------------------------------
# 1. generate_textgrid  (unchanged)
# --------------------------------------------------------------------
# ---------- only the body of generate_textgrid() changed --------------
import os, tempfile, shutil   # add at top of file


# ---------- NEW helper -------------------------------------------------
def batch_align(
        wav_paths: list[Path],
        transcripts: list[str],
        *,
        dictionary="english_us_arpa",
        acoustic_model="english_us_arpa",
        num_jobs: int = 8
) -> Path:
    """Run *one* MFA call over all wav/text pairs and return the dir with TextGrids."""
    corpus_dir   = Path(tempfile.mkdtemp(prefix="mfa_corpus_"))
    output_dir   = Path(tempfile.mkdtemp(prefix="mfa_out_"))
    for wav, txt in zip(wav_paths, transcripts):
        shutil.copy2(wav, corpus_dir / wav.name)
        (corpus_dir / wav.with_suffix(".lab").name).write_text(txt)

    subprocess.run(
        ["mfa", "align", str(corpus_dir), dictionary, acoustic_model,
         str(output_dir), "--clean", "--output_format", "short_textgrid",
         "--num_jobs", str(num_jobs)],
        check=True,
    )
    return output_dir




def generate_textgrid(wav_path, transcript=None, **kwargs):
    wav_path = Path(wav_path).resolve()

    # Use unique tmp folder per call
    mfa_tmp = Path(tempfile.mkdtemp(prefix="mfa_call_", dir="/tmp"))
    tg_path = mfa_tmp / f"{uuid.uuid4().hex}.TextGrid"

    transcript_path = Path(transcript)
    if not transcript_path.exists():
        tmp_txt = mfa_tmp / f"{uuid.uuid4().hex}.lab"
        tmp_txt.write_text(str(transcript))
        transcript_path = tmp_txt

    cmd = [
        "mfa", "align_one",
        str(wav_path),
        str(transcript_path),
        kwargs.get("dictionary", "english_us_arpa"),
        kwargs.get("acoustic_model", "english_us_arpa"),
        str(tg_path),
        "--output_format", kwargs.get("tg_format", "short_textgrid"),
        "--clean",
    ]
    subprocess.run(cmd, check=True)

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
def get_vowel_segments(wav_path: str | Path,
                        vowels: Iterable[str],
                       **align_kwargs) -> Dict[str, List[Tuple[float, float]]]:
    tg_path = generate_textgrid(wav_path, **align_kwargs)
    return segment_vowels(wav_path, vowels, tg_path=tg_path)






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

    if math.isinf(v0_s):
        v0_s = 0.0 if v0_s < 0 else len(data) / sr
    if math.isinf(v1_s):
        v1_s = 0.0 if v1_s < 0 else len(data) / sr

    v0 = int(round(v0_s * sr))
    v1 = int(round(v1_s * sr))

    if (v0 < 0 and not math.isinf(v0_s)) or (v1 > len(data) and not math.isinf(v1_s)):
        warnings.warn(
            f"vowel_span ({v0_s:.3f}s–{v1_s:.3f}s) exceeds audio bounds "
            f"(0–{len(data) / sr:.3f}s). Clipping to fit."
        )
        v0 = max(0, v0)
        v1 = min(len(data), v1)

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


import tempfile, uuid


def generate_splitAudio(wav_path, transcript="bag", out_dir=None,
                        dictionary="english_us_arpa",
                        acoustic_model="english_us_arpa",
                        pad_ms: float | Tuple[float, float] = (0.012, -0.007),
                        noise_buffer_duration: float = 0.15):
    # Create unique temp dir for MFA to avoid collisions
    worker_tmp = Path(tempfile.mkdtemp(prefix="mfa_worker_", dir="/tmp"))

    # Ensure out_dir exists
    out_dir = Path(out_dir or worker_tmp).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Give MFA a unique transcript file
    transcript_file = worker_tmp / f"{uuid.uuid4().hex}.lab"
    transcript_file.write_text(transcript)

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
        transcript=transcript_file,
        dictionary=dictionary,
        acoustic_model=acoustic_model,
    )

    actualSpan = generate_convex_segment_set(spans)

    print(spans)
    print(actualSpan)

    if isinstance(pad_ms, (int, float)):
        pad_c1, pad_c2 = pad_ms, pad_ms
    else:
        pad_c1, pad_c2 = pad_ms
    actualSpan[0] += pad_c1
    actualSpan[1] += pad_c2

    c1, v, c2 = split_cvc_audio(
        wav_path,
        actualSpan,
        out_dir=out_dir,
        prefix=Path(wav_path).stem,
        noise_buffer_duration=noise_buffer_duration
    )

    return c1, v, c2


# ──────────────────────────────────────────── recombine_cvc_audio ──────────────────────────────────────────
def match_rms(audio: AudioSegment, target_dBFS: float) -> AudioSegment:
    change_dBFS = target_dBFS - audio.dBFS
    return audio.apply_gain(change_dBFS)


def safe_append(audio1: AudioSegment, audio2: AudioSegment, crossfade_time=CROSSFADE_TIME) -> AudioSegment:
    """Append with a <100 ms cross-fade – but fall back when either part is empty."""
    if len(audio1) == 0 or len(audio2) == 0:
        return audio1 + audio2            # simple concat, no cross-fade

    min_duration = min(len(audio1), len(audio2))  # ms
    crossfade_ms = max(crossfade_time, min_duration // 200)     # 6–100 ms



    return audio1.append(audio2, crossfade=crossfade_ms)



def _match_rms(audio: AudioSegment, target_dBFS: float) -> AudioSegment:
    """Return *audio* gain‑adjusted so that its RMS equals *target_dBFS*."""
    print("Current dBFS:", audio.dBFS)
    print("Target dBFS:", target_dBFS)
    print(
        "Adjusting volume to match target RMS of",
        target_dBFS,
        "dBFS",
    )
    return audio.apply_gain(target_dBFS - audio.dBFS)


def _scale_linear(audio: AudioSegment, factor: float) -> AudioSegment:
    """
    Multiply the signal by *factor* (linear).  Internally this is a gain
    change of 20·log10(factor) dB, which works for any positive *factor*.
    """
    if factor <= 0:
        raise ValueError("vowel_volume_scaling_factor must be > 0")
    return audio.apply_gain(20 * math.log10(factor))


def _trim_vowel(
    audio: AudioSegment,
    *,
    remove_noise: bool,
    buffer_sec: float,
    target_len: float | None,
    trim_style: str,
) -> AudioSegment:
    """Remove the synthetic noise buffer and optionally centre‑trim to *target_len*."""
    if remove_noise:
        audio = audio[buffer_sec * 1000 : -buffer_sec * 1000]

    if target_len is not None:
        excess = audio.duration_seconds - target_len
        if excess < 0:
            raise ValueError(f"Vowel shorter than target length ({audio.duration_seconds:.3f}s < {target_len:.3f}s)")
        if trim_style == "keep_middle":
            start_off = excess / 2
        elif trim_style == "keep_start":
            start_off = 0.0
        elif trim_style == "keep_end":
            start_off = excess
        else:
            raise ValueError(f"Unknown trim_style '{trim_style}'")
        audio = audio[start_off * 1000 : (start_off + target_len) * 1000]

    return audio


def recombine_cvc_audio(               # ← NEW SIGNATURE
    mod_vowel_path: Union[str, Path],
    c1_path: Union[str, Path],
    c2_path: Union[str, Path],
    *,
    # ————— new arguments ———————————————————————————————————————————————————
    reference_vowel_path: Union[str, Path] | None = None,
    vowel_volume_scaling_factor: float = 1.0,
    # ————— existing arguments (unchanged defaults) ———————————————
    vowel_phoneme: str = "AE",
    dictionary: str = "english_us_arpa",
    acoustic_model: str | None = None,
    tg_format: str = "short_textgrid",
    out_path: Union[str, Path, None] = None,
    remove_noise_buffer: bool = True,
    noise_buffer_duration: float = 0.15,
    vowel_length: float | None = 0.10,
    final_buffer_duration: float | None = 0.5,
    crossfade_time: int | Tuple[int, int] = 3,
    c1_scale: float = 1.0,
    trim_style: str = "keep_middle",
) -> Path:
    """
    Trim (optionally) buffered noise from a modified V file and stitch it
    back between C1 and C2. Optionally pads the final audio with silence.

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
    final_buffer_duration : Optional float, duration (seconds) of silence to
                            add to the beginning and end of final output.
    trim_style : One of {'keep_start', 'keep_middle', 'keep_end'}, determines
                 which part of the vowel to keep when trimming.
    """
    p_mod = Path(mod_vowel_path).expanduser().resolve()
    p_c1 = Path(c1_path).expanduser().resolve()
    p_c2 = Path(c2_path).expanduser().resolve()
    out_p = Path(out_path).expanduser().resolve() if out_path else (
        p_mod.parent / f"{p_mod.stem}_recomb.wav"
    )

    audio_c1 = AudioSegment.from_wav(p_c1)

    if isinstance(crossfade_time, (tuple, list)) and len(crossfade_time) == 2:
        xfade_c1_v, xfade_v_c2 = crossfade_time
    else:
        xfade_c1_v = xfade_v_c2 = int(crossfade_time)

    # Apply C1 volume scaling
    if c1_scale != 1.0:
        if c1_scale <= 0:
            raise ValueError(f"c1_scale must be positive, got {c1_scale}")
        print("Current C1 dBFS:", audio_c1.dBFS)
        audio_c1 = _scale_linear(audio_c1, c1_scale)
        print("New C1 dBFS:", audio_c1.dBFS)


    audio_v = AudioSegment.from_wav(p_mod)
    audio_c2 = AudioSegment.from_wav(p_c2)





    ref_v = None
    if reference_vowel_path:
        ref_v = AudioSegment.from_wav(Path(reference_vowel_path).expanduser().resolve())



    audio_v = _trim_vowel(
        audio_v,
        remove_noise=remove_noise_buffer,
        buffer_sec=noise_buffer_duration,
        target_len=vowel_length,
        trim_style=trim_style,
    )
    if ref_v is not None:
        ref_v = _trim_vowel(
            ref_v,
            remove_noise=remove_noise_buffer,
            buffer_sec=noise_buffer_duration,
            target_len=vowel_length,
            trim_style=trim_style,
        )


    # ---- RMS‑match to reference (if provided) --------------------------------------
    if ref_v is not None:
        audio_v = _match_rms(audio_v, ref_v.dBFS)
        print("Current dBFS:", audio_v.dBFS)

    # ---- optional extra scaling ----------------------------------------------------
    if vowel_volume_scaling_factor != 1.0:
        audio_v = _scale_linear(audio_v, vowel_volume_scaling_factor)

    dur_c1, dur_c2 = len(audio_c1), len(audio_c2)

    if dur_c1 == 0 and dur_c2 == 0:
        audio_cvc = audio_v  # V only
    elif dur_c1 == 0:
        audio_cvc = safe_append(audio_v, audio_c2, xfade_v_c2)  # V + C2
    elif dur_c2 == 0:
        audio_cvc = safe_append(audio_c1, audio_v, xfade_c1_v)  # C1 + V
    else:
        audio_cv = safe_append(audio_c1, audio_v, xfade_c1_v)  # (C1 + V)
        audio_cvc = safe_append(audio_cv, audio_c2, xfade_v_c2)  # ... + C2

    # Add silence before and after if specified
    if final_buffer_duration is not None and final_buffer_duration > 0:
        silence = AudioSegment.silent(duration=final_buffer_duration * 1000)
        audio_cvc = silence + audio_cvc + silence

    # Normalize loudness
    audio_cvc = match_rms(audio_cvc, target_dBFS=-27.0)
    audio_cvc = audio_cvc.set_sample_width(32 // 8)
    codec_map = {16: "pcm_s16le", 24: "pcm_s24le", 32: "pcm_s32le"}
    audio_cvc.export(out_p, format="wav", codec=codec_map[32])

    return out_p







# --------------------------------------------------------------------
# example usage
# --------------------------------------------------------------------
if __name__ == "__main__":
    # os.system(f"mfa model download dictionary english_us_arpa")
    # os.system(f"mfa model download acoustic english_us_arpa ")
    # os.system(f"mfa model download dictionary english_mfa")
    # os.system(f"mfa model download acoustic english_mfa")
    c1, v, c2= generate_splitAudio(transcript="bug",wav_path="./_tmp_cvc/bug/bug_base.wav")
    print("saved:", c1, v, c2)

    final_wav = recombine_cvc_audio(
        mod_vowel_path=v,
        c1_path=c1,
        c2_path=c2,
        vowel_phoneme="AH",  # match the actual vowel you’re using
        out_path="bug_rebuilt.wav",
    )

    exit()
    v_mod=Path("./generated_cvc/bog/tuned_0/bog_V_20250701124311_wave_0.9820402084699846_1.0633614131975866_1.0771034908956034_1.0_1.0.wav")
    final_wav = recombine_cvc_audio(
        mod_vowel_path=v_mod,
        c1_path=c1,
        c2_path=c2,
        vowel_phoneme="AW",  # match the actual vowel you’re using
        out_path="bag_rebuilt2.wav",
    )


    print("Re-created CVC saved to:", final_wav)






