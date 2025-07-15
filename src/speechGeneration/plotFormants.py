import parselmouth
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
from tempfile import NamedTemporaryFile
from pathlib import Path


def extract_formants(sound: parselmouth.Sound, f0_min=60, f0_max=450, max_formant=5500):
    point_proc = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", f0_min, f0_max)
    formants = parselmouth.praat.call(sound, "To Formant (burg)", 0.0025, 5, max_formant, 0.01, 50)
    pitch = sound.to_pitch()

    times, f0 = [], []
    f_tracks = [[] for _ in range(5)]  # F1–F5

    for i in range(1, parselmouth.praat.call(point_proc, "Get number of points") + 1):
        t = parselmouth.praat.call(point_proc, "Get time from index", i)
        f0_val = parselmouth.praat.call(pitch, "Get value at time", t, "Hertz", "Linear")

        formant_vals = [
            parselmouth.praat.call(formants, "Get value at time", formant_idx + 1, t, "Hertz", "Linear")
            for formant_idx in range(5)
        ]

        if not np.isnan(f0_val) and all(not np.isnan(f) for f in formant_vals):
            times.append(t)
            f0.append(f0_val)
            for i, f in enumerate(formant_vals):
                f_tracks[i].append(f)

    return np.array(times), np.array(f0), [np.array(track) for track in f_tracks]


def split_stereo_wav(wav_path: Path):
    rate, data = wavfile.read(wav_path)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("Audio is not stereo.")

    left_path = NamedTemporaryFile(delete=False, suffix=".wav").name
    right_path = NamedTemporaryFile(delete=False, suffix=".wav").name

    wavfile.write(left_path, rate, data[:, 0])
    wavfile.write(right_path, rate, data[:, 1])
    return Path(left_path), Path(right_path)


def hz_to_bark(f):
    return (26.81 * f) / (1960 + f) - 0.53


def _find_segments(mask, times):
    """Return (start, end) pairs (in seconds) for contiguous True regions in `mask`."""
    segs = []
    in_seg = False
    seg_start = None
    for i, m in enumerate(mask):
        if m and not in_seg:
            in_seg = True
            seg_start = times[i]
        elif not m and in_seg:
            in_seg = False
            segs.append((seg_start, times[i]))
    if in_seg:  # handle a segment that runs to the end
        segs.append((seg_start, times[-1]))
    return segs


def plot_formants(
    wav_path: str | Path,
    use_bark: bool = True,
    diff_threshold: float = 1.0,      # in Bark
    diff_color: str = "red",
    diff_alpha: float = 0.25,
    interp_step: float = 0.005        # seconds
):
    """
    Plot left- vs right-channel formants/F0 and highlight any time region
    where |Left – Right| >= diff_threshold Bark on *any* formant track.
    """
    wav_path = Path(wav_path)
    left_wav, right_wav = split_stereo_wav(wav_path)

    # --- extract formants ---------------------------------------------------
    sound_left  = parselmouth.Sound(str(left_wav))
    sound_right = parselmouth.Sound(str(right_wav))

    tL, f0L, fL = extract_formants(sound_left)
    tR, f0R, fR = extract_formants(sound_right)

    # --- convert to Bark if requested --------------------------------------
    if use_bark:
        f0L = hz_to_bark(f0L)
        f0R = hz_to_bark(f0R)
        fL  = [hz_to_bark(track) for track in fL]
        fR  = [hz_to_bark(track) for track in fR]

    # --- build a common time grid ------------------------------------------
    t_start = max(tL[0], tR[0])
    t_end   = min(tL[-1], tR[-1])
    common_t = np.arange(t_start, t_end, interp_step)

    # interpolate every track onto that grid
    L_tracks = [np.interp(common_t, tL, track) for track in fL]
    R_tracks = [np.interp(common_t, tR, track) for track in fR]

    # you can add F0 here too by uncommenting the next two lines
    # L_tracks.append(np.interp(common_t, tL, f0L))
    # R_tracks.append(np.interp(common_t, tR, f0R))

    # --- detect where ANY track differs by >= threshold --------------------
    diff_mask = np.zeros_like(common_t, dtype=bool)
    for Lt, Rt in zip(L_tracks, R_tracks):
        diff_mask |= np.abs(Lt - Rt) >= diff_threshold

    diff_segments = _find_segments(diff_mask, common_t)

    # --- plotting -----------------------------------------------------------
    plt.figure(figsize=(14, 8))
    plt.title("Formant Comparison (Left vs Right Channel)\n"
              f"Red regions: |Δ| ≥ {diff_threshold} Bark")

    colors_left  = ["blue",  "green", "purple",   "magenta",       "navy"]
    colors_right = ["cyan",  "lime",  "orchid",   "hotpink",       "darkturquoise"]

    # Formants F1–F5
    for i in range(5):
        plt.plot(tL, fL[i], label=f"F{i+1} Left",  color=colors_left[i])
        plt.plot(tR, fR[i], label=f"F{i+1} Right", color=colors_right[i],
                 linestyle="--")

    # F0 (optional)
    plt.plot(tL, f0L, label="F0 Left",  color="red",   linewidth=2)
    plt.plot(tR, f0R, label="F0 Right", color="orange", linestyle="--",
             linewidth=2)

    # highlight regions with big differences
    for start, end in diff_segments:
        plt.axvspan(start, end, color=diff_color, alpha=diff_alpha, zorder=0)

    plt.xlabel("Time (s)")
    if use_bark:
        plt.ylabel("Bark Scale")
        plt.yticks(np.arange(0, 25.5, 0.5))
        plt.ylim(0, 25)
    else:
        plt.ylabel("Frequency (Hz)")
        plt.yscale("log")

    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # clean up the temp mono files
    os.remove(left_wav)
    os.remove(right_wav)



if __name__ == "__main__":
    plot_formants("generated_cvc_good/beg/discordants/cvc_variant_EH_discordant_AE_IH.wav")
    plot_formants("generated_cvc_good/beg/discordants/cvc_variant_EH_discordant_AE_|IH_50_AE_50|.wav")
