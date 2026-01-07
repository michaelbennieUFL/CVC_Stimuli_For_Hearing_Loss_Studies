import parselmouth
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
from tempfile import NamedTemporaryFile
from pathlib import Path


def extract_formants(
    sound: parselmouth.Sound,
    time_step: float = 0.005,        # 5 ms spacing
    max_formant: float = 5500,
    num_formants: int = 4
):
    duration = sound.duration
    times = np.arange(0, duration, time_step)
    formant_obj = parselmouth.praat.call(
        sound, "To Formant (burg)", time_step, num_formants, max_formant, 0.035, 50
    )
    pitch_obj = sound.to_pitch()

    f0 = np.array([
        pitch_obj.get_value_at_time(t) or np.nan for t in times
    ])

    f_tracks = []
    for formant_idx in range(1, num_formants + 1):
        formant_track = np.array([
            formant_obj.get_value_at_time(formant_idx, t) or np.nan
            for t in times
        ])
        f_tracks.append(formant_track)

    return times, f0, f_tracks


def split_stereo_wav(wav_path: Path):
    """Return paths to left/right mono WAVs. If input is mono, right is None."""
    rate, data = wavfile.read(wav_path)

    # MONO
    if data.ndim == 1:
        mono_path = NamedTemporaryFile(delete=False, suffix=".wav").name
        wavfile.write(mono_path, rate, data)
        return Path(mono_path), None

    # STEREO
    if data.ndim == 2 and data.shape[1] == 2:
        left_path = NamedTemporaryFile(delete=False, suffix=".wav").name
        right_path = NamedTemporaryFile(delete=False, suffix=".wav").name
        wavfile.write(left_path, rate, data[:, 0])
        wavfile.write(right_path, rate, data[:, 1])
        return Path(left_path), Path(right_path)

    raise ValueError("Unsupported audio shape. Expected mono or stereo WAV.")


def hz_to_bark(f):
    f = np.asarray(f)
    return (26.81 * f) / (1960 + f) - 0.53


def bark_to_hz(z):
    # Inverse of the Zwicker/Bark mapping used above
    z = np.asarray(z)
    return 1960.0 / (26.81 / (z + 0.53) - 1.0)


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
    diff_threshold: float = 1.0,      # in Bark (or Hz if use_bark=False)
    diff_color: str = "red",
    diff_alpha: float = 0.25,
    interp_step: float = 0.005        # seconds
):
    """
    If stereo: plot left vs right and highlight regions where |L–R| >= diff_threshold.
               Also plot the average of F1 and F2 *inside those red regions only*.
    If mono:   plot a single set of formants/F0 (no difference regions).
    """
    wav_path = Path(wav_path)
    left_wav, right_wav = split_stereo_wav(wav_path)
    is_mono = right_wav is None

    # --- extract formants ---------------------------------------------------
    sound_left = parselmouth.Sound(str(left_wav))
    tL, f0L, fL = extract_formants(sound_left)

    if not is_mono:
        sound_right = parselmouth.Sound(str(right_wav))
        tR, f0R, fR = extract_formants(sound_right)

    # --- convert to Bark if requested --------------------------------------
    if use_bark:
        f0L = hz_to_bark(f0L)
        fL  = [hz_to_bark(track) for track in fL]
        if not is_mono:
            f0R = hz_to_bark(f0R)
            fR  = [hz_to_bark(track) for track in fR]

    # --- plotting -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 8))
    if is_mono:
        ax.set_title("Formants (Mono)")
    else:
        ax.set_title("Formant Comparison (Left vs Right Channel)\n"
                     f"Red regions: |Δ| ≥ {diff_threshold} {'Bark' if use_bark else 'Hz'}")

    colors_left  = ["blue", "green", "purple", "magenta", "navy"]
    colors_right = ["cyan", "lime", "orchid", "hotpink", "darkturquoise"]

    # Formants F1–F4
    for i in range(4):
        label = f"F{i+1}" + ("" if is_mono else " Left")
        ax.plot(tL, fL[i], label=label, color=colors_left[i])
        if not is_mono:
            ax.plot(tR, fR[i], label=f"F{i+1} Right", color=colors_right[i], linestyle="--")

    # F0
    ax.plot(tL, f0L, label="F0" + ("" if is_mono else " Left"), color="red", linewidth=2)
    if not is_mono:
        ax.plot(tR, f0R, label="F0 Right", color="orange", linestyle="--", linewidth=2)

    # --- highlight regions + average F1/F2 in those regions (stereo only) ---
    if not is_mono:
        # Build a common time grid
        t_start = max(tL[0], tR[0])
        t_end   = min(tL[-1], tR[-1])
        common_t = np.arange(t_start, t_end, interp_step)

        # Interpolate F1..F4 onto that grid
        L_tracks = [np.interp(common_t, tL, track) for track in fL]
        R_tracks = [np.interp(common_t, tR, track) for track in fR]

        # Difference mask using any formant track
        diff_mask = np.zeros_like(common_t, dtype=bool)
        for Lt, Rt in zip(L_tracks, R_tracks):
            diff_mask |= np.abs(Lt - Rt) >= diff_threshold

        # Shade the regions
        for start, end in _find_segments(diff_mask, common_t):
            ax.axvspan(start, end, color=diff_color, alpha=diff_alpha, zorder=0)

        # --- Averages for F1 and F2 in red regions only ---------------------
        # Prepare masked arrays (NaN outside red regions so nothing is drawn there)
        if np.any(diff_mask):
            avg_F1 = (L_tracks[0] + R_tracks[0]) / 2.0
            avg_F2 = (L_tracks[1] + R_tracks[1]) / 2.0

            avg_F1_plot = np.where(diff_mask, avg_F1, np.nan)
            avg_F2_plot = np.where(diff_mask, avg_F2, np.nan)

            ax.plot(common_t, avg_F1_plot, linewidth=3, label="Avg F1 (red regions)")
            ax.plot(common_t, avg_F2_plot, linewidth=3, linestyle="--", label="Avg F2 (red regions)")
        # else: no red regions -> do not draw any averages

    ax.set_xlabel("Time (s)")
    if use_bark:
        ax.set_ylabel("Bark Scale")
        ax.set_yticks(np.arange(0, 20.5, 0.5))
        ax.set_ylim(0, 20)

        # Right-hand y-axis in Hz (tick labels converted from Bark)
        ax_hz = ax.twinx()
        bark_ticks = ax.get_yticks()
        hz_tick_vals = bark_to_hz(bark_ticks)
        # Clean up extremely high Hz labels outside typical range
        hz_tick_vals = np.clip(hz_tick_vals, 0, 11000)
        ax_hz.set_ylim(ax.get_ylim())
        ax_hz.set_yticks(bark_ticks)
        ax_hz.set_ylabel("Frequency (Hz)")
        ax_hz.set_yticklabels([f"{int(v):d}" for v in hz_tick_vals])
    else:
        ax.set_ylabel("Frequency (Hz)")
        ax.set_yscale("log")

    ax.legend(loc="upper right", ncol=2)
    ax.grid(True)
    fig.tight_layout()
    plt.show()

    # clean up the temp mono files
    os.remove(left_wav)
    if right_wav is not None:
        os.remove(right_wav)


if __name__ == "__main__":
    # Works for mono and stereo. In stereo, will show Avg F1/F2 only if there are red regions.
    plot_formants("generated_cvc_final/dif_def_daf_run/low_f0/dash/discordants/cvc_variant_AE_discordant_IH_Optimized_AE_Optimized.wav")
    plot_formants("generated_cvc_final/dif_def_daf_run/low_f0/dash/cvc_variant_AE_IH_Optimized.wav")
