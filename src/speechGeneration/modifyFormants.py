from __future__ import annotations
import math, shutil, tempfile, os, subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import parselmouth
from parselmouth import praat
from rich.console import Console
from rich.table import Table
import soundfile as sf
import pyloudnorm as pyln
from scipy.optimize import minimize
from src.speechGeneration.SourceFilterNeuralFormants.inference_hifiglot import run_hifiglot_inference_direct

FORMANT_IDX = {"f0": 0, "f1": 1, "f2": 2, "f3": 3, "f4": 4}



def calculate_rms_volume(wav_path: str | Path) -> float:
    sound = parselmouth.Sound(str(wav_path))
    samples = sound.values.flatten()
    rms = np.sqrt(np.mean(samples ** 2))
    return float(rms)


def normalize_volume(target_wav: Path, reference_rms: float):
    sound = parselmouth.Sound(str(target_wav))
    samples = sound.values.flatten()
    current_rms = np.sqrt(np.mean(samples ** 2))

    # Avoid division by zero
    if current_rms == 0:
        return

    normalization_factor = reference_rms / current_rms
    normalized_samples = samples * normalization_factor

    # Save normalized sound back
    normalized_sound = parselmouth.Sound(normalized_samples, sampling_frequency=sound.sampling_frequency)
    normalized_sound.save(str(target_wav), "WAV")

def match_loudness(reference_wav: Path, target_wav: Path):
    ref_data, ref_rate = sf.read(str(reference_wav))
    tgt_data, tgt_rate = sf.read(str(target_wav))

    # Resample target to match reference if needed
    if ref_rate != tgt_rate:
        tgt_data = librosa.resample(tgt_data.T, orig_sr=tgt_rate, target_sr=ref_rate).T
        tgt_rate = ref_rate

    # Compute loudness and normalize
    meter = pyln.Meter(ref_rate)
    ref_loudness = meter.integrated_loudness(ref_data)
    tgt_loudness = meter.integrated_loudness(tgt_data)

    normalized_audio = pyln.normalize.loudness(tgt_data, tgt_loudness, ref_loudness)

    sf.write(str(target_wav), normalized_audio, ref_rate)


# ───────────────────────────────────────── analyse_formants ──────────────────────────────────────────
def analyse_formants(
    wav_path: str | Path,
    f0_min: int = 75,
    f0_max: int = 400,
    max_formant: int = 5_500,
) -> np.ndarray:
    sound = parselmouth.Sound(str(wav_path))
    point_proc = praat.call(sound, "To PointProcess (periodic, cc)", f0_min, f0_max)
    formants = praat.call(sound, "To Formant (burg)", 0.0025, 5, max_formant, 0.025, 50)

    f_tracks = [[] for _ in range(5)]
    for i_pt in range(1, praat.call(point_proc, "Get number of points") + 1):
        t = praat.call(point_proc, "Get time from index", i_pt)
        for ch in range(5):
            val = praat.call(formants, "Get value at time", ch + 1, t, "Hertz", "Linear")
            if not np.isnan(val):
                f_tracks[ch].append(val)

    pitch = sound.to_pitch()
    f0_voiced = pitch.selected_array["frequency"][pitch.selected_array["frequency"] > 0]

    means = [float(np.mean(f0_voiced))] + [float(np.mean(track)) for track in f_tracks]
    return np.array(means[:6])  # F0 … F5


# ────────────────────────────────────────── HiFi-Glot wrapper ───────────────────────────────────────
def run_hifiglot_inference(
        input_path: str,
        output_path: str,
        module_path: str,
        config: str,
        fm_config: str,
        checkpoint_path: str,
        feature_scale: List[float],
):
    wav_files = [str(p) for p in Path(input_path).glob("*.wav")]

    run_hifiglot_inference_direct(
        file_list=wav_files,
        output_path=output_path,
        checkpoint_path=os.path.join(module_path, checkpoint_path),
        config=os.path.join(module_path, config),
        fm_config=os.path.join(module_path, fm_config),
        feature_scale=feature_scale,
    )


# ────────────────────────────────────────────── Adam helper ─────────────────────────────────────────
class Adam:
    def __init__(self, size: int, lr=0.05, beta1=0.75, beta2=0.9, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = np.zeros(size)
        self.v = np.zeros(size)
        self.t = 0

    def step(self, params, grad):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * (grad ** 2)

        # bias corrections
        m_hat = self.m / (1 - self.b1 ** self.t)


        v_hat = self.v / (1 - self.b2 ** self.t)

        step_size = self.lr/(math.log(self.t+1)+1) * math.sqrt(1 - self.b2 ** self.t) / (1 - self.b1 ** self.t)
        return params - step_size * m_hat / (np.sqrt(v_hat) + self.eps)


def formant_loss(log_scales: np.ndarray, original_scales: np.ndarray, target: Dict[str, float],
                 in_wav: Path, output_dir: Path, module_path: str, hifi_cfg: str, fm_cfg: str, ckpt: str) -> float:
    scales = np.exp(log_scales)

    # Run inference with proposed scales
    run_hifiglot_inference(
        input_path=str(in_wav.parent),
        output_path=str(output_dir),
        module_path=module_path,
        config=hifi_cfg,
        fm_config=fm_cfg,
        checkpoint_path=ckpt,
        feature_scale=scales.tolist(),
    )

    # Get latest output file
    outs = list(output_dir.glob("*.wav"))
    if not outs:
        raise FileNotFoundError("No output .wav produced by HiFi-Glot during LBFGS optimization.")
    latest_output = max(outs, key=lambda f: f.stat().st_mtime)

    # Normalize volume to match the original audio
    original_rms = calculate_rms_volume(in_wav)
    normalize_volume(latest_output, original_rms)

    # Calculate resulting formants
    current = analyse_formants(latest_output)[:5]

    # Compute squared relative error only for targets provided
    error = 0.0
    for k, tgt in target.items():
        if k in FORMANT_IDX:
            idx = FORMANT_IDX[k]
            error += ((current[idx] - tgt) / tgt) ** 2

    return error


def refine_with_lbfgs(
    initial_scales: np.ndarray,
    wav_file: Path,
    output_dir: Path,
    target: Dict[str, float],
    module_path: str,
    hifi_cfg: str,
    fm_cfg: str,
    ckpt: str,
    max_iters_lbfgs: int = 10
) -> Tuple[List[float], List[float]]:
    from rich.table import Table
    from rich.console import Console

    console = Console(width=160)

    def callback(log_scales):
        scales = np.exp(log_scales)

        # Run inference
        run_hifiglot_inference(
            input_path=str(wav_file.parent),
            output_path=str(output_dir),
            module_path=module_path,
            config=hifi_cfg,
            fm_config=fm_cfg,
            checkpoint_path=ckpt,
            feature_scale=scales.tolist(),
        )

        outs = list(output_dir.glob("*.wav"))
        if not outs:
            return

        latest_output = max(outs, key=lambda f: f.stat().st_mtime)
        normalize_volume(latest_output, calculate_rms_volume(wav_file))
        current = analyse_formants(latest_output)[:5]

        table = Table(title="LBFGS Iteration")
        headers = ["", "F0", "F1", "F2", "F3", "F4", "Scales"]
        for h in headers:
            table.add_column(h, min_width=8, justify="center", style="cyan", no_wrap=True)

        def fmt_val(x):
            return "–" if x is None or np.isnan(x) else f"{int(round(x))}"

        def ratio_val(meas, key):
            if key not in target or np.isnan(meas): return "–"
            return f"{meas / target[key]:.2f}"

        table.add_row("Target", *(fmt_val(target.get(f"f{i}", None)) for i in range(5)), "—")
        table.add_row("Current", *(fmt_val(current[i]) for i in range(5)), ", ".join(f"{s:.3f}" for s in scales))
        table.add_row("Ratios", *(ratio_val(current[i], f"f{i}") for i in range(5)), "—")

        console.print(table)

    # ─────────────── Actual Optimization ───────────────
    result = minimize(
        fun=formant_loss,
        x0=np.log(initial_scales),
        args=(initial_scales, target, wav_file, output_dir, module_path, hifi_cfg, fm_cfg, ckpt),
        method='L-BFGS-B',
        bounds=[(np.log(0.1), np.log(3.5))] * 5,
        options = {"maxiter": max_iters_lbfgs,
                                 "ftol": 1e-9,  # tighter
                                 "gtol": 1e-7,
                                 "eps": 0.005},  # 4 % step for finite-diff
        callback=callback
    )

    final_scales = np.exp(result.x)

    # Final inference and cleanup
    run_hifiglot_inference(
        input_path=str(wav_file.parent),
        output_path=str(output_dir),
        module_path=module_path,
        config=hifi_cfg,
        fm_config=fm_cfg,
        checkpoint_path=ckpt,
        feature_scale=final_scales.tolist(),
    )

    outs = list(output_dir.glob("*.wav"))
    latest_output = max(outs, key=lambda f: f.stat().st_mtime)

    # Normalize volume to match the original audio
    original_rms = calculate_rms_volume(wav_file)
    current_rms = calculate_rms_volume(latest_output)
    print("Loudness ratio:(pre-norm) :", current_rms / original_rms)
    normalize_volume(latest_output, original_rms)
    # match_loudness(in_wav, latest_output)
    current_rms = calculate_rms_volume(latest_output)
    print("Loudness ratio:(pos-norm) :", current_rms / original_rms)

    final_formants = analyse_formants(latest_output)[:5]

    return final_scales.tolist(), final_formants.tolist()


    # ──────────────────────────────────────────── tune_formants ─────────────────────────────────────────
def tune_formants(
    wav_file: str | Path,
    output_dir: str | Path,
    target: Optional[Dict[str, float]] = None,
    tolerance: float = 50.0,
    max_iters: int = 30,
    module_path: str = "./SourceFilterNeuralFormants/",
    hifi_cfg: str = "checkpoints/HiFi-Glot/config_hifigan.json",
    fm_cfg: str = "checkpoints/HiFi-Glot/config_feature_map.json",
    ckpt: str = "checkpoints/HiFi-Glot",
    adam_lr: float = 0.008,
    freeze_tol: float = 0.008,
) -> Tuple[List[float], List[float]]:
    """
    Optimise HiFi-Glot's five feature-scale factors to hit the requested formants,
    using Adam instead of binary search.
    """
    wav_file, output_dir = Path(wav_file).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    console = Console(width = 160)
    target = target or {}

    # --- helpers ----------------------------------------------------------------
    def fmt_val(x: float | None) -> str:
        return "–" if x is None or np.isnan(x) else f"{int(round(x))}"

    def ratio_val(meas: float, key: str) -> str:
        if key not in target or np.isnan(meas): return "–"
        return f"{meas / target[key]:.2f}"

    # --- prepare -----------------------------------------------------------------
    tmp_in = Path(tempfile.mkdtemp(prefix="hifiglot_in_"))
    in_wav = tmp_in / wav_file.name
    shutil.copy2(wav_file, in_wav)

    original = analyse_formants(in_wav)[:5]          # baseline
    scales = np.ones(5, dtype=float)                 # optimisation vars

    # initialise scales crudely ≈ target/original
    for k, tgt in target.items():
        if k not in FORMANT_IDX: continue
        i = FORMANT_IDX[k.lower()]
        if original[i] > 0:
            scales[i] = np.clip(tgt / original[i], 0.4, 2.5)

    optimiser = Adam(size=5, lr=adam_lr)
    frozen = np.zeros(5, dtype=bool)
    console.print("[bold green]Starting Adam-based formant tuning[/bold green]\n")

    for it in range(1, max_iters + 1):

        # ★────────────────────  CLEAN OUTPUT DIR  ────────────────────★
        for p in output_dir.glob("*"):
            try:
                if p.is_file() or p.is_symlink():
                    p.unlink()
                else:
                    shutil.rmtree(p, ignore_errors=True)
            except Exception as e:
                console.print(f"[yellow]Warning: couldn't delete {p}: {e}[/yellow]")
        # ★────────────────────────────────────────────────────────────★




        run_hifiglot_inference(
            input_path=str(tmp_in),
            output_path=str(output_dir),
            module_path=module_path,
            config=hifi_cfg,
            fm_config=fm_cfg,
            checkpoint_path=ckpt,
            feature_scale=scales.tolist(),
        )

        # pick the newest file that matches the stem
        outs = [f for f in output_dir.glob("*.wav") if wav_file.stem in f.name]
        if not outs:
            raise FileNotFoundError("No output .wav produced by HiFi-Glot.")

        # After inference, pick the newest file:
        outs = [f for f in output_dir.glob("*.wav") if wav_file.stem in f.name]
        if not outs:
            raise FileNotFoundError("No output .wav produced by HiFi-Glot.")

        latest_output = max(outs, key=lambda f: f.stat().st_mtime)

        # Normalize volume to match the original audio
        original_rms = calculate_rms_volume(in_wav)
        current_rms=calculate_rms_volume(latest_output)
        print("Loudness ratio:(pre-norm) :",current_rms/original_rms)
        normalize_volume(latest_output, original_rms)
        #match_loudness(in_wav, latest_output)
        current_rms = calculate_rms_volume(latest_output)
        print("Loudness ratio:(pos-norm) :",current_rms/original_rms)

        # Continue with current formant analysis
        current = analyse_formants(latest_output)[:5]

        # ──────────────── debug table ─────────────────
        table = Table(title=f"Iteration {it} – Adam")
        headers = ["", "F0", "F1", "F2", "F3", "F4", "Scale"]
        for h in headers:
            table.add_column(h, min_width=8, justify="center", style="cyan", no_wrap=True)

        table.add_row("Original", *(fmt_val(x) for x in original), "—")
        table.add_row("Current",  *(fmt_val(x) for x in current),
                      ", ".join(f"{s:.3f}" for s in scales))
        table.add_row("Target",   *(fmt_val(target.get(f"f{i}", None))
                                    for i in range(5)), "—")
        table.add_row("Ratios",   *(ratio_val(current[i], f"f{i}") for i in range(5)), "—")
        console.print(table)

        # ─────────── stopping & gradient calc ─────────
        err = np.zeros(5)
        all_good = True
        for k, tgt in target.items():
            if k not in FORMANT_IDX: continue
            idx = FORMANT_IDX[k]
            abs_diff = current[idx] - tgt
            diff = (current[idx] / tgt) - 1

            if abs(current[idx] / tgt-1) <= freeze_tol:
                frozen[idx] = True
            elif abs(abs_diff) > tolerance or abs(current[idx] / tgt-1) > freeze_tol*1.1:
                frozen[idx] = False


            if not frozen[idx]:
                err[idx] = diff
            else:
                optimiser.m[idx] = 0.0
                optimiser.v[idx] = 0.0
                err[idx] = 0.0


            if (abs(abs_diff) > tolerance and not abs(current[idx] / tgt-1) <= 0.04) and not(it >400 and abs(current[idx] / tgt-1) <= 0.02*(1+it//100)) :
                all_good = False

        if all_good:
            console.print("[bold green]Target achieved![/bold green]")

            # Switch to LBFGS for final refinement
            try:
                console.print("[bold blue]Starting LBFGS refinement…[/bold blue]")
                scales, current = refine_with_lbfgs(
                    initial_scales=scales,
                    wav_file=in_wav,
                    output_dir=output_dir,
                    target=target,
                    module_path=module_path,
                    hifi_cfg=hifi_cfg,
                    fm_cfg=fm_cfg,
                    ckpt=ckpt,
                    max_iters_lbfgs=30
                )
                print("Final scales:", scales)
                print("Achieved:", current)
                console.print("[bold green]LBFGS refinement completed![/bold green]")
            except Exception as e:
                console.print(f"[red]LBFGS refinement failed: {e}[/red]")

            shutil.rmtree(tmp_in, ignore_errors=True)
            return scales, current

        # treat err/tgt as gradient (sign + magnitude), clip to ±0.5 so we don't explode
        grad = np.tanh(err / np.maximum(1, np.array([target.get(f"f{i}", 1) for i in range(5)])))
        grad[frozen] = 0.0
        scales = optimiser.step(scales, grad)

        # keep within valid band
        scales = np.clip(scales, 0.1, 3.5)

    shutil.rmtree(tmp_in, ignore_errors=True)
    raise RuntimeError(
        f"Adam failed to converge within ±{tolerance} Hz after {max_iters} iterations.\n"
        f"Last formant estimate: {current.tolist()}\n"
        f"Scale factors: {scales.tolist()}"
    )




# ──────────────────────────── quick-test invocation (remove in production) ──────────────────────────
if __name__ == "__main__":
    wav = "../../output_files/temp_target/bag_V_20250627103235_mod.wav"
    try:
        scales, freqs = tune_formants(
            wav_file=wav,
            output_dir="../../output_files/temp_result/",
            target={"f0": 166, "f1": 804, "f2": 1188},
            tolerance=6,
            max_iters=40
        )
        print("Final scales:", scales)
        print("Achieved:", freqs)
    except RuntimeError as e:
        print("Tuning failed:", e)
