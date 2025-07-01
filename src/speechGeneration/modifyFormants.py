from __future__ import annotations
import math, shutil, tempfile, os, subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import parselmouth
from parselmouth import praat
from rich.console import Console
from rich.table import Table

FORMANT_IDX = {"f0": 0, "f1": 1, "f2": 2, "f3": 3, "f4": 4}

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
    cmd = [
        "python",
        os.path.join(module_path, "inference_hifiglot.py"),
        "--input_path", input_path,
        "--output_path", output_path,
        "--config", os.path.join(module_path, config),
        "--fm_config", os.path.join(module_path, fm_config),
        "--checkpoint_path", os.path.join(module_path, checkpoint_path),
        "--feature_scale", str([float(x) for x in feature_scale]),
    ]
    subprocess.run(cmd, check=True)


# ────────────────────────────────────────────── Adam helper ─────────────────────────────────────────
class Adam:
    def __init__(self, size: int, lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = np.zeros(size)
        self.v = np.zeros(size)
        self.t = 0

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * (grad ** 2)

        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)

        update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return params - update


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
    adam_lr: float = 0.02,
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
        current = analyse_formants(max(outs, key=lambda f: f.stat().st_mtime))[:5]

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
            diff = current[idx] - tgt
            err[idx] = diff
            if abs(diff) > tolerance:
                all_good = False

        if all_good:
            console.print("[bold green]Target achieved![/bold green]")
            shutil.rmtree(tmp_in, ignore_errors=True)
            return scales.tolist(), current.tolist()

        # treat err/tgt as gradient (sign + magnitude), clip to ±0.5 so we don't explode
        grad = np.tanh(err / np.maximum(1, np.array([target.get(f"f{i}", 1) for i in range(5)])))
        scales = optimiser.step(scales, grad)

        # keep within valid band
        scales = np.clip(scales, 0.4, 2.5)

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
