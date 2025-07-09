#!/usr/bin/env python3
"""
Search optimal vowel-pair points under Bark-distance & SD constraints,
then plot ±1 SD ellipses with the chosen points / lines
and save a TSV of the results.

Usage:
    python3 vowel_pair_search.py \
        --stats ../../input_data/vowel_stats.txt \
        --group NWP_And_Reading \
        --density 40 \
        --workers 4
"""

import argparse
import multiprocessing as mp
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

SDBUFFER = 1.1

_A_flat = _U_flat = _stats = None  # set at run-time


def _init_pool(A_flat, U_flat, stats):
    """Store big arrays in globals so child processes can see them cheaply."""
    global _A_flat, _U_flat, _stats
    _A_flat, _U_flat, _stats = A_flat, U_flat, stats

# ------------------------------------------------------------------
# 1.  Helper: read stats file  (you already have this in your project)
# ------------------------------------------------------------------
try:
    from src.analyzeAmEVowels.parseVowelDistFile import parse_vowel_dist_data
except ImportError:
    raise SystemExit(
        "❌ Could not import parse_vowel_dist_data. "
        "Make sure your project is on PYTHONPATH."
    )

# ------------------------------------------------------------------
# 2.  Psycho-acoustic conversions & geometry helpers
# ------------------------------------------------------------------
def bark(f_hz: float) -> float:
    return 13 * np.arctan(0.00076 * f_hz) + 3.5 * np.arctan((f_hz / 7000) ** 2)


def bark_distance(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Vectorised Bark-space Euclidean distance for 2-column arrays."""
    return np.linalg.norm(bark(p) - bark(q), axis=-1)


def inside_ellipse(points: np.ndarray, mean: np.ndarray, sd: np.ndarray, radius=1.0):
    """Boolean mask of points inside a 1-SD (or scaled) ellipse."""
    z = (points - mean) / sd
    return (z**2).sum(axis=-1) <= radius**2


# ------------------------------------------------------------------
# AA  ↔  EH  -------------------------------------------------------
# ------------------------------------------------------------------

def line_hits_vowel(p, q, vowel, stats, radius=1.0, samples=60) -> bool:
    """Return True if the line segment p–q enters the vowel's ±radius SD ellipse."""
    t = np.linspace(0, 1, samples)[:, None]
    line = p * (1 - t) + q * t
    mu = np.array([stats[vowel]["F1"], stats[vowel]["F2"]])
    sd = np.array([stats[vowel]["F1_sd"], stats[vowel]["F2_sd"]])
    return inside_ellipse(line, mu, sd, radius).any()


def aa_eh_search(best_AA, stats, density=30):
    """
    Choose an EH point that
      • lies inside EH ±1 SD but outside AE/AH SDBUFFER SD,
      • the AA–EH line intersects both AE & AH 1 SD ellipses,
      • minimises |BarkDist(AA,EH)−1|  **plus**  (F1_AA−F1_EH)^2.
    """
    vEH = "EH"
    mean, sd = np.array([stats[vEH]["F1"], stats[vEH]["F2"]]), np.array(
        [stats[vEH]["F1_sd"], stats[vEH]["F2_sd"]]
    )

    # ---- sampling grid -------------------------------------------------
    f1_grid = np.linspace(mean[0] - sd[0], mean[0] + sd[0], density)
    f2_grid = np.linspace(mean[1] - sd[1], mean[1] + sd[1], density)
    F1, F2 = np.meshgrid(f1_grid, f2_grid, indexing="ij")
    EH_points = np.stack([F1, F2], axis=-1)

    # ---- masks ---------------------------------------------------------
    m_in = inside_ellipse(EH_points, mean, sd, 1)
    m_out_AE = ~inside_ellipse(
        EH_points, np.array([stats["AE"]["F1"], stats["AE"]["F2"]]),
        np.array([stats["AE"]["F1_sd"], stats["AE"]["F2_sd"]]), SDBUFFER
    )
    m_out_AH = ~inside_ellipse(
        EH_points, np.array([stats["AH"]["F1"], stats["AH"]["F2"]]),
        np.array([stats["AH"]["F1_sd"], stats["AH"]["F2_sd"]]), SDBUFFER
    )
    pts = EH_points.reshape(-1, 2)[(m_in & m_out_AE & m_out_AH).ravel()]

    # ---- optimisation --------------------------------------------------
    target = 1.0                       # desired Bark distance
    best = (np.inf, None)              # (metric, point)

    for p in pts:
        # line must intersect AE & AH 1-SD ellipses
        if not (line_hits_vowel(best_AA, p, "AE", stats) and
                line_hits_vowel(best_AA, p, "AH", stats)):
            continue

        bark_term = abs(bark_distance(best_AA, p) - target)
        f1_term   = (best_AA[0] - p[0]) ** 2            # NEW loss
        metric    = bark_term + f1_term                 # combine

        if metric < best[0]:
            best = (metric, p)

    return best[1]   # best_EH  (array: [F1_EH , F2_EH])



# ------------------------------------------------------------------
# AE  ↔  UH  -------------------------------------------------------
# ------------------------------------------------------------------
def ae_uh_search(stats, density=30, line_samples=60):
    """
    Find AE–UH points that
      • are ≥SDBUFFER SD away from EH,
      • lie within their own ±1 SD ellipses,
      • minimise Bark distance between AE & UH,
      • and minimise the *worst* minimum edge-distance
        to either the EH or AH 1-SD ellipse.
    """
    vAE, vUH, vEH, vAH = "AE", "UH", "EH", "AH"

    # --- means & SDs -------------------------------------------------
    def mu_sd(v):
        return (np.array([stats[v]["F1"], stats[v]["F2"]]),
                np.array([stats[v]["F1_sd"], stats[v]["F2_sd"]]))
    mu_AE, sd_AE = mu_sd(vAE)
    mu_UH, sd_UH = mu_sd(vUH)
    mu_EH, sd_EH = mu_sd(vEH)
    mu_AH, sd_AH = mu_sd(vAH)

    # --- grids -------------------------------------------------------
    f1_AE = np.linspace(mu_AE[0] - sd_AE[0], mu_AE[0] + sd_AE[0], density)
    f2_AE = np.linspace(mu_AE[1] - sd_AE[1], mu_AE[1] + sd_AE[1], density)
    f1_UH = np.linspace(mu_UH[0] - sd_UH[0], mu_UH[0] + sd_UH[0], density)
    f2_UH = np.linspace(mu_UH[1] - sd_UH[1], mu_UH[1] + sd_UH[1], density)

    F1_AE, F2_AE = np.meshgrid(f1_AE, f2_AE, indexing="ij")
    F1_UH, F2_UH = np.meshgrid(f1_UH, f2_UH, indexing="ij")
    AE_pts = np.stack([F1_AE, F2_AE], axis=-1).reshape(-1, 2)
    UH_pts = np.stack([F1_UH, F2_UH], axis=-1).reshape(-1, 2)

    # --- masks -------------------------------------------------------
    in_AE   = inside_ellipse(AE_pts, mu_AE, sd_AE, 1)
    out_EH  = ~inside_ellipse(AE_pts, mu_EH, sd_EH, SDBUFFER)
    AE_pts  = AE_pts[in_AE & out_EH]

    in_UH   = inside_ellipse(UH_pts, mu_UH, sd_UH, 1)
    out_EH  = ~inside_ellipse(UH_pts, mu_EH, sd_EH, SDBUFFER)
    UH_pts  = UH_pts[in_UH & out_EH]

    # --- helper: min edge-distance for one ellipse -------------------
    def min_edge_distance(line, mu, sd):
        z = (line - mu) / sd          # (N,2) z-scores
        r = np.sqrt((z**2).sum(axis=1))
        return np.min(np.abs(r - 1))  # distance to 1-SD boundary

    # --- search ------------------------------------------------------
    best = (np.inf, None, None)
    t = np.linspace(0, 1, line_samples)[:, None]

    for p in AE_pts:
        for q in UH_pts:
            line   = p * (1 - t) + q * t           # (samples, 2)
            pen_EH = min_edge_distance(line, mu_EH, sd_EH)
            pen_AH = min_edge_distance(line, mu_AH, sd_AH)
            penalty = max(pen_EH, pen_AH)

            d_pair = bark_distance(p, q)
            metric = d_pair + penalty**2           # combine objectives

            if metric < best[0]:
                best = (metric, p, q)

    return best[1], best[2]  # best_AE , best_UH



def _aa_uw_worker(index_chunk):
    """Evaluate AA–UW candidates for a slice of indices."""
    best_d = np.inf
    best_A = best_U = None
    for i in index_chunk:
        p = _A_flat[i]
        q = _U_flat[i]
        if not _intersects_both(p, q, _stats):
            continue
        d = bark_distance(p, q)
        if d < best_d:
            best_d, best_A, best_U = d, p, q
    return best_d, best_A, best_U



# 3.  Search routines  (vectorised + optional multiprocessing)
# ------------------------------------------------------------------
def shared_f2_search(v1, v2, stats, density=30, workers=1):
    """
    Find AA–UW pair (shared F2) with minimum Bark distance
    while line intersects UH & AH and endpoints respect all constraints.
    """
    mean1, sd1 = np.array([stats[v1]["F1"], stats[v1]["F2"]]), np.array(
        [stats[v1]["F1_sd"], stats[v1]["F2_sd"]]
    )
    mean2, sd2 = np.array([stats[v2]["F1"], stats[v2]["F2"]]), np.array(
        [stats[v2]["F1_sd"], stats[v2]["F2_sd"]]
    )

    # Common F2 range (overlap of ±1 SD bands)
    f2_min = max(mean1[1] - sd1[1], mean2[1] - sd2[1])
    f2_max = min(mean1[1] + sd1[1], mean2[1] + sd2[1])
    f2_grid = np.linspace(f2_min, f2_max, density)

    f1_grid1 = np.linspace(mean1[0] - sd1[0], mean1[0] + sd1[0], density)
    f1_grid2 = np.linspace(mean2[0] - sd2[0], mean2[0] + sd2[0], density)

    # Build full grid
    A_f1, U_f1, F2 = np.meshgrid(f1_grid1, f1_grid2, f2_grid, indexing="ij")
    A_points = np.stack([A_f1, F2], axis=-1)
    U_points = np.stack([U_f1, F2], axis=-1)

    # Boolean masks
    valid_A = inside_ellipse(A_points, mean1, sd1, 1) & ~inside_ellipse(
        A_points, np.array([stats["AH"]["F1"], stats["AH"]["F2"]]),  # AH centre
        np.array([stats["AH"]["F1_sd"], stats["AH"]["F2_sd"]]), SDBUFFER
    )
    valid_U = inside_ellipse(U_points, mean2, sd2, 1) & ~inside_ellipse(
        U_points, np.array([stats["AH"]["F1"], stats["AH"]["F2"]]),
        np.array([stats["AH"]["F1_sd"], stats["AH"]["F2_sd"]]), SDBUFFER
    )


    # boolean masks as before -> `valid`
    A_flat = A_points.reshape(-1, 2)
    U_flat = U_points.reshape(-1, 2)
    valid = (valid_A & valid_U).ravel()
    valid_idx = np.where(valid)[0]

    # ---- parallel branch ------------------------------------------
    if workers > 1 and len(valid_idx):
        chunk_size = int(np.ceil(len(valid_idx) / workers))
        chunks = [valid_idx[i:i + chunk_size] for i in range(0, len(valid_idx), chunk_size)]

        with mp.Pool(
                processes=workers,
                initializer=_init_pool,
                initargs=(A_flat, U_flat, stats),
        ) as pool:
            results = pool.map(_aa_uw_worker, chunks)

        best = min(results, key=lambda x: x[0])
        return best[1], best[2]

    # ---- serial fallback ------------------------------------------
    best_d, best_A, best_U = np.inf, None, None
    for i in valid_idx:
        p, q = A_flat[i], U_flat[i]
        if not _intersects_both(p, q, stats):
            continue
        d = bark_distance(p, q)
        if d < best_d:
            best_d, best_A, best_U = d, p, q
    return best_A, best_U


# ---------------------------


def _intersects_both(p, q, stats, samples=60):
    """Does the line p–q pass through UH *and* AH ±1 SD ellipses?"""
    t = np.linspace(0, 1, samples)[:, None]
    line = p * (1 - t) + q * t

    mean_UH = np.array([stats["UH"]["F1"], stats["UH"]["F2"]])
    sd_UH = np.array([stats["UH"]["F1_sd"], stats["UH"]["F2_sd"]])
    mean_AH = np.array([stats["AH"]["F1"], stats["AH"]["F2"]])
    sd_AH = np.array([stats["AH"]["F1_sd"], stats["AH"]["F2_sd"]])

    hit_UH = inside_ellipse(line, mean_UH, sd_UH, 1).any()
    hit_AH = inside_ellipse(line, mean_AH, sd_AH, 1).any()
    return hit_UH and hit_AH


# ------------------------------------------------------------------
# 4.  Plotting
# ------------------------------------------------------------------
def draw_ellipse(ax, vowel, stats, color, label=None):
    e = Ellipse(
        (stats[vowel]["F2"], stats[vowel]["F1"]),
        width=2 * stats[vowel]["F2_sd"],
        height=2 * stats[vowel]["F1_sd"],
        edgecolor=color,
        facecolor="none",
        lw=2,
        label=label,
    )
    ax.add_patch(e)


def plot_all(stats, best_AA, best_UW, best_EH, best_AE, best_UH):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

    # AA-UW
    ax = axs[0]
    for v, c in zip(["AA", "UW", "UH", "AH"], ["blue", "red", "gray", "gray"]):
        draw_ellipse(ax, v, stats, c, label=v if v in ["AA", "UW"] else None)
    ax.plot(
        [best_AA[1], best_UW[1]], [best_AA[0], best_UW[0]], "-k", lw=1.5, zorder=3
    )
    ax.scatter([best_AA[1], best_UW[1]], [best_AA[0], best_UW[0]], c="k", zorder=4)
    ax.set_title("AA ↔ UW   (shared F2)")

    # AA-EH
    ax = axs[1]
    for v, c in zip(["AA", "EH", "AE", "AH"], ["blue", "green", "gray", "gray"]):
        draw_ellipse(ax, v, stats, c, label=v if v in ["AA", "EH"] else None)
    ax.plot(
        [best_AA[1], best_EH[1]], [best_AA[0], best_EH[0]], "-k", lw=1.5, zorder=3
    )
    ax.scatter([best_EH[1]], [best_EH[0]], c="k", zorder=4)
    ax.set_title("AA ↔ EH   (x₁ fixed)")

    # AE-UH
    ax = axs[2]
    for v, c in zip(["AE", "UH", "EH", "AH"], ["purple", "orange", "gray", "gray"]):
        draw_ellipse(ax, v, stats, c, label=v if v in ["AE", "UH"] else None)
    ax.plot(
        [best_AE[1], best_UH[1]], [best_AE[0], best_UH[0]], "-k", lw=1.5, zorder=3
    )
    ax.scatter([best_AE[1], best_UH[1]], [best_AE[0], best_UH[0]], c="k", zorder=4)
    ax.set_title("AE ↔ UH   (balanced)")

    for ax in axs:
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_xlabel("F2 (Hz)")
        ax.set_ylabel("F1 (Hz)")
        ax.grid(alpha=0.4)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------
# 5.  main()
# ------------------------------------------------------------------
def main(stats_path, group_label, density=40, workers=1):
    """
    Run vowel pair search and plotting pipeline using passed parameters.

    Parameters:
        stats_path (str or Path): Path to vowel stats file
        group_label (str): The group name to filter
        density (int): Grid resolution
        workers (int): Number of parallel workers (≥2 enables multiprocessing)
    """
    # ------------------------------------------------------------------
    # 1. Load stats for chosen group
    # ------------------------------------------------------------------
    df = parse_vowel_dist_data(stats_path)
    sub = df[df["group"] == group_label].reset_index(drop=True)
    if sub.empty:
        raise ValueError(f"❌ Group '{group_label}' not found in {stats_path}")

    stats = {
        row.vowel: {
            "F1": row.f1_mean,
            "F1_sd": row.f1_sd,
            "F2": row.f2_mean,
            "F2_sd": row.f2_sd,
        }
        for row in sub.itertuples()
    }

    # ------------------------------------------------------------------
    # 2. Search optimal vowel points
    # ------------------------------------------------------------------
    best_AA, best_UW = shared_f2_search("AA", "UW", stats, density, workers)
    best_EH = aa_eh_search(best_AA, stats, density)
    best_AE, best_UH = ae_uh_search(stats, density)

    # ------------------------------------------------------------------
    # 3. Save TSV of results
    # ------------------------------------------------------------------
    out_rows = [
        ["AA_UW", "AA", *best_AA, "UW", *best_UW],
        ["AA_EH", "AA", *best_AA, "EH", *best_EH],
        ["AE_UH", "AE", *best_AE, "UH", *best_UH],
    ]

    out_df = pd.DataFrame(
        out_rows, columns=["PAIR", "V1", "F1_1", "F2_1", "V2", "F1_2", "F2_2"]
    )
    tsv_path = Path("best_pairs.tsv")
    out_df.to_csv(tsv_path, sep="\t", index=False)
    print(f"✅ Results written to {tsv_path}")

    # ------------------------------------------------------------------
    # 4. Plot vowel space with ellipses + best pairs
    # ------------------------------------------------------------------
    plot_all(stats, best_AA, best_UW, best_EH, best_AE, best_UH)

if __name__ == "__main__":
    main(
        stats_path="../../input_data/vowel_stats.txt",
        group_label="NWP_And_Reading",
        density=100,
        workers=18
    )
