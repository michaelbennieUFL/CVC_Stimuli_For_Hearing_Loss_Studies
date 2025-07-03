import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, colors
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle, Ellipse

from parseVowelDistFile import parse_vowel_dist_data


def segment_intersects_ellipse(p1, p2, center, a, b, *, tol=1e-12):
    """
    Return True iff the line *segment* P1-P2 intersects (or touches)
    the axis-aligned ellipse centred at `center` with radii `a` (x-axis)
    and `b` (y-axis).

    Parameters
    ----------
    p1, p2  : (x, y) tuples   – segment end-points
    center  : (h, k) tuple    – ellipse centre
    a, b    : float           – semi-axes (must be > 0)
    tol     : float           – numerical tolerance
    """
    (x1, y1), (x2, y2) = p1, p2
    h, k = center


    # --- quadratic intersection test --------------------------------
    dx, dy = x2 - x1, y2 - y1           # segment direction

    # Coefficients of  A t² + B t + C = 0
    A = (dx / a) ** 2 + (dy / b) ** 2
    B = 2 * ((dx * (x1 - h)) / a ** 2 + (dy * (y1 - k)) / b ** 2)
    C = ((x1 - h) / a) ** 2 + ((y1 - k) / b) ** 2 - 1

    # Discriminant
    D = B * B - 4 * A * C
    if D < -tol:
        return False          # no real roots → no intersection

    # Tangent or secant: compute roots t1, t2
    sqrtD = math.sqrt(max(D, 0.0))
    t1 = (-B - sqrtD) / (2 * A)
    t2 = (-B + sqrtD) / (2 * A)

    # Does either root fall on the *segment* (0 ≤ t ≤ 1)?
    return (0 - tol <= t1 <= 1 + tol) or (0 - tol <= t2 <= 1 + tol)

def count_ellipse_intersections(df,
                                start,      # (F1, F2) of segment start
                                end,        # (F1, F2) of segment end
                                scale: float = 1.0,
                                group=None,
                                tol: float = 1e-12):
    """
    Parameters
    ----------
    df : DataFrame
        Columns required:
        ['vowel', 'group', 'f1_mean', 'f1_sd', 'f2_mean', 'f2_sd'].
    start, end : tuple[float, float]
        Segment end-points in Hz (F1, F2).
    scale : float, default 1.0
        Semi-axes = 2 * sd * scale   (1.0 → ±2 sd ellipse).
    group : str | iterable | None
        Restrict the test to one or several groups.
    tol : float, default 1e-12
        Numerical tolerance forwarded to `segment_intersects_ellipse`.

    Returns
    -------
    n_hits : int
        Number of distinct ellipses intersected.
    hits : set[tuple[str, str]]
        {(vowel, group), …}   – labels of intersected ellipses.
    """
    # ----------- optional group filtering --------------------------------
    if group is not None:
        sub = df[df["group"].isin(group)] if isinstance(group, (list, set, tuple)) \
              else df[df["group"] == group]
    else:
        sub = df

    hits = set()
    p1, p2 = start, end

    # ----------- loop through ellipses -----------------------------------
    for _, row in sub.iterrows():
        cx, cy = row["f1_mean"], row["f2_mean"]
        a      = row["f1_sd"] * scale      # semi-axis along F1
        b      = row["f2_sd"] * scale      # semi-axis along F2

        if segment_intersects_ellipse(p1, p2, (cx, cy), a, b, tol=tol):
            hits.add((row["vowel"], row["group"]))

    return max(len(hits)-3,0), hits


import numpy as np

def generate_sampling_points(vowel_row,
                             block_res,              # (f1_width, f2_width)
                             scale=1.0):
    """
    Return every block-centre (F1, F2) that falls inside the vowel’s
    ellipse   (F1-mean, F2-mean, sd’s) × `scale`.

    Parameters
    ----------
    vowel_row : pd.Series  (or dict-like)
        Needs keys 'f1_mean', 'f1_sd', 'f2_mean', 'f2_sd'.
        Usually one row from the dataframe, e.g.
        `df.loc[(df['vowel']=='i') & (df['group']=='m')].iloc[0]`
    block_res : tuple (f1_width, f2_width)
        Size of each rectangular “block” in Hz.
    scale : float, default 1.0
        1.0 → ±2 sd ellipse (matches `count_ellipse_intersections`);
        0.5 → ±1 sd, 2 → ±4 sd, …

    Returns
    -------
    pts : list[tuple]
        [(F1c, F2c), …] centre coordinates for every block whose centre
        is inside the ellipse.
    """
    # ellipse parameters
    cx, cy  = vowel_row['f1_mean'], vowel_row['f2_mean']
    rx      = vowel_row['f1_sd'] * scale   # semi-axes
    ry      = vowel_row['f2_sd'] * scale

    f1_w, f2_w = block_res
    half_f1, half_f2 = f1_w/2, f2_w/2

    # ----- build a bounding grid just big enough -------------
    f1_min = np.floor((cx - rx) / f1_w) * f1_w + half_f1
    f1_max = np.ceil ((cx + rx) / f1_w) * f1_w + half_f1
    f2_min = np.floor((cy - ry) / f2_w) * f2_w + half_f2
    f2_max = np.ceil ((cy + ry) / f2_w) * f2_w + half_f2

    f1_centres = np.arange(f1_min, f1_max + f1_w*0.5, f1_w)
    f2_centres = np.arange(f2_min, f2_max + f2_w*0.5, f2_w)

    pts = []
    for f1c in f1_centres:
        for f2c in f2_centres:
            # ellipse-membership test
            if ((f1c - cx)/rx)**2 + ((f2c - cy)/ry)**2 <= 1:
                pts.append((f1c, f2c))

    return pts

def plot_ploints(vowel_row,
                 points,          # list[(F1, F2)]
                 scale=1.0,
                 show_ellipse=True,
                 invert_f1=True,
                 ax=None):
    """
    Visualise the sampling points for one vowel.

    Parameters
    ----------
    vowel_row : pd.Series or dict-like
        Must provide 'f1_mean', 'f1_sd', 'f2_mean', 'f2_sd'.
    points : list[tuple]
        The centre points returned by `generate_sampling_points`.
    scale : float, default 1.0
        Matches the scale you used when generating points
        (1.0 → ±2 sd rectangle / ellipse).
    show_ellipse : bool, default True
        If True, also outlines the ±2 sd ellipse for reference.
    invert_f1 : bool, default True
        Linguists usually plot F1 downward.  Set False to keep normal
        Cartesian orientation.
    ax : matplotlib.axes.Axes | None
        Supply your own Axes to plot into; if None, a new figure is made.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    # --- geometric params -------------------------------------------------
    cx, cy = vowel_row['f1_mean'], vowel_row['f2_mean']
    rx     = vowel_row['f1_sd'] * scale
    ry     = vowel_row['f2_sd'] * scale

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # bounding rectangle
    rect = Rectangle((cx - rx, cy - ry), 2*rx, 2*ry,
                     facecolor='none', edgecolor='tab:red',
                     linewidth=1.6, linestyle='--')
    ax.add_patch(rect)

    # optional ellipse outline (nice visual cue)
    if show_ellipse:
        ell = Ellipse((cx, cy), 2*rx, 2*ry,
                      facecolor='none', edgecolor='tab:blue',
                      linewidth=1.2, alpha=0.7)
        ax.add_patch(ell)

    # scatter the sampling points
    if points:
        f1s, f2s = zip(*points)
        ax.scatter(f1s, f2s, s=25, marker='o',
                   edgecolors='k', facecolors='tab:green', alpha=0.8)

    # cosmetics ------------------------------------------------------------
    ax.set_xlabel("F1 (Hz)")
    ax.set_ylabel("F2 (Hz)")
    ax.set_title(f"Sampling points for {vowel_row['vowel']} ({vowel_row['group']})")

    if invert_f1:
        ax.invert_xaxis()    # F1 low-to-high, left-to-right
    ax.grid(True, linestyle=':', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')

    return ax

def max_intersections_between_spaces(vowel_row_a,
                                     vowel_row_b,
                                     df_all,
                                     *,
                                     block_res=(10, 10),
                                     scale=1.0,
                                     group=None):
    """
    For each sampling point in vowel-A’s space, compute the maximum number
    of vowel-ellipse intersections (via `count_ellipse_intersections`)
    that occur on a line segment to *any* point in vowel-B’s space, and
    do the symmetrical calculation for vowel-B’s points.

    Parameters
    ----------
    vowel_row_a, vowel_row_b : pd.Series (or dict-like)
        Rows for the two target vowels (must contain F1/F2 mean & SD).
    df_all : pd.DataFrame
        The complete vowel table – *including* the two target rows.
    block_res : tuple, default (10, 10)
        Grid resolution in Hz passed to `generate_sampling_points`.
    scale : float, default 1.0
        Must match the scale you use elsewhere (1.0 → ±2 sd ellipses).
    group : str | iterable | None, default None
        Optional group filter forwarded to `count_ellipse_intersections`.

    Returns
    -------
    result : dict
        {
          "A": { (F1, F2): max_hits, ... },
          "B": { (F1, F2): max_hits, ... }
        }
        Keys "A" / "B" correspond to `vowel_row_a` / `vowel_row_b`.
        Each sub-dict maps a sampling-grid point to its highest
        intersection count.
    """
    # --- build the two point clouds --------------------------------------
    pts_a = generate_sampling_points(vowel_row_a, block_res, scale)
    pts_b = generate_sampling_points(vowel_row_b, block_res, scale)

    # convenience: choose label strings once for nicer output
    label_a = f"{vowel_row_a['vowel']}({vowel_row_a['group']})"
    label_b = f"{vowel_row_b['vowel']}({vowel_row_b['group']})"

    res_a = {}
    for p in pts_a:
        best = 0
        for q in pts_b:
            n, _ = count_ellipse_intersections(df_all, p, q,
                                               scale=scale,
                                               group=group)
            if n > best:
                best = n
                # quick break if we already hit the theoretical max
                # (all vowels); useful when the inventory is small
                if best == len(df_all):
                    break
        res_a[p] = best
    print(f"Done {label_a}: {len(pts_a)} grid points processed")

    res_b = {}
    for q in pts_b:
        best = 0
        for p in pts_a:
            n, _ = count_ellipse_intersections(df_all, q, p,
                                               scale=scale,
                                               group=group)
            if n > best:
                best = n
                if best == len(df_all):
                    break
        res_b[q] = best
    print(f"Done {label_b}: {len(pts_b)} grid points processed")

    return {"A": res_a, "B": res_b}


def plot_intersection_heatmap(vowel_row_a,
                              vowel_row_b,
                              result_dict,
                              df_all,
                              *,
                              which="A",           # "A", "B" or "both"
                              cmap="turbo",
                              scale=1.0,
                              point_size=50,
                              invert_f1=True,
                              ax=None,
                              show_colorbar=True):
    """
    Parameters
    ----------
    vowel_row_a, vowel_row_b : pd.Series
        The two target vowel rows (same objects you passed to
        `max_intersections_between_spaces`).
    result_dict : dict
        The dict returned by `max_intersections_between_spaces`.
    df_all : pd.DataFrame
        Complete vowel table; all ellipses are drawn from this.
    which : {"A", "B", "both"}, default "A"
        Which set of sampling points to colour:
          "A" → vowel-A points,
          "B" → vowel-B points,
          "both" → combine the two (uses A’s value for overlaps).
    cmap : str or Colormap, default "viridis"
        Colormap for the heatmap.
    scale : float, default 1.0
        Matches the ±2 sd (or other) scale used everywhere else.
    point_size : int, default 50
        Marker area in scatter points.
    invert_f1 : bool, default True
        Plot F1 downward (common in vowel plots).
    ax : matplotlib.axes.Axes | None
        Plot into an existing axes if supplied.
    show_colorbar : bool, default True
        Add a colour bar at the side.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    # ---------------------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # choose which grid(s) to plot
    pts_val = {}
    if which in ("A", "both"):
        pts_val.update(result_dict["A"])
    if which in ("B", "both"):
        # if a point appears in both dictionaries, keep A’s value
        for k, v in result_dict["B"].items():
            pts_val.setdefault(k, v)

    if not pts_val:
        raise ValueError('No points to plot – check "which" argument.')

    # scatter the heat-map -----------------------------------------
    xs, ys, zs = zip(*[(f1, f2, v) for (f1, f2), v in pts_val.items()])
    sc = ax.scatter(xs, ys, c=zs, cmap=cmap, s=point_size,
                    edgecolors="none")

    if show_colorbar:
        plt.colorbar(sc, ax=ax, label="max # intersections")

    # draw every vowel ellipse -------------------------------------
    for _, row in df_all.iterrows():
        cx, cy = row["f1_mean"], row["f2_mean"]
        rx     = row["f1_sd"] * scale
        ry     = row["f2_sd"] * scale
        ell = Ellipse((cx, cy), 2*rx, 2*ry,
                      facecolor="none",
                      edgecolor="black",
                      linewidth=0.8,
                      alpha=0.4)
        ax.add_patch(ell)

    # cosmetics -----------------------------------------------------
    lbl_a = f"{vowel_row_a['vowel']}({vowel_row_a['group']})"
    lbl_b = f"{vowel_row_b['vowel']}({vowel_row_b['group']})"
    title_map = {"A": lbl_a, "B": lbl_b, "both": f"{lbl_a} & {lbl_b}"}
    ax.set_title(f"Heat-map of intersection cost from {title_map[which]}")
    ax.set_xlabel("F1 (Hz)")
    ax.set_ylabel("F2 (Hz)")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.set_aspect("equal", adjustable="box")
    if invert_f1:
        ax.invert_xaxis()

    return ax

from itertools import combinations

def aggregate_pairwise_load(df,
                            group="m",
                            *,
                            block_res=(10, 10),
                            scale=1.0):
    """
    Sum the pair-wise max-intersection counts over *all* unordered
    vowel pairs for the chosen `group`.

    Returns
    -------
    load : dict
        { vowel : { point : total_hits, … }, … }
    """
    df_g = df[df["group"] == group]

    # --- cache rows + sampling grids ---------------------------------
    rows   = {v: df_g[df_g["vowel"] == v].iloc[0]
              for v in df_g["vowel"].unique()}
    grids  = {v: generate_sampling_points(rows[v], block_res, scale)
              for v in rows}
    # start counters at zero
    load   = {v: {pt: 0 for pt in grids[v]} for v in grids}

    # --- iterate over every unordered pair ---------------------------
    for vA, vB in combinations(rows, 2):
        res = max_intersections_between_spaces(rows[vA], rows[vB], df_g,
                                               block_res=block_res,
                                               scale=scale,
                                               group=group)
        for pt, val in res["A"].items():
            load[vA][pt] += val
        for pt, val in res["B"].items():
            load[vB][pt] += val

    return load



import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plot_total_load(load_dict,
                    df,
                    group="m",
                    *,
                    scale=1.0,
                    cmap="turbo",
                    point_size=45,
                    col_wrap=4):
    """
    Visualise the aggregated intersection load for every vowel
    in `load_dict` (one subplot per vowel).
    """
    df_g = df[df["group"] == group]
    vowels = sorted(load_dict.keys())

    ncols = col_wrap
    nrows = math.ceil(len(vowels) / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4, nrows * 4),
                             squeeze=False)

    all_vals = [val
                for grid in load_dict.values()
                for val in grid.values()]
    vmin, vmax = min(all_vals), max(all_vals)

    for idx, vowel in enumerate(vowels):
        ax  = axes[idx // ncols][idx % ncols]
        row = df_g[df_g["vowel"] == vowel].iloc[0]
        data = load_dict[vowel]

        xs, ys, zs = zip(*[(f1, f2, z) for (f1, f2), z in data.items()])
        sc = ax.scatter(xs, ys, c=zs, cmap=cmap,
                        s=point_size, edgecolors="none",
                        vmin=vmin, vmax=vmax)

        # ±scale·SD ellipse for context
        cx, cy = row["f1_mean"], row["f2_mean"]
        rx, ry = row["f1_sd"] * scale, row["f2_sd"] * scale
        ax.add_patch(Ellipse((cx, cy), 2*rx, 2*ry,
                             facecolor='none', edgecolor='black',
                             linewidth=0.8))

        ax.set_title(vowel)
        ax.set_aspect('equal', adjustable='box')
        ax.invert_xaxis()
        ax.grid(True, linestyle=':')

    # tidy blank cells if the grid is not full
    for j in range(len(vowels), nrows * ncols):
        fig.delaxes(axes[j // ncols][j % ncols])
    fig.subplots_adjust(right=0.78)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    # shared colour-bar
    sm = ScalarMappable(cmap=cmap,
                        norm=colors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])  # dummy; required by colourbar

    fig.colorbar(sm, cax=cbar_ax,
                 label="total intersections\n(across all other vowels)")

    fig.suptitle(f"Aggregated intersection load — {group} speakers",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def report_max_per_vowel(load_dict):
    """
    Print the grid-centre with the largest total-intersection
    score for each vowel in `load_dict`.
    """
    print("Highest-load point for each vowel:")
    for v, grid in sorted(load_dict.items()):
        best_pt, best_val = max(grid.items(), key=lambda kv: kv[1])
        print(f"  {v:>2}  →  {best_pt}   total = {best_val}")


def save_load_tsv(load_dict, outfile="vowel_load_points.tsv"):
    """
    Write the aggregated load to a TSV file with columns
    vowel <tab> total <tab> f1 <tab> f2
    """
    rows = []
    for vowel, grid in load_dict.items():
        for (f1, f2), total in grid.items():
            rows.append({"vowel": vowel, "total": total, "f1": f1, "f2": f2})
    df_out = pd.DataFrame(rows)
    df_out.to_csv(outfile, sep="\t", index=False)


if __name__ == '__main__':

    scale=0.8
    res = 5
    df = parse_vowel_dist_data('../../input_data/vowel_stats.txt')
    df=df[df["group"]=="cbm"]
    hits, hit_set = count_ellipse_intersections(
        df,
        start=(550, 1750),
        end=(462.5, 1312.5),
        scale=scale)  # → ±2 sd ellipses
    print("Number of vowels crossed:", hits)
    print("Which ones:", hit_set)

    row = df.loc[(df['vowel'] == 'AH') & (df['group'] == 'cbm')].iloc[0]


    points = generate_sampling_points(row, block_res=(res, res), scale=scale)
    print(len(points), "points")
    print(points[:10])  # first few centre-points
    plot_ploints(row, points)
    plt.show()

    row_a = df.loc[(df["vowel"] == "IY") & (df["group"] == "cbm")].iloc[0]
    row_b = df.loc[(df["vowel"] == "AH") & (df["group"] == "cbm")].iloc[0]

    result = max_intersections_between_spaces(row_a, row_b, df,
                                              block_res=(res, res),
                                              scale=scale)

    # result["A"] is a dict: {(f1c, f2c): max_hits, …}
    max_for_ih = max(result["A"].values())
    print("Most intersections from any /ɪ/ point to /æ/ space:", max_for_ih)

    plot_intersection_heatmap(row_a, row_b, result, df,
                              which="both",  # or "both"
                              scale=scale,
                              point_size=2)
    plt.show()

    load = aggregate_pairwise_load(df, group="cbm",
                                   block_res=(res, res),
                                   scale=scale)  # ±1 SD ellipses

    save_load_tsv(load, "male_vowel_grid_load.tsv")
    report_max_per_vowel(load)

    plot_total_load(load, df, group="m",
                    scale=scale,  # use the same SD scale
                    point_size=20,
                    col_wrap=3)  # 3 columns per row
    plt.show()

