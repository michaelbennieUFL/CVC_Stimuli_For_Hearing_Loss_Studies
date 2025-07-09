import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


# --- Frequency conversion functions ---
def hz_to_erb(f):
    return 214 * np.log10(0.00437 * f + 1)


def hz_to_mel(f):
    return 1127.01048 * np.log(1 + f / 700)


def hz_to_bark(f):
    return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7000) ** 2)


# Set dynamic axis limits with 20% padding
def padded_limits(values):
    min_v, max_v = min(values), max(values)
    padding = 0.2 * (max_v - min_v)
    return (max_v + padding, min_v - padding)  # reversed for vowel space


# --- Plotting Function ---
def plot_vowel_gaussians_scales(
        file_path,
        gender='m',
        scales=('hz', 'bark', 'mel', 'erb'),
        xlim_hz=(3500, 700),
        ylim_hz=(1100, 200),
        max_sd=5000,
        scale=0.5
):
    # --- Parse file into DataFrame ---
    data_dict = {}
    measures = ['Duration', 'F0', 'F1', 'F2', 'F3', 'F4']
    with open(file_path, 'r') as file:
        for line in file:
            tokens = line.strip().split()
            if len(tokens) < 8:
                continue
            vowel, group, mean, sd, *_, measure = tokens
            if measure not in measures:
                continue
            key = (vowel, group)
            if key not in data_dict:
                data_dict[key] = {}
            data_dict[key][f"{measure.lower()}_mean"] = float(mean)
            data_dict[key][f"{measure.lower()}_sd"] = float(sd)

    df = pd.DataFrame.from_dict(data_dict, orient='index')
    df.index = pd.MultiIndex.from_tuples(df.index, names=['vowel', 'group'])
    df = df.reset_index()

    # Filter by gender and remove outliers
    sub = df[df['group'] == gender]
    sub = sub[(sub['f1_sd'] < max_sd) & (sub['f2_sd'] < max_sd)]

    # --- Setup Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    scale_funcs = {
        'hz': lambda x: x,
        'bark': hz_to_bark,
        'mel': hz_to_mel,
        'erb': hz_to_erb
    }

    titles = {
        'hz': 'Normal (Hz)',
        'bark': 'Bark Scale',
        'mel': 'Mel Scale (Cent)',
        'erb': 'ERB Scale'
    }

    for ax, scale_type in zip(axes, scales):
        convert = scale_funcs[scale_type]

        f1_vals = []
        f2_vals = []

        for _, row in sub.iterrows():
            f2_mean = convert(row['f2_mean'])
            f1_mean = convert(row['f1_mean'])
            f2_sd = abs(convert(row['f2_mean'] + row['f2_sd']) - f2_mean)
            f1_sd = abs(convert(row['f1_mean'] + row['f1_sd']) - f1_mean)

            ellipse = Ellipse(
                (f2_mean, f1_mean),
                width=2 * f2_sd * scale,
                height=2 * f1_sd * scale,
                edgecolor='blue', facecolor='none', lw=2, alpha=0.7
            )
            ax.add_patch(ellipse)
            ax.text(f2_mean, f1_mean, row['vowel'],
                    fontsize=10, ha='center', va='center')

            # collect values for limits
            f1_vals.append(f1_mean)
            f2_vals.append(f2_mean)



        ax.set_xlim(padded_limits(f2_vals))
        ax.set_ylim(padded_limits(f1_vals))

        # ax.invert_xaxis()
        # ax.invert_yaxis()

    plt.tight_layout()
    plt.show()


# --- Example Usage ---
if __name__ == "__main__":
    plot_vowel_gaussians_scales(
        '../../input_data/vowel_stats.txt',
        gender='m',
        scales=('hz', 'bark', 'mel', 'erb'),
        scale=1
    )
    plot_vowel_gaussians_scales(
        '../../input_data/vowel_stats.txt',
        gender='cbm',
        scales=('hz', 'bark', 'mel', 'erb'),
        scale=1
    )
    plot_vowel_gaussians_scales(
        '../../input_data/vowel_stats.txt',
        gender='NWP_And_Citation',
        scales=('hz', 'bark', 'mel', 'erb'),
        scale=1
    )

    plot_vowel_gaussians_scales(
        '../../input_data/vowel_stats.txt',
        gender='NWP_And_Reading',
        scales=('hz', 'bark', 'mel', 'erb'),
        scale=1
    )