import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plot_vowel_f1f2_gaussians(
    file_path,
    gender='m',
    xlim=(3200, 1000),  # F2 range (Hz), reversed for vowel space
    ylim=(900, 200),    # F1 range (Hz), reversed for vowel space
    max_sd=5000
):
    # --- Parse the file and make a DataFrame ---
    data_dict = {}
    measures = ['Duration', 'F0', 'F1', 'F2', 'F3', 'F4']
    with open(file_path, 'r') as file:
        for line in file:
            tokens = line.strip().split()
            if len(tokens) < 8:
                continue
            vowel, group, mean, sd, *_ , measure = tokens
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

    # --- Filter by gender and remove outliers ---
    sub = df[df['group'] == gender]
    sub = sub[(sub['f1_sd'] < max_sd) & (sub['f2_sd'] < max_sd)]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in sub.iterrows():
        ellipse = Ellipse(
            (row['f2_mean'], row['f1_mean']),
            width=2*row['f2_sd'],
            height=2*row['f1_sd'],
            edgecolor='blue', facecolor='none', lw=2, alpha=0.7
        )
        ax.add_patch(ellipse)
        ax.text(row['f2_mean'], row['f1_mean'], row['vowel'],
                fontsize=10, ha='center', va='center')
    ax.set_xlabel('F2 Mean (Hz)')
    ax.set_ylabel('F1 Mean (Hz)')
    ax.set_title(f"{'Male' if gender=='m' else 'Female'} Vowel Gaussians: 1σ Ellipses (F1/F2)")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # ax.invert_xaxis()
    # ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

if __name__ =="__main__":
    plot_vowel_f1f2_gaussians('../../input_data/vowelDistData.txt', gender='m', xlim=(3500, 700), ylim=(1100, 200))
    plot_vowel_f1f2_gaussians('../../input_data/vowelDistData.txt', gender='w', xlim=(3500, 700), ylim=(1100, 200))
