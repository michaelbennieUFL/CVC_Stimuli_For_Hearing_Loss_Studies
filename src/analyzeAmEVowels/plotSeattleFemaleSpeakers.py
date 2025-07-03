import matplotlib.pyplot as plt

# Data extracted from Appendix A (female speakers, pooled) https://doi.org/10.1017/S0954394514000234
vowels_data = {
    'iy': {'F1': [364.10, 357.28, 357.25], 'F2': [2403.64, 2446.21, 2311.52]},
    'ɪ':  {'F1': [431.47, 451.77, 447.55], 'F2': [2121.44, 2063.22, 1985.62]},
    'ey': {'F1': [459.09, 435.24, 420.63], 'F2': [2263.40, 2371.50, 2324.23]},
    'ɛ':  {'F1': [545.90, 575.68, 551.80], 'F2': [1994.77, 1959.07, 1914.96]},
    'æ': {'F1': [639.38, 693.94, 677.89], 'F2': [1998.45, 1933.76, 1845.98]},
    'a':  {'F1': [715.13, 752.36, 712.33], 'F2': [1314.55, 1327.76, 1435.68]},
    'ɑ':  {'F1': [671.42, 701.82, 672.46], 'F2': [1243.87, 1253.58, 1339.93]},
    'ow': {'F1': [494.29, 457.12, 422.43], 'F2': [1195.86, 1097.74, 1124.84]},
    'ɔ':  {'F1': [648.76, 673.48, 669.54], 'F2': [1301.98, 1216.08, 1319.80]},
    'uw': {'F1': [373.95, 370.74, 369.82], 'F2': [1513.36, 1434.95, 1449.14]},
    'ʊ':  {'F1': [445.18, 462.76, 450.60], 'F2': [1401.02, 1450.63, 1545.70]},
    'ʌ':  {'F1': [629.18, 640.14, 588.65], 'F2': [1582.90, 1590.92, 1623.78]},
    'ɔj': {'F1': [468.42, 482.92, 423.56], 'F2': [1141.06, 1525.59, 2147.51]},
    'aj': {'F1': [735.75, 655.89, 501.46], 'F2': [1629.36, 1868.72, 2106.95]},
    'aw': {'F1': [734.76, 702.29, 579.66], 'F2': [1676.87, 1444.56, 1267.59]},
    'ɝ': {'F1': [456.25, 436.42, 420.36], 'F2': [1610.97, 1620.26, 1709.74]},
}

fig, ax = plt.subplots(figsize=(8, 10))

for vowel, vals in vowels_data.items():
    f1 = vals['F1']
    f2 = vals['F2']
    # Plot points
    ax.plot(f2, f1, marker='o')
    # Connect with arrows: 20%->50%, 50%->80%
    ax.annotate(
        '', xy=(f2[1], f1[1]), xytext=(f2[0], f1[0]),
        arrowprops=dict(arrowstyle='->')
    )
    ax.annotate(
        '', xy=(f2[2], f1[2]), xytext=(f2[1], f1[1]),
        arrowprops=dict(arrowstyle='->')
    )
    # Label at midpoint
    ax.text(f2[1], f1[1], vowel, ha='center', va='center', fontsize=9)

ax.set_xlabel('F2 (Hz)')
ax.set_ylabel('F1 (Hz)')
ax.set_title('Seattle Female Vowel Trajectories (20%-50%-80%)')

# Reverse axes to match conventional vowel plots
ax.invert_xaxis()
ax.invert_yaxis()

plt.tight_layout()
plt.show()
