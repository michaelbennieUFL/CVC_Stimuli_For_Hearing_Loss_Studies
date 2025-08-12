import json
import re

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

input_path = "data/NH_subjects_data.json"  # adjust if needed

# Load JSON
with open(input_path, "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Chart 1: AE Vowel sounds (from Dichotic_Heard_vowels)
ae_vowel_counts = Counter([v for sublist in df["Dichotic_Heard_vowels"] for v in sublist])
plt.figure()
plt.pie(ae_vowel_counts.values(), labels=ae_vowel_counts.keys(), autopct="%1.1f%%")
plt.title("AE Vowel Sounds (Dichotic Heard Vowels)")
ae_vowel_chart_path = "./charts/pie_ae_vowel_sounds.png"
plt.savefig(ae_vowel_chart_path)
plt.close()

# Chart 2: Monaural sound quality (Mono_Word_Sounds)
plt.figure()
df["Mono_Word_Sounds"].value_counts().plot.pie(autopct="%1.1f%%", title="Monaural Whole Word Sound Quality")
mono_word_sounds_chart_path = "./charts/pie_mono_word_sounds.png"
plt.ylabel("")
plt.savefig(mono_word_sounds_chart_path)
plt.close()

# Chart 3: Monaural vowel sound quality (Mono_Word_vowels)
plt.figure()
df["Mono_Word_vowels"].value_counts().plot.pie(autopct="%1.1f%%", title="Monaural # Vowels Heard")
mono_vowel_sounds_chart_path = "./charts/pie_mono_vowel_sounds.png"
plt.ylabel("")
plt.savefig(mono_vowel_sounds_chart_path)
plt.close()

# Chart 4: Subjective AE sound quality (AE_Vowels parsed tokens)
# ---- Build per-participant token SET from AE_Vowels free text ----
def parse_tokens(s):
    if pd.isna(s):
        return set()
    # Remove "sounds like" phrase, split words, keep letters, uppercase
    s = s.lower().replace("sounds", "").replace("like", "")
    toks = re.findall(r"[A-Za-z]+", s)
    toks = [t.upper() for t in toks if t.strip()]
    # keep short code-like tokens only (heuristic: length 1-3)
    toks = [t for t in toks if 1 <= len(t) <= 3]
    return set(toks)

token_sets = df["AE_Vowels"].apply(parse_tokens)

# All tokens observed across participants
all_tokens = sorted(set().union(*token_sets.tolist()))
n_participants = len(df)

# Relative frequency = % of participants that included the token at least once
rel_freq = {
    tok: 100.0 * sum(tok in s for s in token_sets) / n_participants
    for tok in all_tokens
}

# Make a bar chart
tokens = list(rel_freq.keys())
values = [rel_freq[t] for t in tokens]

out_path = "./charts/subjective_ae_relative_freq.png"

plt.figure()
plt.bar(tokens, values)
plt.ylim(0, 110)
plt.ylabel("Participants endorsing token (%)")
plt.xlabel("Token")
plt.title("Subjective AE: Relative Frequency by Token (per-participant)")
for i, v in enumerate(values):
    plt.text(i, v + 1, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
plt.savefig(out_path, bbox_inches="tight")
plt.show()

(ae_vowel_chart_path, mono_word_sounds_chart_path, mono_vowel_sounds_chart_path, subjective_ae_chart_path)
