import json
import pandas as pd
import matplotlib.pyplot as plt
import os

input_path = "data/NH_subjects_data.json"  # adjust if needed

# Load JSON
with open(input_path, "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# 1) Vocab scores plot
fig1 = plt.figure()
plt.scatter(df["ID"], df["Vocab_score"], label="Participant Score")
plt.ylim(190, 370)

# Reference lines
mean_val = 302.4
minus_1sd = 259.2
plus_1sd = 360.0  # You called it "+1 1 SD" but values suggest mean + 1 SD and an upper bound

plt.axhline(mean_val, color="blue", linestyle="--", label="Mean")
plt.axhline(minus_1sd, color="red", linestyle="--", label="-1 SD")
plt.axhline(plus_1sd, color="red", linestyle="--", label="+1 SD")

plt.title("Vocabulary Scores with Mean and ±1 SD reference lines \n of previous AmE adult vocabulary studies")
plt.xlabel("Subject ID")
plt.ylabel("Vocabulary Score")
plt.legend()

os.makedirs("charts", exist_ok=True)
out1 = "./charts/vocab_scores.png"
plt.savefig(out1, bbox_inches="tight")
plt.show()

# 2) Pie chart of Origin
fig2 = plt.figure()
origin_counts = df["Origin"].value_counts()
plt.pie(origin_counts.values, labels=origin_counts.index, autopct="%1.1f%%")
plt.title("Origin Distribution")

out2 = "./charts/origin_pie.png"
plt.savefig(out2, bbox_inches="tight")
plt.show()

print(f"Saved:\n- {out1}\n- {out2}")
