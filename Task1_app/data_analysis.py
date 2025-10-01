# data_analysis.py
# Place this file at: Operating_Project1/Task1_app/data_analysis.py
# Run from inside Task1_app: python data_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import sys
import warnings
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent.resolve()
DATA_PATH = BASE / "data" / "All_Diets.csv"
OUTPUT_DIR = BASE / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Helpful logging
print(f"Base folder: {BASE}")
print(f"Looking for CSV at: {DATA_PATH}")

if not DATA_PATH.exists():
    print("ERROR: All_Diets.csv not found at:", DATA_PATH)
    print("Please put the CSV into the data/ folder and re-run.")
    sys.exit(2)

# Read with python engine in case of odd quoting in the CSV
df = pd.read_csv(DATA_PATH, engine="python", encoding="utf-8")

# Normalize possible column name variants (be tolerant)
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

protein_col = find_col(df, ["Protein(g)", "Protein (g)", "Protein_g", "Protein"])
carbs_col   = find_col(df, ["Carbs(g)", "Carbs (g)", "Carbs_g", "Carbs"])
fat_col     = find_col(df, ["Fat(g)", "Fat (g)", "Fat_g", "Fat"])

# Basic column checks
if not protein_col or not carbs_col or not fat_col:
    print("ERROR: Could not find expected macro columns. Found columns:")
    print(list(df.columns)[:40])
    sys.exit(3)

# Ensure Diet_type and Cuisine_type exist
if "Diet_type" not in df.columns:
    print("ERROR: 'Diet_type' column missing.")
    sys.exit(4)
if "Cuisine_type" not in df.columns:
    df["Cuisine_type"] = "Unknown"

# Coerce numeric columns safely
for col in [protein_col, carbs_col, fat_col]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill numeric NaNs with column mean
for col in [protein_col, carbs_col, fat_col]:
    mean_val = df[col].mean(skipna=True)
    df[col] = df[col].fillna(mean_val)

# Compute ratios safely (avoid divide-by-zero)
df["Protein_to_Carbs_ratio"] = df[protein_col] / df[carbs_col].replace({0: np.nan})
df["Carbs_to_Fat_ratio"] = df[carbs_col] / df[fat_col].replace({0: np.nan})
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Average macros per diet
avg_macros = df.groupby("Diet_type")[[protein_col, carbs_col, fat_col]].mean().round(2)
avg_macros.to_csv(OUTPUT_DIR / "avg_macros_per_diet.csv")
print("Saved avg_macros_per_diet.csv")

# Top 5 protein-rich recipes per diet
top_protein = df.sort_values(by=protein_col, ascending=False).groupby("Diet_type").head(5)
top_protein.to_csv(OUTPUT_DIR / "top_5_protein_per_diet.csv", index=False)
print("Saved top_5_protein_per_diet.csv")

# Most common cuisines per diet (top 3)
common_cuisines = (df.groupby("Diet_type")["Cuisine_type"]
                     .apply(lambda s: s.value_counts().head(3).to_dict())
                     .reset_index(name="top_cuisines"))
common_cuisines.to_json(OUTPUT_DIR / "common_cuisines_per_diet.json", orient="records")
print("Saved common_cuisines_per_diet.json")

# Save full dataframe with ratios
df.to_csv(OUTPUT_DIR / "all_recipes_with_ratios.csv", index=False)
print("Saved all_recipes_with_ratios.csv")

# --- Visualizations ---
sns.set(style="whitegrid")

# 1) Bar chart: average macros by diet (stacked view via melt)
avg_reset = avg_macros.reset_index().melt(id_vars="Diet_type", var_name="Macro", value_name="Grams")
plt.figure(figsize=(10,6))
sns.barplot(data=avg_reset, x="Diet_type", y="Grams", hue="Macro")
plt.xticks(rotation=45, ha="right")
plt.title("Average Macros by Diet Type")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "avg_macros_by_diet.png")
plt.close()
print("Saved avg_macros_by_diet.png")

# 2) Heatmap: avg macros (diet x macro)
plt.figure(figsize=(8,6))
sns.heatmap(avg_macros, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Heatmap - Average Macros (g) per Diet Type")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "heatmap_macros_by_diet.png")
plt.close()
print("Saved heatmap_macros_by_diet.png")

# 3) Scatter: top protein recipes (protein vs carbs)
if not top_protein.empty:
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=top_protein, x=protein_col, y=carbs_col, hue="Diet_type", s=100)
    plt.title("Top Protein Recipes (protein vs carbs)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_protein_scatter.png")
    plt.close()
    print("Saved top_protein_scatter.png")

# Save a compact JSON summary (useful for simulated NoSQL)
summary = {
    "avg_macros_per_diet": avg_macros.reset_index().to_dict(orient="records"),
    "top_5_protein_per_diet": top_protein.to_dict(orient="records"),
}
with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print("Saved summary.json")

print("\nALL DONE -- outputs are in:", OUTPUT_DIR)

