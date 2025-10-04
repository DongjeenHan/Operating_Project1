# data_analysis.py
# Run from inside Task1_app:
#   python data_analysis.py --csv data/All_Diets.csv --out outputs

import argparse, os, json, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

# ---------- Utility Functions ----------

def coerce_numeric(df):
    """Convert macro columns to numeric safely."""
    for col in ["Protein(g)", "Carbs(g)", "Fat(g)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def fill_missing_numeric(df):
    """Fill missing numeric values with column mean."""
    for col in ["Protein(g)", "Carbs(g)", "Fat(g)"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean(skipna=True))
    return df

def compute_ratios(df):
    """Add Protein/Carbs and Carbs/Fat ratios with safe division."""
    df["Protein_to_Carbs_ratio"] = np.where(df["Carbs(g)"] != 0,
                                            df["Protein(g)"] / df["Carbs(g)"], np.nan)
    df["Carbs_to_Fat_ratio"] = np.where(df["Fat(g)"] != 0,
                                        df["Carbs(g)"] / df["Fat(g)"], np.nan)
    return df

# ---------- Core Analyses ----------

def avg_macros_per_diet(df):
    return (df.groupby("Diet_type")[["Protein(g)", "Carbs(g)", "Fat(g)"]]
              .mean()
              .sort_values("Protein(g)", ascending=False))

def topN_protein_per_diet(df, n):
    return (df.sort_values("Protein(g)", ascending=False)
              .groupby("Diet_type", group_keys=False)
              .head(n))

def most_common_cuisine_by_diet(df):
    s = (df.groupby("Diet_type")["Cuisine_type"]
           .agg(lambda x: x.value_counts().idxmax())
           .reset_index())
    s.columns = ["Diet_type", "Most_Common_Cuisine"]
    return s

def highest_protein_summary(df, avg_df):
    max_recipe = df.loc[df["Protein(g)"].idxmax()]
    single_diet = str(max_recipe["Diet_type"])
    single_val = float(max_recipe["Protein(g)"])
    max_avg_diet = avg_df.index[0]
    max_avg_val = float(avg_df.iloc[0]["Protein(g)"])
    return {
        "diet_with_highest_single_recipe_protein": single_diet,
        "highest_single_recipe_protein_g": single_val,
        "diet_with_highest_avg_protein": max_avg_diet,
        "highest_avg_protein_g": max_avg_val
    }

# ---------- Extra Analyses (nice-to-have) ----------

def avg_macros_by_cuisine(df):
    return df.groupby("Cuisine_type")[["Protein(g)", "Carbs(g)", "Fat(g)"]].mean().round(2)

def recipe_counts_by_diet(df):
    return df["Diet_type"].value_counts().rename_axis("Diet_type").reset_index(name="Recipe_Count")

def macros_correlation(df):
    return df[["Protein(g)", "Carbs(g)", "Fat(g)"]].corr()

def macro_outliers(df):
    q = df[["Protein(g)", "Carbs(g)", "Fat(g)"]].quantile(0.99)
    return df[(df["Protein(g)"] > q["Protein(g)"]) |
              (df["Carbs(g)"] > q["Carbs(g)"]) |
              (df["Fat(g)"] > q["Fat(g)"])]

# ---------- Plot Functions ----------

def plot_bar_avg_macros(avg_df, outdir):
    plt.figure(figsize=(10,6))
    avg_melt = avg_df.reset_index().melt(id_vars="Diet_type", var_name="Macro", value_name="Grams")
    sns.barplot(data=avg_melt, x="Diet_type", y="Grams", hue="Macro")
    plt.title("Average Macronutrients by Diet Type")
    plt.ylabel("Grams")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path = os.path.join(outdir, "bar_avg_macros_by_diet.png")
    plt.savefig(path, dpi=150); plt.close()
    return path

def plot_heatmap_avg_macros(avg_df, outdir):
    plt.figure(figsize=(8, max(4, len(avg_df.index)*0.4)))
    sns.heatmap(avg_df, annot=True, cmap="YlGnBu", fmt=".1f")
    plt.title("Heatmap: Average Macros by Diet Type")
    plt.tight_layout()
    path = os.path.join(outdir, "heatmap_avg_macros_by_diet.png")
    plt.savefig(path, dpi=150); plt.close()
    return path

def plot_scatter_topN(top_df, outdir):
    if top_df.empty:
        return None
    plt.figure(figsize=(10,6))
    sns.scatterplot(data=top_df, x="Carbs(g)", y="Protein(g)", hue="Diet_type", s=120, edgecolor="black")
    plt.xlabel("Carbs (g)"); plt.ylabel("Protein (g)")
    plt.title("Top-N Protein-Rich Recipes by Diet Type")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(outdir, "scatter_topN_protein.png")
    plt.savefig(path, dpi=150); plt.close()
    return path

# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--topn", type=int, default=5)
    parser.add_argument("--show", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.csv)

    required = ["Diet_type","Recipe_name","Cuisine_type","Protein(g)","Carbs(g)","Fat(g)"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # Clean & engineer
    df = compute_ratios(fill_missing_numeric(coerce_numeric(df)))
    df.to_csv(os.path.join(args.out, "cleaned_with_ratios.csv"), index=False)

    # Core outputs
    avg_df = avg_macros_per_diet(df)
    avg_df.to_csv(os.path.join(args.out, "avg_macros_by_diet.csv"))
    top_df = topN_protein_per_diet(df, args.topn)
    top_df.to_csv(os.path.join(args.out, "topN_protein_by_diet.csv"), index=False)
    mc_df = most_common_cuisine_by_diet(df)
    mc_df.to_csv(os.path.join(args.out, "most_common_cuisine_by_diet.csv"), index=False)
    summary = highest_protein_summary(df, avg_df)
    pd.DataFrame([summary]).to_csv(os.path.join(args.out, "highest_protein_summary.csv"), index=False)

    # Extras
    avg_cuisine = avg_macros_by_cuisine(df)
    avg_cuisine.to_csv(os.path.join(args.out, "avg_macros_by_cuisine.csv"))
    counts = recipe_counts_by_diet(df)
    counts.to_csv(os.path.join(args.out, "recipe_counts_by_diet.csv"), index=False)
    corr = macros_correlation(df)
    corr.to_csv(os.path.join(args.out, "macros_correlation.csv"))
    outliers = macro_outliers(df)
    outliers.to_csv(os.path.join(args.out, "macro_outliers.csv"), index=False)

    # Plots
    b = plot_bar_avg_macros(avg_df, args.out)
    h = plot_heatmap_avg_macros(avg_df, args.out)
    s = plot_scatter_topN(top_df, args.out)

    # Console summaries (handy for screenshots)
    print("\n=== Analysis Summary ===")
    print(pd.DataFrame([summary]).to_string(index=False))
    print("\nRecipe counts by diet:")
    print(counts.to_string(index=False))
    print("\nMacros correlation matrix:")
    print(corr.round(2).to_string())
    print("\nAll outputs saved to:", os.path.abspath(args.out))

    # JSON summary (useful for Cosmos-style ingestion)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "highest_protein": summary,
            "avg_macros_per_diet": avg_df.reset_index().to_dict(orient="records"),
            "most_common_cuisine": mc_df.to_dict(orient="records")
        }, f, indent=2)

if __name__ == "__main__":
    main()
