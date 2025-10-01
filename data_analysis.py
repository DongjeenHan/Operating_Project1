import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Convert numeric columns safely
def coerce_numeric(df):
    for col in ["Protein(g)", "Carbs(g)", "Fat(g)"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# Fill missing numeric values with column mean
def fill_missing_numeric(df):
    for col in ["Protein(g)", "Carbs(g)", "Fat(g)"]:
        df[col] = df[col].fillna(df[col].mean(skipna=True))
    return df

# Compute ratios
def compute_ratios(df):
    df["Protein_to_Carbs_ratio"] = np.where(df["Carbs(g)"] != 0,
                                            df["Protein(g)"] / df["Carbs(g)"], np.nan)
    df["Carbs_to_Fat_ratio"] = np.where(df["Fat(g)"] != 0,
                                        df["Carbs(g)"] / df["Fat(g)"], np.nan)
    return df

# Average macros per diet
def avg_macros_per_diet(df):
    return df.groupby("Diet_type")[["Protein(g)", "Carbs(g)", "Fat(g)"]].mean().sort_values("Protein(g)", ascending=False)

# Top-N protein recipes per diet
def topN_protein_per_diet(df, n):
    return df.sort_values("Protein(g)", ascending=False).groupby("Diet_type", group_keys=False).head(n)

# Most common cuisine per diet
def most_common_cuisine_by_diet(df):
    return (df.groupby("Diet_type")["Cuisine_type"]
              .agg(lambda x: x.value_counts().idxmax()))

# Summary of highest protein
def highest_protein_summary(df, avg_df):
    max_row = df.loc[df["Protein(g)"].idxmax()]
    max_single_diet = str(max_row["Diet_type"])
    max_single_value = float(max_row["Protein(g)"])
    avg_top = avg_df.iloc[0]
    max_avg_diet = avg_df.index[0]
    max_avg_value = float(avg_top["Protein(g)"])
    return {
        "diet_with_highest_single_recipe_protein": max_single_diet,
        "highest_single_recipe_protein_g": max_single_value,
        "diet_with_highest_avg_protein": max_avg_diet,
        "highest_avg_protein_g": max_avg_value
    }

# Plot functions
def plot_bar_avg_macros(avg_df, outdir):
    plt.figure(figsize=(10,6))
    x = np.arange(len(avg_df.index))
    width = 0.25
    plt.bar(x - width, avg_df["Protein(g)"], width, label="Protein(g)")
    plt.bar(x, avg_df["Carbs(g)"], width, label="Carbs(g)")
    plt.bar(x + width, avg_df["Fat(g)"], width, label="Fat(g)")
    plt.xticks(x, avg_df.index, rotation=20, ha="right")
    plt.title("Average Macronutrients by Diet Type")
    plt.ylabel("grams")
    plt.legend()
    path = os.path.join(outdir, "bar_avg_macros_by_diet.png")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    return path

def plot_heatmap_avg_macros(avg_df, outdir):
    data = avg_df[["Protein(g)","Carbs(g)","Fat(g)"]].values
    plt.figure(figsize=(8, max(4, len(avg_df.index)*0.4)))
    plt.imshow(data, aspect="auto")
    plt.colorbar(label="grams")
    plt.yticks(range(len(avg_df.index)), avg_df.index)
    plt.xticks(range(3), ["Protein(g)","Carbs(g)","Fat(g)"])
    plt.title("Heatmap: Average Macros by Diet Type")
    path = os.path.join(outdir, "heatmap_avg_macros_by_diet.png")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    return path

def plot_scatter_topN(top_df, outdir):
    if top_df.empty:
        return None
    plt.figure(figsize=(10,6))
    for d, g in top_df.groupby("Diet_type"):
        plt.scatter(g["Carbs(g)"], g["Protein(g)"], label=d, alpha=0.7)
    plt.xlabel("Carbs (g)")
    plt.ylabel("Protein (g)")
    plt.title("Top-N Protein-Rich Recipes (by Diet)")
    plt.legend(fontsize=8)
    path = os.path.join(outdir, "scatter_topN_protein.png")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    return path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--topn", type=int, default=5)
    parser.add_argument("--show", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load dataset
    df = pd.read_csv(args.csv)

    # Ensure required columns
    required = ["Diet_type", "Recipe_name", "Cuisine_type", "Protein(g)", "Carbs(g)", "Fat(g)"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"missing column: {c}")

    # Clean and process
    df = coerce_numeric(df)
    df = fill_missing_numeric(df)
    df = compute_ratios(df)

    # Save intermediate cleaned data
    df.to_csv(os.path.join(args.out, "cleaned_with_ratios.csv"), index=False)

    # Analysis
    avg_df = avg_macros_per_diet(df)
    avg_df.to_csv(os.path.join(args.out, "avg_macros_by_diet.csv"))

    top_df = topN_protein_per_diet(df, args.topn)
    top_df.to_csv(os.path.join(args.out, "topN_protein_by_diet.csv"), index=False)

    mc_df = most_common_cuisine_by_diet(df)
    mc_df.to_csv(os.path.join(args.out, "most_common_cuisine_by_diet.csv"))

    summary = highest_protein_summary(df, avg_df)
    pd.DataFrame([summary]).to_csv(os.path.join(args.out, "highest_protein_summary.csv"), index=False)

    # Plots
    b = plot_bar_avg_macros(avg_df, args.out)
    h = plot_heatmap_avg_macros(avg_df, args.out)
    s = plot_scatter_topN(top_df, args.out)

    # Optional display
    if args.show == 1:
        for path in [b, h, s]:
            if path:
                img_arr = plt.imread(path)
                plt.figure(); plt.imshow(img_arr); plt.axis("off"); plt.title(os.path.basename(path)); plt.show()

if __name__ == "__main__":
    main()
