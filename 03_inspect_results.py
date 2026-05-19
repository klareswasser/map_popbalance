import pandas as pd
import numpy as np
from pathlib import Path

DIR_PATH = Path(r"G:\マイドライブ\02_Projects\分析_exp\00_datasets_visualize\国勢調査_メッシュ年齢分布分析")
file_path_250 = DIR_PATH / "analysis_250m.parquet"

print("Loading 250m analysis data...")
df = pd.read_parquet(file_path_250)

# Filter columns
prop_cols = [c for c in df.columns if c.startswith("prop_pop_") and c != "prop_pop_total"]
age_labels = [c.replace("prop_pop_", "") for c in prop_cols]

print("\n=== Cluster Profiles (250m) ===")
cluster_counts = df["cluster_id"].value_counts().sort_index()

for cluster_id in sorted(df["cluster_id"].dropna().unique()):
    subset = df[df["cluster_id"] == cluster_id]
    mean_props = subset[prop_cols].mean()
    top5 = mean_props.sort_values(ascending=False).head(5)
    
    print(f"\nCluster {cluster_id} (count: {len(subset):,}, {len(subset)/len(df):.1%})")
    print("  Top 5 age groups:")
    for col, val in top5.items():
        print(f"  - {col.replace('prop_pop_', '')}歳: {val:.1%}")
        
    # Check 0-64 top vs 65+ top
    pop_0_64_prop = mean_props[:13].sum()
    pop_65_over_prop = mean_props[13:].sum()
    print(f"  [0-64 ratio: {pop_0_64_prop:.1%} | 65+ ratio: {pop_65_over_prop:.1%}]")

print("\n=== Top High Gini (0-64) Meshes ===")
# Inspect top 20 meshes with highest 0-64 Gini
top_gini = df.sort_values("gini_0_64", ascending=False).head(20)

for idx, row in top_gini.iterrows():
    # Find which age group dominates among 0-64
    zero_to_64_cols = prop_cols[:13]
    vals = row[zero_to_64_cols]
    top_age = vals.idxmax().replace("prop_pop_", "")
    top_val = vals.max()
    
    print(f"Mesh: {row['KEY_CODE']} | Pop: {row['pop_total']} | Gini(0-64): {row['gini_0_64']:.3f} | Cluster: {row['cluster_id']} | Dominant (0-64): {top_age}歳 ({row[vals.idxmax()]*row['pop_total']:.0f}人, 割合: {top_val:.1%})")
