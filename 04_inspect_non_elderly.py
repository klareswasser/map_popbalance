import pandas as pd
import numpy as np
from pathlib import Path

DIR_PATH = Path(r"G:\マイドライブ\02_Projects\分析_exp\00_datasets_visualize\国勢調査_メッシュ年齢分布分析")
file_path_250 = DIR_PATH / "analysis_250m.parquet"

df = pd.read_parquet(file_path_250)
prop_cols = [c for c in df.columns if c.startswith("prop_pop_") and c != "prop_pop_total"]

print("\n=== Cluster 3,5 High Gini (0-64) Meshes ===")
# Exclude Cluster 2 and 4 (which are elderly dominated) to find anomalies in working age
df_working_age = df[df["cluster_id"].isin([1, 3, 5])]
top_gini_working = df_working_age.sort_values("gini_0_64", ascending=False).head(20)

for idx, row in top_gini_working.iterrows():
    zero_to_64_cols = prop_cols[:13]
    vals = row[zero_to_64_cols]
    top_age = vals.idxmax().replace("prop_pop_", "")
    top_val = vals.max()
    
    print(f"Mesh: {row['KEY_CODE']} | Pop: {row['pop_total']} | Gini(0-64): {row['gini_0_64']:.3f} | Cluster: {row['cluster_id']} | Dominant: {top_age}歳 ({row[vals.idxmax()]*row['pop_total']:.0f}人, 割合: {top_val:.1%})")

