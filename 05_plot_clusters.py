import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 日本語フォントの設定（Windows向けの一般的なフォント）
plt.rcParams['font.family'] = ['Meiryo', 'MS Gothic', 'Yu Gothic']

DIR_PATH = Path(r"G:\マイドライブ\02_Projects\分析_exp\00_datasets_visualize\国勢調査_メッシュ年齢分布分析")
file_path = DIR_PATH / "analysis_250m.parquet"
out_img = DIR_PATH / "cluster_pyramids.png"

print("Loading data...")
df = pd.read_parquet(file_path)

prop_cols = [c for c in df.columns if c.startswith("prop_pop_") and c != "prop_pop_total"]
age_labels = [c.replace("prop_pop_", "") for c in prop_cols]
# ラベルを見やすく変換（例: 0_4 -> 0〜4歳）
display_labels = [label.replace('_', '〜') + '歳' if label != '95_over' else '95歳以上' for label in age_labels]

# クラスターごとの平均構成比を算出
cluster_means = df.groupby("cluster_id")[prop_cols].mean()

cluster_titles = {
    0: "【Cluster 0】バランス型高年齢地域",
    1: "【Cluster 1】ファミリー層地域",
    2: "【Cluster 2】超高齢施設地域",
    3: "【Cluster 3】学生・若年単身地域",
    4: "【Cluster 4】進行性高齢化地域",
    5: "【Cluster 5】若手社会人エリア"
}

print("Plotting figures...")
fig, axes = plt.subplots(3, 2, figsize=(14, 18), sharex=True)
axes = axes.flatten()

for i in range(6):
    ax = axes[i]
    if i in cluster_means.index:
        # %表記にするため100倍
        vals = cluster_means.loc[i].values * 100
        y_pos = range(len(vals))
        
        # 横向き棒グラフ（ピラミッドの片側のような表現）
        bars = ax.barh(y_pos, vals, color='cornflowerblue', edgecolor='black', alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(display_labels, fontsize=10)
        ax.set_title(cluster_titles.get(i, f"Cluster {i}"), fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel("人口割合 (%)", fontsize=12)
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # 最大目盛り
        ax.set_xlim(0, 30)

plt.tight_layout()
plt.savefig(out_img, dpi=200, bbox_inches='tight')
print(f"Saved figure to {out_img}")
