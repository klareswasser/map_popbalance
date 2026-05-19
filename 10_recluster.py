"""
10_recluster.py
人口100人以上のメッシュを対象に K-Means (k=6) クラスタリングを再実施。
- analysis_500m.parquet を上書き更新
- cluster_pyramids_v2.png に年齢分布図を出力
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DIR_PATH = Path(__file__).parent

# ── 日本語フォント設定 ──
def find_jp_font():
    candidates = [
        "Hiragino Sans",
        "Hiragino Kaku Gothic ProN",
        "Hiragino Maru Gothic Pro",
        "Arial Unicode MS",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in available:
            return c
    return None

jp_font = find_jp_font()
if jp_font:
    plt.rcParams["font.family"] = jp_font
else:
    print("WARNING: Japanese font not found, labels may be garbled")


# ── データ読み込み ──
MIN_POP = 150
print(f"Loading population_500m.parquet ...")
df = pd.read_parquet(DIR_PATH / "population_500m.parquet")
print(f"Total meshes: {len(df):,}")

age_cols = [c for c in df.columns if c.startswith("pop_") and c != "pop_total"]
# ── フィルタリング ──
mask = df["pop_total"] >= MIN_POP
df_tgt = df[mask].copy()
print(f"Meshes with pop >= {MIN_POP}: {len(df_tgt):,}  (excluded: {(~mask).sum():,})")

# ── 年齢割合計算 ──
prop_cols = []
for col in age_cols:
    pc = f"prop_{col}"
    df_tgt[pc] = df_tgt[col] / df_tgt["pop_total"]
    prop_cols.append(pc)

# ── K-Means クラスタリング（標準化 + PCA） ──
N_CLUSTERS = 6
print(f"Running StandardScaler + PCA + K-Means (k={N_CLUSTERS}) ...")

X_raw = df_tgt[prop_cols].fillna(0).values

# 1) StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# 2) PCA — 95%分散説明に必要な成分数
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"  PCA: {X_raw.shape[1]} features → {X_pca.shape[1]} components "
      f"(explained var: {pca.explained_variance_ratio_.sum()*100:.1f}%)")

# 3) K-Means
km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20)
df_tgt["cluster_id"] = km.fit_predict(X_pca)

# ── クラスター統計の出力 ──
AGE_LABELS = [c.replace("pop_", "") for c in age_cols]

# 平均年齢でクラスターIDを振り直す（若=0 → 老=N-1）
AGE_MIDPOINTS_DICT = {
    "0_4": 2, "5_9": 7, "10_14": 12, "15_19": 17, "20_24": 22,
    "25_29": 27, "30_34": 32, "35_39": 37, "40_44": 42, "45_49": 47,
    "50_54": 52, "55_59": 57, "60_64": 62, "65_69": 67, "70_74": 72,
    "75_79": 77, "80_84": 82, "85_89": 87, "90_94": 92, "95_over": 97,
}
midpoints = np.array([AGE_MIDPOINTS_DICT.get(l, 97) for l in AGE_LABELS])
mean_ages_raw = {}
for cid in range(N_CLUSTERS):
    sub_tmp = df_tgt[df_tgt["cluster_id"] == cid]
    mean_ages_raw[cid] = float(np.dot(sub_tmp[prop_cols].mean().values, midpoints))
sorted_cids = sorted(mean_ages_raw, key=lambda c: mean_ages_raw[c])
id_map = {old: new for new, old in enumerate(sorted_cids)}
df_tgt["cluster_id"] = df_tgt["cluster_id"].map(id_map)
print("  Remapped cluster IDs (young=0 → old=N-1):")
for new_id in range(N_CLUSTERS):
    old_id = sorted_cids[new_id]
    print(f"    new {new_id}: mean_age={mean_ages_raw[old_id]:.1f}y")

# グラデーション色（若→老: 鮮やか青→黄→鮮やか赤）
gradient_colors = [
    '#ff9800',  # 0: オレンジ
    '#4caf50',  # 1: 緑
    '#bdbdbd',  # 2: 薄めのグレー
    '#1565c0',  # 3: 濃いめの青
    '#f44336',  # 4: 赤
    '#795548',  # 5: 茶色
]
print("\n  Gradient colors (new IDs 0=young/blue … N-1=old/red):")
for i, c in enumerate(gradient_colors):
    print(f"    {i}: {c}")

CHILD_COLS   = [f"prop_{c}" for c in age_cols if c in [f"pop_{a}" for a in ["0_4","5_9","10_14"]]]
WORKING_COLS = [f"prop_{c}" for c in age_cols if c in
                [f"pop_{a}" for a in ["15_19","20_24","25_29","30_34","35_39",
                                       "40_44","45_49","50_54","55_59","60_64"]]]
ELDERLY_COLS = [f"prop_{c}" for c in age_cols if c in
                [f"prop_{c2}" for c2 in age_cols[13:]]]
ELDERLY_COLS = [f"prop_{c}" for c in age_cols[13:]]

print("\n" + "="*70)
print("CLUSTER STATISTICS")
print("="*70)
summary_rows = []
for cid in sorted(df_tgt["cluster_id"].unique()):
    sub = df_tgt[df_tgt["cluster_id"] == cid]
    n = len(sub)
    mean_pop = sub["pop_total"].mean()
    child_pct   = sub[CHILD_COLS].sum(axis=1).mean() * 100
    working_pct = sub[WORKING_COLS].sum(axis=1).mean() * 100
    elderly_pct = sub[ELDERLY_COLS].sum(axis=1).mean() * 100


    # 平均年齢分布（元の比率空間）から Top3 計算
    mean_props = sub[prop_cols].mean().values
    top3_idx = mean_props.argsort()[::-1][:3]
    top3 = [(AGE_LABELS[i], mean_props[i] * 100) for i in top3_idx]
    top3_str = ", ".join(f"{l}({v:.1f}%)" for l, v in top3)

    mean_age = float(np.dot(sub[prop_cols].mean().values, midpoints))

    print(f"\nCluster {cid}  (n={n:,}, mean_pop={mean_pop:.0f}, mean_age={mean_age:.1f}y)")
    print(f"  子供(0-14歳): {child_pct:.1f}%  |  現役(15-64歳): {working_pct:.1f}%  |  高齢(65+歳): {elderly_pct:.1f}%")
    print(f"  Top3 age: {top3_str}")

    top1_label = top3[0][0].replace("_", "〜") + "歳" if top3[0][0] != "95_over" else "95歳以上"

    summary_rows.append({
        "cluster_id": cid,
        "n_mesh": n,
        "mean_pop": mean_pop,
        "mean_age": mean_age,
        "top1_age": top1_label,
        "child_pct": child_pct,
        "working_pct": working_pct,
        "elderly_pct": elderly_pct,
    })

print("\n" + "="*70)
summary = pd.DataFrame(summary_rows)
print(summary.to_string(index=False))
print("="*70)

# ── analysis_500m.parquet に保存 ──
out_cols = ["KEY_CODE", "cluster_id"] + prop_cols
df_merged = df.merge(df_tgt[out_cols], on="KEY_CODE", how="left")
out_parquet = DIR_PATH / "analysis_500m.parquet"
df_merged.to_parquet(out_parquet, index=False)
print(f"\nSaved: {out_parquet}")

# ── 年齢分布図の描画 ──
print("Plotting cluster age pyramids ...")
cluster_means = df_tgt.groupby("cluster_id")[prop_cols].mean()

# 年齢ラベル（日本語）
if jp_font:
    display_labels = [l.replace("_", "〜") + "歳" if l != "95_over" else "95歳以上"
                      for l in AGE_LABELS]
else:
    display_labels = [l.replace("_over", "+") for l in AGE_LABELS]

fig, axes = plt.subplots(2, 3, figsize=(18, 14), sharex=True)
axes = axes.flatten()

colors = gradient_colors

for i in range(N_CLUSTERS):
    ax = axes[i]
    if i in cluster_means.index:
        vals = cluster_means.loc[i].values * 100
        y_pos = range(len(vals))
        ax.barh(y_pos, vals, color=colors[i], edgecolor="white", alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(display_labels, fontsize=9)
        n = summary_rows[i]["n_mesh"]
        ep = summary_rows[i]["elderly_pct"]
        cp = summary_rows[i]["child_pct"]
        ma = summary_rows[i]["mean_age"]
        mp = summary_rows[i]["mean_pop"]
        t1 = summary_rows[i]["top1_age"]
        ax.set_title(
            f"Cluster {i}  (n={n:,}, 最頼{t1}, 平均{mp:.0f}人/メッシュ)\n"
            f"子供{cp:.1f}% / 現役{summary_rows[i]['working_pct']:.1f}% / 高齢{ep:.1f}%",
            fontsize=11, fontweight="bold", pad=8
        )
        ax.set_xlabel("人口割合 (%)" if jp_font else "Population share (%)", fontsize=10)
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        ax.set_xlim(0, 28)


for ax in axes[N_CLUSTERS:]:  # 余ったサブプロットを非表示
    ax.set_visible(False)
plt.tight_layout(rect=[0, 0, 1, 0.97])
fig.suptitle(f"クラスター別 年齢構成（500mメッシュ・人口{MIN_POP}人以上）",
             fontsize=14, fontweight="bold")

out_png = DIR_PATH / "cluster_pyramids.png"
plt.savefig(out_png, dpi=180, bbox_inches="tight")
print(f"Saved: {out_png}")
print("\nDone.")
