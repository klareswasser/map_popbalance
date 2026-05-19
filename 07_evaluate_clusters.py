import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

DIR_PATH = Path(__file__).parent
out_img = DIR_PATH / "cluster_evaluation.png"

print("Loading 500m data for cluster evaluation...")
df = pd.read_parquet(DIR_PATH / "population_500m.parquet")

# 解析対象の準備（人口100人以上）
target_mask = df["pop_total"] >= 50
df_analysis = df[target_mask].copy()
print(f"Target meshes: {len(df_analysis):,}")

age_cols = [c for c in df.columns if c.startswith("pop_") and c != "pop_total"]
X_full = df_analysis[age_cols].div(df_analysis["pop_total"], axis=0).fillna(0).values

# StandardScaler + PCA（10_recluster.pyと同じ前処理）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full)
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA: {X_full.shape[1]} features → {X_pca.shape[1]} components "
      f"(explained var: {pca.explained_variance_ratio_.sum()*100:.1f}%)")

# ランダムに20,000サンプリングして評価を実施
np.random.seed(42)
sample_idx = np.random.choice(X_pca.shape[0], 20000, replace=False)
X_sample = X_pca[sample_idx]

inertias = []
sil_scores = []
K_range = range(2, 11)

print("Evaluating cluster numbers (K) from 2 to 10...")
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_sample)

    inertias.append(kmeans.inertia_)
    score = silhouette_score(X_sample, labels)
    sil_scores.append(score)
    print(f"K={k} | Inertia(SSE): {kmeans.inertia_:.1f} | Silhouette: {score:.4f}")

# 結果のプロット
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Number of clusters (K)', fontsize=12)
ax1.set_ylabel('Inertia (SSE) - Blue', color=color, fontsize=12)
ax1.plot(K_range, inertias, marker='o', color=color, linewidth=2)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Silhouette Score - Red', color=color, fontsize=12)
ax2.plot(K_range, sil_scores, marker='s', color=color, linewidth=2)
ax2.tick_params(axis='y', labelcolor=color)

plt.title("Elbow Method and Silhouette Score for Age Distribution", fontsize=14)
fig.tight_layout()
plt.savefig(out_img, dpi=200)
print(f"Saved evaluation plot to {out_img}")
