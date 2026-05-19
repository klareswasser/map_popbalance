import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import warnings

# Ignore pandas warnings about fragmentation
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

DIR_PATH = Path(r"G:\マイドライブ\02_Projects\分析_exp\00_datasets_visualize\国勢調査_メッシュ年齢分布分析")

def calculate_gini(array):
    """
    配列のジニ係数を計算する。
    値は非負であること。すべて0の場合はNaNを返す。
    """
    array = np.array(array, dtype=float)
    if np.sum(array) == 0:
        return np.nan
    
    # 昇順ソート
    array = np.sort(array)
    n = array.shape[0]
    
    # \sum_{i=1}^n i * y_i
    index = np.arange(1, n + 1)
    
    return ((2 * np.sum(index * array)) / (n * np.sum(array))) - ((n + 1) / n)

def analyze_population_distribution(size_label: str = "500m", min_population: int = 100):
    input_file = DIR_PATH / f"population_{size_label}.parquet"
    if not input_file.exists():
        print(f"File not found: {input_file}")
        return
    
    print(f"Loading {input_file}...")
    df = pd.read_parquet(input_file)
    
    # 年齢区分カラムの取得
    # "pop_0_4", "pop_5_9" ... "pop_90_94", "pop_95_over"
    age_cols = [c for c in df.columns if c.startswith("pop_") and c != "pop_total"]
    
    # 0〜64歳まで（生産年齢・若年）のカラムを中心に抽出
    target_age_cols = age_cols[:13] # idx 12 is pop_60_64
    
    print(f"Target age columns for Gini: {target_age_cols}")
    
    # 分析対象の絞り込み: メッシュ人口が一定(min_population)以上のものを対象（ノイズ排除）
    target_mask = df["pop_total"] >= min_population
    df_analysis = df[target_mask].copy()
    print(f"Analyzing {len(df_analysis)} meshes (Total Pop >= {min_population}) out of {len(df)}.")
    
    # --- 1. 現役世代の偏り（ジニ係数）計算 ---
    print("Calculating Gini coefficients (0-64 years)...")
    # numpy行列として抜き出し、行ごとに適用
    vals_0_64 = df_analysis[target_age_cols].values
    
    gini_scores = []
    for row in tqdm(vals_0_64):
        gini_scores.append(calculate_gini(row))
        
    df_analysis["gini_0_64"] = gini_scores
    
    # --- 2. 人口ピラミッド形態でのクラスタリング（全体年齢構成割合） ---
    # 各年齢階級の総人口に対する割合を計算
    print("Calculating age proportions and clustering...")
    prop_cols = []
    for col in age_cols:
        prop_col = f"prop_{col}"
        # 割り算時のゼロ除算対策はすでに min_population でフィルタ済みだが念のため
        df_analysis[prop_col] = df_analysis[col] / df_analysis["pop_total"]
        prop_cols.append(prop_col)
        
    # K-Meansクラスタリング
    # クラスター数は 6 つに仮設定（子育て層、若年独身、高齢化、バランス型など想定）
    N_CLUSTERS = 6
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    
    # X行列の作成 (全年齢の割合)
    X = df_analysis[prop_cols].fillna(0).values
    df_analysis["cluster_id"] = kmeans.fit_predict(X)
    
    # クラスターの中心傾向を確認
    print("\n--- Cluster Centers (Top 3 age groups per cluster) ---")
    centers = kmeans.cluster_centers_
    for i in range(N_CLUSTERS):
        center = centers[i]
        top3_idx = center.argsort()[::-1][:3]
        top3_labels = [age_cols[idx].replace("pop_", "") for idx in top3_idx]
        top3_vals = [center[idx] for idx in top3_idx]
        label_str = ", ".join([f"{l}({v:.1%})" for l, v in zip(top3_labels, top3_vals)])
        print(f"Cluster {i}: Focus on {label_str}")

    # --- 3. 保存 ---
    # 元の df に分析結果をマージ（分析対象外は NaN）
    output_cols = ["KEY_CODE", "gini_0_64", "cluster_id"] + prop_cols
    df_out = df.merge(df_analysis[output_cols], on="KEY_CODE", how="left")
    
    out_file = DIR_PATH / f"analysis_{size_label}.parquet"
    df_out.to_parquet(out_file, index=False)
    print(f"Saved analysis results to {out_file}\n")
    
    # 上位のジニ係数のサンプルを出力
    top_gini = df_analysis.sort_values("gini_0_64", ascending=False).head(5)
    print("Top 5 meshes by high Gini (0-64):")
    print(top_gini[["KEY_CODE", "pop_total", "gini_0_64", "cluster_id"]])

if __name__ == "__main__":
    analyze_population_distribution("500m")
    analyze_population_distribution("250m")
