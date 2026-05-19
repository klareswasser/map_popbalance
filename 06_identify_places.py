import pandas as pd
import numpy as np
import time
import requests
from pathlib import Path

DIR_PATH = Path(r"G:\マイドライブ\02_Projects\分析_exp\00_datasets_visualize\国勢調査_メッシュ年齢分布分析")
file_path_250 = DIR_PATH / "analysis_250m.parquet"

def mesh_to_latlon(mesh_code_str):
    """
    10桁の250mメッシュコードを緯度経度に変換する（南西端の座標）
    標準地域メッシュ体系に準拠
    """
    code = str(mesh_code_str)
    if len(code) < 8:
        return np.nan, np.nan
        
    try:
        # 1次メッシュ（約80km四方）
        lat_1 = int(code[0:2]) / 1.5
        lon_1 = int(code[2:4]) + 100.0
        
        # 2次メッシュ（約10km四方）
        lat_2 = int(code[4:5]) * (5.0 / 60.0)
        lon_2 = int(code[5:6]) * (7.5 / 60.0)
        
        # 3次メッシュ（約1km四方）
        lat_3 = int(code[6:7]) * (30.0 / 3600.0)
        lon_3 = int(code[7:8]) * (45.0 / 3600.0)
        
        lat = lat_1 + lat_2 + lat_3
        lon = lon_1 + lon_2 + lon_3
        
        # 4次メッシュ(500m) - 9桁目
        if len(code) >= 9:
            d4 = int(code[8:9])
            if d4 in [3, 4]:
                lat += 15.0 / 3600.0
            if d4 in [2, 4]:
                lon += 22.5 / 3600.0
                
        # 5次メッシュ(250m) - 10桁目
        if len(code) >= 10:
            d5 = int(code[9:10])
            if d5 in [3, 4]:
                lat += 7.5 / 3600.0
            if d5 in [2, 4]:
                lon += 11.25 / 3600.0
                
        # 中心座標を返すため、250mサイズの半分を足す (緯度約3.75秒、経度約5.625秒)
        lat += 3.75 / 3600.0
        lon += 5.625 / 3600.0
        
        return lat, lon
    except Exception:
        return np.nan, np.nan

def get_address(lat, lon):
    """
    OpenStreetMap Nominatim を用いて逆ジオコーディング
    """
    url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json&accept-language=ja"
    headers = {'User-Agent': 'DataAnalysisBot/1.0'}
    try:
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            data = res.json()
            address = data.get('display_name', '')
            # 細かすぎる情報がある場合は適当に省略
            parts = address.split(', ')
            return ", ".join(parts[:3]) if len(parts) > 3 else address
    except:
        pass
    return "Unknown Address"

print("Loading data...")
df = pd.read_parquet(file_path_250)

# クラスタ名定義
cluster_names = {
    0: "バランス型高年齢地域",
    1: "ファミリー層地域",
    2: "超高齢施設地域",
    3: "学生・若年単身地域",
    4: "進行性高齢化地域",
    5: "若手社会人エリア"
}

# 抽出ターゲット（各クラスターの典型例 = 人口スコア＆属性特化度の高いメッシュ）
results = []
for cid in range(6):
    print(f"\n--- 【Cluster {cid}】 {cluster_names[cid]} の具体例 ---")
    subset = df[df["cluster_id"] == cid]
    
    # 典型例を選ぶためのロジック:
    # クラスタを象徴する年齢層の割合が高く、かつ人がしっかり住んでいる(人口>500)ところを優先
    if cid == 0:
        c = subset[(subset["pop_total"] > 100)]
        target = c.sort_values("pop_total", ascending=False).head(3)
    elif cid == 1: # ファミリーは 10_14 の割合でソート
        c = subset[subset["pop_total"] > 500]
        target = c.sort_values("prop_pop_40_44", ascending=False).head(3)
    elif cid == 2: # 超高齢は 85_89 でソート
        c = subset[subset["pop_total"] > 150]
        target = c.sort_values("prop_pop_85_89", ascending=False).head(3)
    elif cid == 3: # 学生は 15_19 でソート
        c = subset[subset["pop_total"] > 300]
        target = c.sort_values("prop_pop_15_19", ascending=False).head(3)
    elif cid == 4: # 進行高齢は 70_74
        c = subset[subset["pop_total"] > 200]
        target = c.sort_values("prop_pop_70_74", ascending=False).head(3)
    elif cid == 5: # 若手社会人は 30_34
        c = subset[subset["pop_total"] > 500]
        target = c.sort_values("prop_pop_30_34", ascending=False).head(3)
        
    for _, row in target.iterrows():
        lat, lon = mesh_to_latlon(row['KEY_CODE'])
        address = get_address(lat, lon)
        time.sleep(1.1) # limit rate
        
        map_url = f"https://www.google.co.jp/maps/search/?api=1&query={lat},{lon}"
        print(f"メッシュ: {row['KEY_CODE']} | 人口: {row['pop_total']}人")
        print(f"📍 住所: {address}")
        print(f"🔗 Map: {map_url}")
