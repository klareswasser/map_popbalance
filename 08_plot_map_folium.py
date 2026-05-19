import pandas as pd
import numpy as np
import folium
import json
from pathlib import Path

DIR_PATH = Path(__file__).parent
# 500mメッシュを使用（250mより約1/4のメッシュ数）
file_path = DIR_PATH / "analysis_500m.parquet"
out_map = Path.home() / "Desktop" / "cluster_map_tokyo.html"

def mesh_to_latlon_vec(codes: pd.Series):
    """ベクトル化されたメッシュコード→緯度経度変換"""
    c = codes.astype(str)
    lat = c.str[0:2].astype(float) / 1.5
    lon = c.str[2:4].astype(float) + 100.0
    lat += c.str[4:5].astype(float) * (5.0 / 60.0)
    lon += c.str[5:6].astype(float) * (7.5 / 60.0)
    lat += c.str[6:7].astype(float) * (30.0 / 3600.0)
    lon += c.str[7:8].astype(float) * (45.0 / 3600.0)
    # 4次メッシュ (500m): 9桁目
    d4 = c.str[8:9].astype(float)
    lat += np.where(d4.isin([3, 4]), 15.0 / 3600.0, 0)
    lon += np.where(d4.isin([2, 4]), 22.5 / 3600.0, 0)
    # 中心座標（500mメッシュの半分）
    lat += 7.5 / 3600.0
    lon += 11.25 / 3600.0
    return lat, lon

print("Loading 500m analysis data...")
df = pd.read_parquet(file_path)
df = df.dropna(subset=["cluster_id"]).copy()

print("Calculating Lat/Lon (vectorized)...")
df["lat"], df["lon"] = mesh_to_latlon_vec(df["KEY_CODE"])

# 首都圏全体
lat_min, lat_max = 35.2, 36.1
lon_min, lon_max = 139.0, 140.4
df_map = df[
    (df["lat"] >= lat_min) & (df["lat"] <= lat_max) &
    (df["lon"] >= lon_min) & (df["lon"] <= lon_max)
].copy()
df_map["cluster_id"] = df_map["cluster_id"].astype(int)

print(f"Plotting {len(df_map)} meshes (500m) on the map in Tokyo metro area...")

center_lat = (lat_min + lat_max) / 2
center_lon = (lon_min + lon_max) / 2
m = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles="CartoDB positron")

# 色の設定
colors_dict = {
    0: "#ff9800",
    1: "#4caf50",
    2: "#bdbdbd",
    3: "#1565c0",
    4: "#f44336",
    5: "#795548",
}

# クラスターごとにGeoJSONをまとめて描画（DOM要素を大幅削減）
for cid, color in colors_dict.items():
    subset = df_map[df_map["cluster_id"] == cid]
    if subset.empty:
        continue
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [row["lon"], row["lat"]]},
            "properties": {"pop": int(row["pop_total"])}
        }
        for _, row in subset.iterrows()
    ]
    geojson = {"type": "FeatureCollection", "features": features}
    folium.GeoJson(
        geojson,
        name=f"Cluster {cid}",
        marker=folium.CircleMarker(
            radius=4,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            weight=0,
        ),
    ).add_to(m)

folium.LayerControl().add_to(m)

legend_html = '''
     <div style="position: fixed; 
     bottom: 50px; left: 50px; width: 220px; height: 180px; 
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color:white; opacity:0.9; padding:10px;">
     <b>クラスター凡例</b><br>
     <i style="color:#ff9800;">■</i> 0: 学生街エリア<br>
     <i style="color:#4caf50;">■</i> 1: 近年人気が高まる子育てエリア<br>
     <i style="color:#bdbdbd;">■</i> 2: 多世代が住まう都市型エリア<br>
     <i style="color:#1565c0;">■</i> 3: 高齢化が見え始めた郊外エリア<br>
     <i style="color:#f44336;">■</i> 4: 準・限界住宅地エリア<br>
     <i style="color:#795548;">■</i> 5: 限界住宅地エリア<br>
     </div>
     '''
m.get_root().html.add_child(folium.Element(legend_html))

m.save(out_map)
print(f"Saved map to {out_map}")
