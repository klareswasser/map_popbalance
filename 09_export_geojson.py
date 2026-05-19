"""
09_export_geojson.py
全国の500mメッシュ クラスターデータを GeoJSON に書き出す。
出力: docs/cluster_500m.geojson
その後 tippecanoe で PMTiles に変換する。

使用方法:
  python 09_export_geojson.py
  tippecanoe -o docs/cluster.pmtiles --force -Z4 -z12 \
    --drop-densest-as-needed --extend-zooms-if-still-dropping \
    docs/cluster_500m.geojson
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

DIR_PATH = Path(__file__).parent
OUT_GEOJSON = DIR_PATH / "docs" / "cluster_500m.geojson"
OUT_GEOJSON.parent.mkdir(exist_ok=True)

# 500m メッシュのセル幅（度単位）
LAT_STEP = 15.0 / 3600.0   # ≈ 0.004167°
LON_STEP = 22.5 / 3600.0   # ≈ 0.00625°

def mesh_sw(code: str):
    """メッシュコードから南西端 (lat, lon) を返す（ベクトル化なしの単体版）"""
    lat = int(code[0:2]) / 1.5
    lon = int(code[2:4]) + 100.0
    lat += int(code[4:5]) * (5.0 / 60.0)
    lon += int(code[5:6]) * (7.5 / 60.0)
    lat += int(code[6:7]) * (30.0 / 3600.0)
    lon += int(code[7:8]) * (45.0 / 3600.0)
    if len(code) >= 9:
        d4 = int(code[8:9])
        if d4 in (3, 4): lat += 15.0 / 3600.0
        if d4 in (2, 4): lon += 22.5 / 3600.0
    return lat, lon


def mesh_sw_vec(codes: pd.Series):
    """ベクトル化版: 南西端を計算して (lat_array, lon_array) を返す"""
    c = codes.astype(str)
    lat = c.str[0:2].astype(float) / 1.5
    lon = c.str[2:4].astype(float) + 100.0
    lat += c.str[4:5].astype(float) * (5.0 / 60.0)
    lon += c.str[5:6].astype(float) * (7.5 / 60.0)
    lat += c.str[6:7].astype(float) * (30.0 / 3600.0)
    lon += c.str[7:8].astype(float) * (45.0 / 3600.0)
    d4 = c.str[8:9].astype(float)
    lat += np.where(d4.isin([3, 4]), 15.0 / 3600.0, 0.0)
    lon += np.where(d4.isin([2, 4]), 22.5 / 3600.0, 0.0)
    return lat.values, lon.values


print("Loading analysis_500m.parquet ...")
df = pd.read_parquet(DIR_PATH / "analysis_500m.parquet")
df = df.dropna(subset=["cluster_id"]).copy()
df["cluster_id"] = df["cluster_id"].astype(int)

print(f"Total meshes: {len(df):,}")

print("Computing SW corners (vectorized)...")
sw_lat, sw_lon = mesh_sw_vec(df["KEY_CODE"])
df["sw_lat"] = sw_lat
df["sw_lon"] = sw_lon

cluster_names = {
    0: "学生街エリア",                 # 33.3歳
    1: "近年人気が高まる子育てエリア",   # 38.6歳
    2: "多世代が住まう都市型エリア",     # 42.0歳
    3: "高齢化が見え始めた郊外エリア",   # 46.6歳
    4: "準・限界住宅地エリア",           # 53.4歳
    5: "限界住宅地エリア",               # 67.2歳
}

print("Building GeoJSON features...")
features = []
for row in df.itertuples(index=False):
    s_lat = row.sw_lat
    s_lon = row.sw_lon
    n_lat = s_lat + LAT_STEP
    e_lon = s_lon + LON_STEP
    # GeoJSON polygon (CCW)
    coords = [[
        [s_lon, s_lat],
        [e_lon, s_lat],
        [e_lon, n_lat],
        [s_lon, n_lat],
        [s_lon, s_lat],
    ]]
    cid = int(row.cluster_id)
    feat = {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": coords},
        "properties": {
            "cluster_id": cid,
            "cluster_name": cluster_names.get(cid, ""),
            "pop_total": int(row.pop_total),
        },
    }
    # gini があれば追加
    if hasattr(row, "gini_0_64") and not (row.gini_0_64 != row.gini_0_64):
        feat["properties"]["gini"] = round(float(row.gini_0_64), 4)
    features.append(feat)

geojson = {"type": "FeatureCollection", "features": features}

print(f"Writing {OUT_GEOJSON} ...")
with open(OUT_GEOJSON, "w", encoding="utf-8") as f:
    json.dump(geojson, f, ensure_ascii=False, separators=(",", ":"))

import os
size_mb = os.path.getsize(OUT_GEOJSON) / 1e6
print(f"Done. {len(features):,} features, {size_mb:.1f} MB")
print()
print("Next step:")
print(f"  tippecanoe -o {DIR_PATH}/docs/cluster.pmtiles --force \\")
print(f"    -Z4 -z12 --drop-densest-as-needed \\")
print(f"    --extend-zooms-if-still-dropping \\")
print(f"    {OUT_GEOJSON}")
