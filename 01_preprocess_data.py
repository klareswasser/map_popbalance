import os
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

RAW_DIR = Path(r"G:\マイドライブ\02_Projects\分析_exp\00_datasets\data_raw\メッシュ人口_2020")
OUT_DIR = Path(__file__).parent

def process_meshes(size_dir_name: str, prefix: str):
    """
    指定されたサイズのメッシュ人口データを読み込み、
    総人口と5歳階級別人口（総数）を抽出・統合してParquet形式で保存する。
    """
    csv_dir = RAW_DIR / size_dir_name / "csv"
    files = list(csv_dir.glob("*.txt"))
    if not files:
        print(f"No files found in {csv_dir}")
        return

    # 抽出するカラムの定義
    # 001 = 総人口
    # 004, 007, ... 061 = 5歳階級（男女計）
    cols_to_keep = ["KEY_CODE", f"{prefix}001"]
    age_cols = [f"{prefix}{str(i).zfill(3)}" for i in range(4, 62, 3)]
    cols_to_keep.extend(age_cols)

    # リネーム用の辞書
    rename_dict = {f"{prefix}001": "pop_total"}
    
    # 0-4, 5-9 ... 90-94
    for i, col in enumerate(age_cols[:-1]):
        age_start = i * 5
        age_end = age_start + 4
        rename_dict[col] = f"pop_{age_start}_{age_end}"
    
    # 95以上
    rename_dict[age_cols[-1]] = "pop_95_over"

    all_dfs = []
    
    print(f"Processing {len(files)} files for {size_dir_name}...")
    for f in tqdm(files):
        try:
            # 2行目は日本語のカラム説明なので skiprows=[1] で除外
            df = pd.read_csv(f, skiprows=[1], encoding="cp932", usecols=cols_to_keep, dtype={"KEY_CODE": str})
            
            # None や "X", "-" 等の補完処理 ("*", "-" を 0 に置換)
            df = df.replace(["*", "-"], "0")
            
            # 数値型へ変換
            for col in cols_to_keep[1:]:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int32")
                
            df = df.rename(columns=rename_dict)
            all_dfs.append(df)
            
        except Exception as e:
            print(f"Error reading {f.name}: {e}")

    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        # NaNなコードがあれば除外
        merged_df = merged_df.dropna(subset=["KEY_CODE"])
        
        # 保存
        out_file = OUT_DIR / f"population_{size_dir_name}.parquet"
        merged_df.to_parquet(out_file, index=False)
        print(f"Saved to {out_file} (Rows: {len(merged_df)})")

if __name__ == "__main__":
    process_meshes("500m", "T001192")
    process_meshes("250m", "T001196")
