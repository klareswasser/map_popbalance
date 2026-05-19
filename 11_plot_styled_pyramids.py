"""
11_plot_styled_pyramids.py
クラスター別人口ピラミッドを kanto_born_here_ratio.png 風のデザインで出力する。
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from matplotlib.gridspec import GridSpec

DIR_PATH = Path(__file__).parent

# ── フォント設定 ──────────────────────────────────────────────
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
    plt.rcParams["axes.unicode_minus"] = False

# ── カラー / ラベル定義 ─────────────────────────────────────────
CLUSTER_COLORS = [
    "#ff9800",  # 0: オレンジ
    "#4caf50",  # 1: 緑
    "#bdbdbd",  # 2: 薄グレー
    "#1565c0",  # 3: 濃青
    "#f44336",  # 4: 赤
    "#795548",  # 5: 茶
]
CLUSTER_NAMES = [
    "学生街エリア",
    "近年人気が高まる子育てエリア",
    "多世代が住まう都市型エリア",
    "高齢化が見え始めた郊外エリア",
    "準・限界住宅地エリア",
    "限界住宅地エリア",
]

# 参照画像由来のデザイン色
BG_COLOR      = "#cfe3f0"   # 外枠の水色
PANEL_COLOR   = "#f7fbfd"   # 各グラフパネルの背景色
TITLE_COLOR   = "#1a1a1a"
AXIS_COLOR    = "#444444"
GRID_COLOR    = "#d9e8f0"

# ── データ読み込み ──────────────────────────────────────────────
print("Loading analysis_500m.parquet ...")
df = pd.read_parquet(DIR_PATH / "analysis_500m.parquet")
df = df.dropna(subset=["cluster_id"])
df["cluster_id"] = df["cluster_id"].astype(int)

age_cols = sorted([c for c in df.columns if c.startswith("pop_") and c != "pop_total"])
prop_cols = [f"prop_{c}" for c in age_cols]

AGE_LABELS = [c.replace("pop_", "") for c in age_cols]
display_labels = [
    l.replace("_", "〜") + "歳" if l != "95_over" else "95歳以上"
    for l in AGE_LABELS
]

AGE_MIDPOINTS_DICT = {
    "0_4": 2, "5_9": 7, "10_14": 12, "15_19": 17, "20_24": 22,
    "25_29": 27, "30_34": 32, "35_39": 37, "40_44": 42, "45_49": 47,
    "50_54": 52, "55_59": 57, "60_64": 62, "65_69": 67, "70_74": 72,
    "75_79": 77, "80_84": 82, "85_89": 87, "90_94": 92, "95_over": 97,
}
midpoints = np.array([AGE_MIDPOINTS_DICT.get(l, 97) for l in AGE_LABELS])

CHILD_COLS   = [c for c in prop_cols if any(a in c for a in ["0_4", "5_9", "10_14"])]
ELDERLY_COLS = [c for c in prop_cols if any(
    a in c for a in ["65_69","70_74","75_79","80_84","85_89","90_94","95_over"])]
WORKING_COLS = [c for c in prop_cols if c not in CHILD_COLS + ELDERLY_COLS]

N_CLUSTERS = 6
cluster_means = df.groupby("cluster_id")[prop_cols].mean()
cluster_stats = []
for cid in range(N_CLUSTERS):
    sub = df[df["cluster_id"] == cid]
    mean_props = sub[prop_cols].mean().fillna(0).values
    mean_age = float(np.dot(mean_props, midpoints))
    top1_label = AGE_LABELS[mean_props.argsort()[::-1][0]]
    top1_str = top1_label.replace("_", "〜") + "歳" if top1_label != "95_over" else "95歳以上"
    cluster_stats.append({
        "n_mesh": len(sub),
        "mean_pop": sub["pop_total"].mean() if "pop_total" in df.columns else 0,
        "mean_age": mean_age,
        "top1_age": top1_str,
        "child_pct": sub[CHILD_COLS].sum(axis=1).mean() * 100,
        "working_pct": sub[WORKING_COLS].sum(axis=1).mean() * 100,
        "elderly_pct": sub[ELDERLY_COLS].sum(axis=1).mean() * 100,
    })

# ── プロット ──────────────────────────────────────────────────
# 全体サイズ: 参照画像比率に合わせ横長
FIG_W, FIG_H = 18, 12
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG_COLOR)

# 上部: タイトル行, 下部: グラフ2×3, 下余白（凡例）
# left/right に余白を設ける
gs_outer = GridSpec(
    3, 1, figure=fig,
    height_ratios=[0.12, 1, 0.05],
    hspace=0.0,
    top=0.96, bottom=0.03, left=0.04, right=0.97
)

# ── タイトル行 ──────────────────────────────────────────────────
ax_title = fig.add_subplot(gs_outer[0])
ax_title.set_facecolor(BG_COLOR)
ax_title.axis("off")

# 大タイトル（左）
ax_title.text(
    0.0, 0.5,
    "年齢構成クラスターの人口ピラミッド",
    transform=ax_title.transAxes,
    fontsize=22, fontweight="bold",
    color=TITLE_COLOR, va="center", ha="left",
    fontfamily=jp_font or "sans-serif",
)
# サブタイトル
ax_title.text(
    0.0, -0.22,
    "500mメッシュ・人口150人以上を対象に K=6 クラスタリング",
    transform=ax_title.transAxes,
    fontsize=10, color="#555555", va="center", ha="left",
    fontfamily=jp_font or "sans-serif",
)
# 右上: 出典
ax_title.text(
    1.0, 1.0,
    "総務省『国勢調査』（2020）",
    transform=ax_title.transAxes,
    fontsize=9, color="#666666", va="top", ha="right",
    fontfamily=jp_font or "sans-serif",
)
ax_title.text(
    1.0, 0.5,
    "500mメッシュ別年齢構成比から KMeans クラスタリング",
    transform=ax_title.transAxes,
    fontsize=9, color="#666666", va="center", ha="right",
    fontfamily=jp_font or "sans-serif",
)

# ── 2×3グリッド ──────────────────────────────────────────────
gs_inner = gs_outer[1].subgridspec(
    2, 3,
    hspace=0.50, wspace=0.38
)

XMAX = 26
y_pos = range(len(display_labels))

for i in range(N_CLUSTERS):
    ax = fig.add_subplot(gs_inner[i // 3, i % 3])
    ax.set_facecolor(PANEL_COLOR)

    # 外枠をパネル背景色に近い薄い色で
    for spine in ax.spines.values():
        spine.set_edgecolor("#b0c8d8")
        spine.set_linewidth(0.8)

    if i in cluster_means.index:
        vals = cluster_means.loc[i].fillna(0).values * 100
        color = CLUSTER_COLORS[i]

        # バー
        bars = ax.barh(
            list(y_pos), vals,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.88,
            height=0.75,
        )

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(display_labels, fontsize=8.5, color=AXIS_COLOR,
                       fontfamily=jp_font or "sans-serif")
    ax.set_xlim(0, XMAX)
    ax.set_xlabel("人口割合 (%)", fontsize=8.5, color=AXIS_COLOR,
                  fontfamily=jp_font or "sans-serif", labelpad=3)
    ax.tick_params(axis="x", labelsize=8, colors=AXIS_COLOR)

    # グリッド
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID_COLOR, linestyle="-", linewidth=0.8)
    ax.yaxis.grid(False)

    st = cluster_stats[i]
    stats_text = (
        f"平均年齢 {st['mean_age']:.1f}歳  ／  "
        f"子供{st['child_pct']:.0f}% ・ 現役{st['working_pct']:.0f}% ・ 高齢{st['elderly_pct']:.0f}%"
    )
    ax.set_title(
        f"CL{i}｜{CLUSTER_NAMES[i]}\n{stats_text}",
        fontsize=9.5, fontweight="bold",
        color=TITLE_COLOR, pad=6,
        fontfamily=jp_font or "sans-serif",
        linespacing=1.6,
    )
    # statsはtitleに内包したので削除

# ── 凡例行 ─────────────────────────────────────────────────────
ax_leg = fig.add_subplot(gs_outer[2])
ax_leg.set_facecolor(BG_COLOR)
ax_leg.axis("off")

patches = [
    mpatches.Patch(facecolor=CLUSTER_COLORS[i], edgecolor="#888888",
                   linewidth=0.6, label=f"CL{i}: {CLUSTER_NAMES[i]}")
    for i in range(N_CLUSTERS)
]
ax_leg.legend(
    handles=patches,
    ncol=3,
    loc="center",
    frameon=True,
    framealpha=0.85,
    edgecolor="#aaaaaa",
    fontsize=8.5,
    prop={"family": jp_font or "sans-serif", "size": 8.5},
    handlelength=1.4,
    handleheight=1.1,
    columnspacing=1.5,
    handletextpad=0.6,
)

out_png = DIR_PATH / "cluster_pyramids_styled.png"
plt.savefig(out_png, dpi=180, bbox_inches="tight", facecolor=BG_COLOR)
print(f"Saved: {out_png}")
