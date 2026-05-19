"""
11_plot_styled_pyramids.py
kanto_born_here_ratio.png の外装デザインを完全に踏襲したクラスター別人口ピラミッド図。
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle

DIR_PATH = Path(__file__).parent

def find_jp_font():
    cands = ["Hiragino Sans", "Hiragino Kaku Gothic ProN",
             "Hiragino Maru Gothic Pro", "Arial Unicode MS"]
    available = {f.name for f in fm.fontManager.ttflist}
    for c in cands:
        if c in available:
            return c
    return None

JP = find_jp_font()
if JP:
    plt.rcParams["font.family"] = JP
plt.rcParams["axes.unicode_minus"] = False

BG_OUTER = "#cfe3ef"
BG_PANEL = "#fafaf2"
BORDER_DARK = "#3a3a3a"
TEXT_MAIN = "#1a1a1a"
TEXT_SUB = "#555555"
GRID = "#d9d4c4"

CLUSTER_COLORS = ["#ff9800","#4caf50","#bdbdbd","#1565c0","#f44336","#795548"]
CLUSTER_NAMES = [
    "学生街エリア",
    "近年人気が高まる子育てエリア",
    "多世代が住まう都市型エリア",
    "高齢化が見え始めた郊外エリア",
    "準・限界住宅地エリア",
    "限界住宅地エリア",
]

df = pd.read_parquet(DIR_PATH / "analysis_500m.parquet")
df = df.dropna(subset=["cluster_id"])
df["cluster_id"] = df["cluster_id"].astype(int)

MIDPT = {"0_4":2,"5_9":7,"10_14":12,"15_19":17,"20_24":22,"25_29":27,"30_34":32,
         "35_39":37,"40_44":42,"45_49":47,"50_54":52,"55_59":57,"60_64":62,
         "65_69":67,"70_74":72,"75_79":77,"80_84":82,"85_89":87,"90_94":92,"95_over":97}

# 年齢の数値順（若→老）に並べる（文字列ソートだと "10_14" < "5_9" になるのを防ぐ）
raw_age_cols = [c for c in df.columns if c.startswith("pop_") and c != "pop_total"]
age_cols = sorted(raw_age_cols, key=lambda c: MIDPT[c.replace("pop_", "")])
prop_cols = [f"prop_{c}" for c in age_cols]
AGE_LABELS = [c.replace("pop_", "") for c in age_cols]
disp_labels = [l.replace("_", "〜") + "歳" if l != "95_over" else "95歳以上" for l in AGE_LABELS]
mid = np.array([MIDPT[l] for l in AGE_LABELS])

CHILD = [c for c in prop_cols if any(a in c for a in ["0_4","5_9","10_14"])]
ELD = [c for c in prop_cols if any(a in c for a in ["65_69","70_74","75_79","80_84","85_89","90_94","95_over"])]
WORK = [c for c in prop_cols if c not in CHILD + ELD]

N = 6
cluster_means = df.groupby("cluster_id")[prop_cols].mean()
stats = []
for cid in range(N):
    sub = df[df["cluster_id"] == cid]
    p = sub[prop_cols].mean().fillna(0).values
    stats.append({
        "mean_age": float(np.dot(p, mid)),
        "child":   sub[CHILD].sum(axis=1).mean() * 100,
        "work":    sub[WORK].sum(axis=1).mean() * 100,
        "eld":     sub[ELD].sum(axis=1).mean() * 100,
    })

# ── 描画 ──
FIG_W, FIG_H = 16, 12
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG_OUTER)

ax_bg = fig.add_axes([0, 0, 1, 1])
ax_bg.set_xlim(0, 1); ax_bg.set_ylim(0, 1)
ax_bg.axis("off")
ax_bg.set_facecolor(BG_OUTER)

# タイトル（左上、大きく）
ax_bg.text(0.04, 0.94,
           "年齢構成クラスターの人口ピラミッド",
           fontsize=30, color=TEXT_MAIN, va="top", ha="left",
           fontfamily=JP or "sans-serif")

# 出典（右上、2行）
ax_bg.text(0.96, 0.955,
           "総務省『国勢調査』（2020）",
           fontsize=11, color=TEXT_SUB, va="top", ha="right",
           fontfamily=JP or "sans-serif")
ax_bg.text(0.96, 0.928,
           "500mメッシュ別年齢構成比 / K-Means クラスタリング (K=6)",
           fontsize=11, color=TEXT_SUB, va="top", ha="right",
           fontfamily=JP or "sans-serif")

# 中央の大パネル
PANEL_X, PANEL_Y = 0.04, 0.06
PANEL_W, PANEL_H = 0.92, 0.82
ax_bg.add_patch(Rectangle(
    (PANEL_X, PANEL_Y), PANEL_W, PANEL_H,
    facecolor=BG_PANEL, edgecolor=BORDER_DARK, linewidth=1.6,
    transform=ax_bg.transAxes, zorder=1,
))

PAD_L, PAD_R, PAD_T, PAD_B = 0.025, 0.025, 0.035, 0.045
inner_x0 = PANEL_X + PAD_L
inner_y0 = PANEL_Y + PAD_B
inner_w = PANEL_W - PAD_L - PAD_R
inner_h = PANEL_H - PAD_T - PAD_B

cols, rows = 3, 2
gap_x, gap_y = 0.022, 0.07
cell_w = (inner_w - gap_x * (cols - 1)) / cols
cell_h = (inner_h - gap_y * (rows - 1)) / rows

y_pos = np.arange(len(disp_labels))
XMAX = 26

for i in range(N):
    r, c = i // cols, i % cols
    x0 = inner_x0 + c * (cell_w + gap_x)
    y0 = inner_y0 + (rows - 1 - r) * (cell_h + gap_y)

    title_h = 0.055
    ax = fig.add_axes([x0, y0, cell_w, cell_h - title_h])
    ax.set_facecolor(BG_PANEL)

    vals = cluster_means.loc[i].fillna(0).values * 100 if i in cluster_means.index else np.zeros(len(y_pos))
    color = CLUSTER_COLORS[i]
    ax.barh(y_pos, vals, color=color, edgecolor="white",
            linewidth=0.4, alpha=0.9, height=0.78)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(disp_labels, fontsize=7.5, color=TEXT_MAIN,
                       fontfamily=JP or "sans-serif")
    ax.set_xlim(0, XMAX)
    ax.set_ylim(-0.7, len(y_pos) - 0.3)
    ax.tick_params(axis="x", labelsize=7, colors=TEXT_SUB, length=0)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xlabel("人口割合 (%)", fontsize=7.5, color=TEXT_SUB,
                  fontfamily=JP or "sans-serif", labelpad=2)

    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#bcb6a5")
        ax.spines[s].set_linewidth(0.7)

    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, linestyle="-", linewidth=0.6)

    # セル上のタイトル
    title_top = y0 + cell_h - 0.008
    fig.text(x0 + 0.004, title_top,
             f"CL{i}　{CLUSTER_NAMES[i]}",
             fontsize=12, fontweight="bold", color=TEXT_MAIN,
             va="top", ha="left",
             fontfamily=JP or "sans-serif")
    st = stats[i]
    fig.text(x0 + 0.004, title_top - 0.025,
             f"平均年齢 {st['mean_age']:.1f}歳   子供 {st['child']:.0f}%  ・  現役 {st['work']:.0f}%  ・  高齢 {st['eld']:.0f}%",
             fontsize=8.5, color=TEXT_SUB,
             va="top", ha="left",
             fontfamily=JP or "sans-serif")

# ── 凡例ボックスは不要（各パネルにクラスター名表示済み） ──

out = DIR_PATH / "cluster_pyramids_styled.png"
plt.savefig(out, dpi=180, facecolor=BG_OUTER)
print(f"Saved: {out}")
