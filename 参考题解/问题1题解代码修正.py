import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from collections import deque

# =========================
# 全局绘图设置
# =========================

plt.rcParams["font.sans-serif"] = ["SimHei"]      # 中文字体
plt.rcParams["axes.unicode_minus"] = False        # 正常显示负号

plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 11


# =========================
# 1. 读取数据
# =========================

df = pd.read_csv(
    r"D:\数据分析\附件1：molecular_interaction_manifest.csv",
    skiprows=[1],
    encoding="gb18030"
)

num_cols = [
    "Manifold_X", "Manifold_Y", "Bioactivity_Score",
    "LogP", "TPSA", "MolWt", "Dipole_Proxy",
    "Max_Partial_Charge", "Balaban_J", "Bertz_CT"
]

for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(
    subset=["Manifold_X", "Manifold_Y", "Bioactivity_Score"]
).reset_index(drop=True)

X = df[["Manifold_X", "Manifold_Y"]].values
A = df["Bioactivity_Score"].values
n = len(df)


# =========================
# 2. 参数自适应设置
# =========================
# eps 用第 8 近邻距离的中位数
# MinActivity 用邻域活性总和的 75% 分位数
# =========================

k = 8

nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
distances, indices = nbrs.kneighbors(X)

eps = np.median(distances[:, k])

print("eps =", eps)


# =========================
# 3. 构造邻域
# =========================

neighbors = []

for i in range(n):
    d = np.sqrt(np.sum((X - X[i]) ** 2, axis=1))
    neigh = np.where(d <= eps)[0]
    neighbors.append(neigh)

neighbor_activity_sum = np.array([
    A[neighbors[i]].sum() for i in range(n)
])

MinActivity = np.quantile(neighbor_activity_sum, 0.75)

print("MinActivity =", MinActivity)


# =========================
# 4. 活性阈值滤波
# =========================
# 使用活性分数的中位数作为初筛阈值
# =========================

activity_filter = np.quantile(A, 0.50)
high_activity_candidate = A >= activity_filter

print("活性初筛阈值 =", activity_filter)
print("初筛后分子数 =", high_activity_candidate.sum())


# =========================
# 5. E-DBSCAN 聚类
# =========================

labels = np.full(n, -1)       # -1 表示普通点或噪声点
visited = np.zeros(n, dtype=bool)

cluster_id = 0

for i in range(n):

    if visited[i]:
        continue

    visited[i] = True

    # 未通过活性初筛，不作为热点起点
    if not high_activity_candidate[i]:
        continue

    # 判断是否为核心点
    if neighbor_activity_sum[i] < MinActivity:
        continue

    # 新建一个热点区域
    labels[i] = cluster_id
    queue = deque(neighbors[i])

    while queue:
        j = queue.popleft()

        if not visited[j]:
            visited[j] = True

            # 只有高活性且满足密度条件的点才继续扩展
            if high_activity_candidate[j] and neighbor_activity_sum[j] >= MinActivity:
                queue.extend(neighbors[j])

        # 通过初筛的点归入当前热点
        if labels[j] == -1 and high_activity_candidate[j]:
            labels[j] = cluster_id

    cluster_id += 1


df["Hotspot_Label"] = labels
df["Region"] = np.where(df["Hotspot_Label"] >= 0, "Hotspot", "Ordinary")

print(df["Region"].value_counts())
print("热点区域数量 =", cluster_id)


# =========================
# 6. 区域统计比较
# =========================

compare_cols = [
    "Bioactivity_Score", "LogP", "TPSA", "MolWt",
    "Dipole_Proxy", "Max_Partial_Charge",
    "Balaban_J", "Bertz_CT"
]

summary = df.groupby("Region")[compare_cols].agg(
    ["count", "mean", "std", "median"]
)

print("\n热点区域与普通区域统计比较：")
print(summary)

summary.to_csv("Q1_region_summary.csv", encoding="utf-8-sig")


# =========================
# 7. 各热点区域统计
# =========================

hot_detail = df[df["Region"] == "Hotspot"].groupby("Hotspot_Label")[compare_cols].agg(
    ["count", "mean", "median"]
)

print("\n各热点区域统计：")
print(hot_detail)

hot_detail.to_csv("Q1_hotspot_detail.csv", encoding="utf-8-sig")


# =========================
# 8. 选代表性分子
# =========================

representatives = []

# 每个热点区域选：活性最高分子 + 区域中心分子
for label in sorted(df[df["Hotspot_Label"] >= 0]["Hotspot_Label"].unique()):

    sub = df[df["Hotspot_Label"] == label].copy()

    # 活性最高分子
    top = sub.sort_values("Bioactivity_Score", ascending=False).head(1).copy()
    top["Representative_Type"] = "热点内最高活性分子"

    # 离热点区域中心最近的分子
    cx = sub["Manifold_X"].mean()
    cy = sub["Manifold_Y"].mean()

    sub["dist_center"] = np.sqrt(
        (sub["Manifold_X"] - cx) ** 2 +
        (sub["Manifold_Y"] - cy) ** 2
    )

    center = sub.sort_values("dist_center").head(1).copy()
    center["Representative_Type"] = "热点中心代表分子"

    representatives.append(top)
    representatives.append(center)


# 普通区域选几个代表分子
ordinary = df[df["Region"] == "Ordinary"].copy()

if len(ordinary) > 0:
    cx = ordinary["Manifold_X"].mean()
    cy = ordinary["Manifold_Y"].mean()

    ordinary["dist_center"] = np.sqrt(
        (ordinary["Manifold_X"] - cx) ** 2 +
        (ordinary["Manifold_Y"] - cy) ** 2
    )

    ordinary_rep = ordinary.sort_values("dist_center").head(3).copy()
    ordinary_rep["Representative_Type"] = "普通区域代表分子"

    representatives.append(ordinary_rep)


rep = pd.concat(representatives)

rep_cols = [
    "Region", "Hotspot_Label", "Representative_Type",
    "ID", "SMILES", "Manifold_X", "Manifold_Y",
    "Bioactivity_Score", "LogP", "TPSA", "MolWt", "Bertz_CT"
]

rep = rep[rep_cols]

print("\n代表性分子：")
print(rep)

rep.to_csv("Q1_representative_molecules.csv", index=False, encoding="utf-8-sig")


# =========================
# 9. 画图
# =========================
# 本次调整：
# 1. 图1和图2的宽度从 8 增加到 10，高度保持 6
# 2. x 轴方向更舒展
# 3. 点更小、更清晰
# 4. 保存分辨率提高到 dpi=400
# =========================


# -------------------------
# 图1：二维流形空间中的分子活性分布
# -------------------------

plt.figure(figsize=(10, 6))

scatter = plt.scatter(
    df["Manifold_X"],
    df["Manifold_Y"],
    c=df["Bioactivity_Score"],
    cmap="viridis",
    s=16,                 # 点更小，减少重叠
    alpha=0.70,           # 适度透明
    edgecolor="none"      # 去掉黑边
)

cbar = plt.colorbar(scatter)
cbar.set_label("Bioactivity Score")

plt.xlabel("Manifold_X")
plt.ylabel("Manifold_Y")
plt.title("二维流形空间中的分子活性分布")

# 给 x 轴两侧留一点空白，使图更舒展
x_min = df["Manifold_X"].min()
x_max = df["Manifold_X"].max()
x_pad = (x_max - x_min) * 0.08
plt.xlim(x_min - x_pad, x_max + x_pad)

plt.tight_layout()
plt.savefig("Q1_activity_distribution.png", dpi=400)
plt.show()


# -------------------------
# 图2：热点区域识别结果
# -------------------------

plt.figure(figsize=(10, 6))

ordinary_sub = df[df["Region"] == "Ordinary"]
hotspot_sub = df[df["Region"] == "Hotspot"]

# 普通区域点：更小、更淡
plt.scatter(
    ordinary_sub["Manifold_X"],
    ordinary_sub["Manifold_Y"],
    label="Ordinary",
    s=14,
    c="lightgray",
    alpha=0.45,
    edgecolor="none"
)

# 热点区域点：稍大、更醒目
plt.scatter(
    hotspot_sub["Manifold_X"],
    hotspot_sub["Manifold_Y"],
    label="Hotspot",
    s=24,
    c="red",
    alpha=0.80,
    edgecolor="none"
)

plt.xlabel("Manifold_X")
plt.ylabel("Manifold_Y")
plt.title("基于活性密度 E-DBSCAN 的热点区域识别")

# 给 x 轴两侧留一点空白，使图更舒展
x_min = df["Manifold_X"].min()
x_max = df["Manifold_X"].max()
x_pad = (x_max - x_min) * 0.08
plt.xlim(x_min - x_pad, x_max + x_pad)

plt.legend()
plt.tight_layout()
plt.savefig("Q1_hotspot_partition.png", dpi=400)
plt.show()


# -------------------------
# 图3：热点区域与普通区域活性箱线图
# -------------------------

plt.figure(figsize=(6, 5))

df.boxplot(
    column="Bioactivity_Score",
    by="Region",
    grid=True
)

plt.title("热点区域与普通区域活性得分比较")
plt.suptitle("")
plt.xlabel("区域")
plt.ylabel("Bioactivity Score")

plt.tight_layout()
plt.savefig("Q1_activity_boxplot.png", dpi=400)
plt.show()


# =========================
# 10. 保存结果
# =========================

df.to_csv("Q1_molecules_with_region.csv", index=False, encoding="utf-8-sig")

print("\n问题1完成，结果已保存。")