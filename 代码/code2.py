import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# 0. 基本设置
# =========================

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

script_dir = Path(__file__).resolve().parent
data_dir = script_dir / "数据"
if not data_dir.exists():
    data_dir = script_dir.parent / "数据"

out_dir = script_dir / "数据处理"
out_dir.mkdir(exist_ok=True)


# =========================
# 1. 读取附件1：分子活性数据
# =========================

mol_file = data_dir / "附件1：molecular_interaction_manifest.csv"

df = pd.read_csv(
    mol_file,
    skiprows=[1],
    encoding="gb18030"
)

# 如果没有 ID 列，自动用行号作为 ID
if "ID" not in df.columns:
    df["ID"] = df.index

df["Bioactivity_Score"] = pd.to_numeric(
    df["Bioactivity_Score"],
    errors="coerce"
)

df = df.dropna(subset=["Bioactivity_Score"]).reset_index(drop=True)

print("附件1分子数量：", len(df))


# =========================
# 2. 读取附件2：KNN 图边数据
# =========================

edge_file = data_dir / "附件2：knn_graph_edges.csv"

edges = pd.read_csv(
    edge_file,
    skiprows=[1],
    encoding="gb18030"
)

# 转为数值
edges["Source"] = pd.to_numeric(edges["Source"], errors="coerce")
edges["Target"] = pd.to_numeric(edges["Target"], errors="coerce")
edges["Tanimoto_Similarity"] = pd.to_numeric(
    edges["Tanimoto_Similarity"],
    errors="coerce"
)

edges = edges.dropna().reset_index(drop=True)

edges["Source"] = edges["Source"].astype(int)
edges["Target"] = edges["Target"].astype(int)

print("附件2边数量：", len(edges))


# =========================
# 3. 合并活性数据
# =========================
# Source 和 Target 是附件1中的行号索引
# =========================

activity = df["Bioactivity_Score"].values
ids = df["ID"].values

records = []

act_max = activity.max()
act_min = activity.min()
act_range = act_max - act_min + 1e-12

for _, row in edges.iterrows():

    i = int(row["Source"])
    j = int(row["Target"])

    # 防止索引越界
    if i >= len(df) or j >= len(df):
        continue

    sim = row["Tanimoto_Similarity"]

    ai = activity[i]
    aj = activity[j]

    act_diff = abs(ai - aj)

    # 文献公式：活性相似度
    act_sim = 1 - act_diff / act_range

    # 文献公式：SALI
    sali = act_diff / (1 - sim + 1e-12)

    records.append({
        "Source": i,
        "Target": j,
        "Mol1_ID": ids[i],
        "Mol2_ID": ids[j],
        "Mol1_Activity": ai,
        "Mol2_Activity": aj,
        "Tanimoto_Similarity": sim,
        "Activity_Difference": act_diff,
        "Activity_Similarity": act_sim,
        "SALI": sali
    })

result = pd.DataFrame(records)

print("有效分子对数量：", len(result))


# =========================
# 4. 识别活性悬崖
# =========================

# 结构高相似阈值：Tanimoto 前25%
sim_cut = result["Tanimoto_Similarity"].quantile(0.75)

# 活性相似度低阈值：活性相似度后25%
act_sim_cut = result["Activity_Similarity"].quantile(0.25)

# SALI 高阈值：前5%
sali_cut = result["SALI"].quantile(0.95)

result["Type"] = "Other"

# 结构相似但活性差异大：Activity Cliff
result.loc[
    (result["Tanimoto_Similarity"] >= sim_cut) &
    (result["Activity_Similarity"] <= act_sim_cut),
    "Type"
] = "Activity Cliff"

# SALI 最高的前5%：High SALI Cliff
result.loc[
    result["SALI"] >= sali_cut,
    "Type"
] = "High SALI Cliff"

# 结构相似且活性也相似：Smooth SAR
result.loc[
    (result["Tanimoto_Similarity"] >= sim_cut) &
    (result["Activity_Similarity"] >= result["Activity_Similarity"].quantile(0.75)),
    "Type"
] = "Smooth SAR"

print("\n阈值：")
print("结构相似度阈值 =", sim_cut)
print("活性相似度低阈值 =", act_sim_cut)
print("SALI前5%阈值 =", sali_cut)

print("\n分类统计：")
print(result["Type"].value_counts())


# =========================
# 5. 保存结果
# =========================

result = result.sort_values("SALI", ascending=False).reset_index(drop=True)

result.to_csv(
    out_dir / "问题2_全部分子对_SALI活性悬崖指标.csv",
    index=False,
    encoding="utf-8-sig"
)

result.head(100).to_csv(
    out_dir / "问题2_前100个高SALI活性悬崖分子对.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n前10个活性悬崖分子对：")
print(result.head(10))


# =========================
# 6. 统计每个分子参与活性悬崖次数
# =========================

cliffs = result[result["Type"].isin(["Activity Cliff", "High SALI Cliff"])]

mol_count = pd.concat([
    cliffs["Source"],
    cliffs["Target"]
]).value_counts().reset_index()

mol_count.columns = ["Index", "Cliff_Count"]

source_sensitivity = result[["Source", "SALI"]].rename(columns={"Source": "Index"})
target_sensitivity = result[["Target", "SALI"]].rename(columns={"Target": "Index"})
local_sensitivity = pd.concat([source_sensitivity, target_sensitivity])
local_sensitivity = local_sensitivity.groupby("Index")["SALI"].agg(
    Local_Sensitivity="mean",
    Max_Local_SALI="max",
    Neighbor_Count="count"
).reset_index()

mol_summary = df.copy()
mol_summary["Index"] = mol_summary.index

mol_summary = mol_summary.merge(
    mol_count,
    on="Index",
    how="left"
)

mol_summary = mol_summary.merge(
    local_sensitivity,
    on="Index",
    how="left"
)

region_file = out_dir / "问题1_分子热点区域标记结果.csv"
if region_file.exists():
    region_df = pd.read_csv(region_file, encoding="utf-8-sig")
    region_cols = ["ID", "Region", "Hotspot_Label"]
    mol_summary = mol_summary.merge(
        region_df[region_cols],
        on="ID",
        how="left"
    )
else:
    mol_summary["Region"] = "Unknown"
    mol_summary["Hotspot_Label"] = np.nan

mol_summary["Cliff_Count"] = mol_summary["Cliff_Count"].fillna(0).astype(int)
mol_summary["Local_Sensitivity"] = mol_summary["Local_Sensitivity"].fillna(0)
mol_summary["Max_Local_SALI"] = mol_summary["Max_Local_SALI"].fillna(0)
mol_summary["Neighbor_Count"] = mol_summary["Neighbor_Count"].fillna(0).astype(int)

region_summary = mol_summary.groupby("Region").agg(
    Molecule_Count=("Index", "count"),
    Mean_Activity=("Bioactivity_Score", "mean"),
    Mean_Local_Sensitivity=("Local_Sensitivity", "mean"),
    Median_Local_Sensitivity=("Local_Sensitivity", "median"),
    Mean_Cliff_Count=("Cliff_Count", "mean")
).reset_index()

region_summary.to_csv(
    out_dir / "问题2_不同区域局部敏感度统计.csv",
    index=False,
    encoding="utf-8-sig"
)

mol_summary = mol_summary.sort_values(
    "Local_Sensitivity",
    ascending=False
)

mol_summary.to_csv(
    out_dir / "问题2_分子局部敏感度统计.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n局部敏感度最高的分子：")
print(mol_summary[["Index", "ID", "Bioactivity_Score", "Local_Sensitivity", "Cliff_Count", "Region"]].head(10))

print("\n不同区域局部敏感度统计：")
print(region_summary)


# =========================
# 7. 绘制 SAS Map
# =========================

plt.figure(figsize=(9, 6))

colors = {
    "High SALI Cliff": "red",
    "Activity Cliff": "orange",
    "Smooth SAR": "green",
    "Other": "lightgray"
}

for t, sub in result.groupby("Type"):
    plt.scatter(
        sub["Tanimoto_Similarity"],
        sub["Activity_Similarity"],
        s=18,
        alpha=0.7,
        c=colors.get(t, "gray"),
        label=t,
        edgecolor="none"
    )

plt.axvline(sim_cut, color="black", linestyle="--", linewidth=1)
plt.axhline(act_sim_cut, color="black", linestyle="--", linewidth=1)

plt.xlabel("Tanimoto Similarity")
plt.ylabel("Activity Similarity")
plt.title("SAS Map：结构-活性相似性图")

plt.legend()
plt.tight_layout()

plt.savefig(
    out_dir / "问题2_SAS结构活性相似性图.png",
    dpi=400
)

plt.show()


# =========================
# 8. 绘制 SALI 分布图
# =========================

plt.figure(figsize=(8, 5))

plt.hist(
    np.log1p(result["SALI"]),
    bins=40,
    color="steelblue",
    alpha=0.75,
    edgecolor="white"
)

plt.axvline(
    np.log1p(sali_cut),
    color="red",
    linestyle="--",
    linewidth=2,
    label="Top 5% SALI"
)

plt.xlabel("log(1 + SALI)")
plt.ylabel("Frequency")
plt.title("SALI 活性地貌指数分布")

plt.legend()
plt.tight_layout()

plt.savefig(
    out_dir / "问题2_SALI活性悬崖指数分布图.png",
    dpi=400
)

plt.show()


# =========================
# 9. 绘制活性-局部敏感度散点图
# =========================

corr = mol_summary["Bioactivity_Score"].corr(mol_summary["Local_Sensitivity"])

plt.figure(figsize=(8, 6))

region_colors = {
    "Hotspot": "crimson",
    "Ordinary": "steelblue",
    "Unknown": "gray"
}

for region, sub in mol_summary.groupby("Region"):
    plt.scatter(
        sub["Bioactivity_Score"],
        sub["Local_Sensitivity"],
        s=35,
        alpha=0.75,
        c=region_colors.get(region, "gray"),
        label=region,
        edgecolor="white",
        linewidth=0.3
    )

plt.xlabel("Bioactivity Score")
plt.ylabel("Local Sensitivity")
plt.title(f"活性得分与局部敏感度关系图（r = {corr:.3f}）")
plt.legend(title="Region")
plt.tight_layout()

plt.savefig(
    out_dir / "问题2_活性得分与局部敏感度关系图.png",
    dpi=400
)

plt.show()


# =========================
# 10. 绘制不同区域平均局部敏感度柱状图
# =========================

plot_regions = [r for r in ["Ordinary", "Hotspot"] if r in mol_summary["Region"].dropna().unique()]
if len(plot_regions) >= 2:
    plot_summary = region_summary.set_index("Region").loc[plot_regions]
    means = plot_summary["Mean_Local_Sensitivity"].values
    medians = plot_summary["Median_Local_Sensitivity"].values

    plt.figure(figsize=(7, 5))
    x = np.arange(len(plot_regions))

    bars = plt.bar(
        x,
        means,
        color=[region_colors.get(region, "gray") for region in plot_regions],
        alpha=0.75,
        width=0.55,
        label="平均局部敏感度"
    )

    plt.scatter(
        x,
        medians,
        color="black",
        s=60,
        marker="D",
        label="中位局部敏感度",
        zorder=3
    )

    for i, region in enumerate(plot_regions):
        sub = mol_summary.loc[mol_summary["Region"] == region, "Local_Sensitivity"].reset_index(drop=True)
        jitter = np.linspace(-0.18, 0.18, len(sub)) if len(sub) > 1 else np.array([0])
        plt.scatter(
            np.full(len(sub), x[i]) + jitter,
            sub,
            color="#4f83b3",
            alpha=0.55,
            s=18,
            edgecolors="none",
            zorder=2
        )
        plt.text(
            x[i],
            means[i] + 0.08,
            f"{means[i]:.2f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.xticks(x, plot_regions)
    plt.ylabel("Local Sensitivity")
    plt.title("不同空间分区的平均局部敏感度比较")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        out_dir / "问题2_不同区域平均局部敏感度柱状图.png",
        dpi=400
    )

    plt.show()
else:
    print("第一问分区结果不足，跳过不同区域平均局部敏感度柱状图。")


print("\n第二题完成，结果保存在数据处理文件夹：", out_dir)