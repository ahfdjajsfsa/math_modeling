import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
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

out_dir = "Q2_results"
os.makedirs(out_dir, exist_ok=True)


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
    os.path.join(out_dir, "Q2_all_edges_SALI.csv"),
    index=False,
    encoding="utf-8-sig"
)

result.head(100).to_csv(
    os.path.join(out_dir, "Q2_top100_activity_cliffs.csv"),
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

mol_summary = df.copy()
mol_summary["Index"] = mol_summary.index

mol_summary = mol_summary.merge(
    mol_count,
    on="Index",
    how="left"
)

mol_summary["Cliff_Count"] = mol_summary["Cliff_Count"].fillna(0).astype(int)

mol_summary = mol_summary.sort_values(
    "Cliff_Count",
    ascending=False
)

mol_summary.to_csv(
    os.path.join(out_dir, "Q2_molecule_cliff_count.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\n参与活性悬崖最多的分子：")
print(mol_summary[["Index", "ID", "Bioactivity_Score", "Cliff_Count"]].head(10))


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
    os.path.join(out_dir, "Q2_SAS_map.png"),
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
    os.path.join(out_dir, "Q2_SALI_distribution.png"),
    dpi=400
)

plt.show()


# =========================
# 9. 绘制活性悬崖网络图
# =========================

try:
    import networkx as nx

    # 取前80条高SALI边，避免图太乱
    top_edges = result[
        result["Type"].isin(["Activity Cliff", "High SALI Cliff"])
    ].head(80)

    G = nx.Graph()

    for _, row in top_edges.iterrows():

        s = int(row["Source"])
        t = int(row["Target"])

        G.add_node(s, activity=activity[s])
        G.add_node(t, activity=activity[t])

        G.add_edge(
            s,
            t,
            weight=row["SALI"]
        )

    if len(G.nodes) > 0:

        pos = nx.spring_layout(G, seed=42)

        node_color = [G.nodes[n]["activity"] for n in G.nodes]

        node_size = []

        count_dict = dict(zip(mol_count["Index"], mol_count["Cliff_Count"]))

        for node in G.nodes:
            node_size.append(100 + count_dict.get(node, 1) * 30)

        plt.figure(figsize=(10, 8))

        nodes = nx.draw_networkx_nodes(
            G,
            pos,
            node_color=node_color,
            cmap="viridis",
            node_size=node_size,
            alpha=0.85
        )

        nx.draw_networkx_edges(
            G,
            pos,
            alpha=0.35,
            edge_color="gray"
        )

        nx.draw_networkx_labels(
            G,
            pos,
            font_size=8
        )

        cbar = plt.colorbar(nodes)
        cbar.set_label("Bioactivity Score")

        plt.title("Activity Cliff Network")
        plt.axis("off")
        plt.tight_layout()

        plt.savefig(
            os.path.join(out_dir, "Q2_activity_cliff_network.png"),
            dpi=400
        )

        plt.show()

except ImportError:
    print("未安装 networkx，跳过网络图。")


print("\n第二题完成，结果保存在文件夹：", out_dir)