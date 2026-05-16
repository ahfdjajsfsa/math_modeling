import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# =========================
# 0. 基本设置
# =========================

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

random.seed(42)
np.random.seed(42)

out_dir = "Q3_results"
os.makedirs(out_dir, exist_ok=True)

# 需要选择的代表性分子数量
# 如果题目规定选多少个，就改这里
K = 30

# 遗传算法参数
POP_SIZE = 80
N_GEN = 200
MUT_RATE = 0.2


# =========================
# 1. 读取附件1
# =========================

mol_file = r"D:\数据分析\附件1：molecular_interaction_manifest.csv"

df = pd.read_csv(
    mol_file,
    skiprows=[1],
    encoding="gb18030"
)

if "ID" not in df.columns:
    df["ID"] = df.index

target_col = "Bioactivity_Score"

df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

# 可用于理化性质多样性的列
desc_cols = [
    "LogP", "TPSA", "MolWt",
    "Dipole_Proxy", "Max_Partial_Charge",
    "Balaban_J", "Bertz_CT"
]

desc_cols = [c for c in desc_cols if c in df.columns]

for c in desc_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=[target_col] + desc_cols).reset_index(drop=True)

n = len(df)

print("有效分子数量：", n)
print("使用理化性质列：", desc_cols)

if K >= n:
    raise ValueError("K 不能大于等于分子总数，请调小 K。")


# =========================
# 2. 读取附件2：Tanimoto相似图
# =========================

edge_file = r"D:\数据分析\附件2：knn_graph_edges.csv"

edges = pd.read_csv(
    edge_file,
    skiprows=[1],
    encoding="gb18030"
)

edges["Source"] = pd.to_numeric(edges["Source"], errors="coerce")
edges["Target"] = pd.to_numeric(edges["Target"], errors="coerce")
edges["Tanimoto_Similarity"] = pd.to_numeric(
    edges["Tanimoto_Similarity"],
    errors="coerce"
)

edges = edges.dropna()
edges["Source"] = edges["Source"].astype(int)
edges["Target"] = edges["Target"].astype(int)

# 构造相似度矩阵
sim_matrix = np.zeros((n, n))

for _, row in edges.iterrows():
    i = row["Source"]
    j = row["Target"]
    s = row["Tanimoto_Similarity"]

    if i < n and j < n:
        sim_matrix[i, j] = s
        sim_matrix[j, i] = s

print("附件2有效边数量：", len(edges))


# =========================
# 3. 数据标准化
# =========================

X = df[desc_cols].values

# robust z-score，参考文献中也使用 robust 标准化思想
median = np.nanmedian(X, axis=0)
iqr = np.nanpercentile(X, 75, axis=0) - np.nanpercentile(X, 25, axis=0)
X_std = (X - median) / (iqr + 1e-12)

activity = df[target_col].values


# =========================
# 4. 定义三个目标函数
# =========================

def evaluate(ind):
    """
    ind: 一个分子索引列表
    返回：
    structure_score：结构相似度总和，越小越好
    property_score：理化性质距离总和，越大越好
    activity_score：活性标准差，越大越好
    """

    ind = list(ind)

    structure_score = 0
    property_score = 0

    for a in range(len(ind) - 1):
        for b in range(a + 1, len(ind)):
            i = ind[a]
            j = ind[b]

            # 结构相似度，越小越好
            structure_score += sim_matrix[i, j]

            # 理化性质欧氏距离，越大越好
            property_score += np.linalg.norm(X_std[i] - X_std[j])

    # 活性多样性，越大越好
    activity_score = np.std(activity[ind])

    return structure_score, property_score, activity_score


# =========================
# 5. 遗传算法基本操作
# =========================

def create_individual():
    return tuple(sorted(np.random.choice(n, K, replace=False)))


def crossover(p1, p2):
    pool = list(set(p1) | set(p2))

    if len(pool) >= K:
        child = np.random.choice(pool, K, replace=False)
    else:
        rest = list(set(range(n)) - set(pool))
        child = pool + list(np.random.choice(rest, K - len(pool), replace=False))

    return tuple(sorted(child))


def mutate(ind):
    ind = list(ind)

    if np.random.rand() < MUT_RATE:
        out_pos = np.random.randint(K)
        candidates = list(set(range(n)) - set(ind))
        ind[out_pos] = np.random.choice(candidates)

    return tuple(sorted(ind))


def normalize_score(values, reverse=False):
    values = np.array(values)
    if values.max() == values.min():
        return np.ones_like(values)
    s = (values - values.min()) / (values.max() - values.min())
    return 1 - s if reverse else s


# =========================
# 6. 运行简化遗传算法
# =========================

population = [create_individual() for _ in range(POP_SIZE)]

history = []
all_records = []

for gen in range(N_GEN):

    scores = []

    for ind in population:
        st, pr, ac = evaluate(ind)
        scores.append([st, pr, ac])

        all_records.append({
            "Generation": gen,
            "Individual": ",".join(map(str, ind)),
            "Structure_Score": st,
            "Property_Score": pr,
            "Activity_Score": ac
        })

    scores = np.array(scores)

    # 结构分数越小越好，所以 reverse=True
    s1 = normalize_score(scores[:, 0], reverse=True)
    s2 = normalize_score(scores[:, 1], reverse=False)
    s3 = normalize_score(scores[:, 2], reverse=False)

    # 简化综合适应度
    # 参考文献中更强调结构覆盖，这里给结构稍高权重
    fitness = 2 * s1 + s2 + s3

    # 保留前一半
    idx = np.argsort(fitness)[::-1]
    parents = [population[i] for i in idx[:POP_SIZE // 2]]

    # 生成下一代
    new_pop = parents.copy()

    while len(new_pop) < POP_SIZE:
        p1, p2 = random.sample(parents, 2)
        child = crossover(p1, p2)
        child = mutate(child)
        new_pop.append(child)

    population = new_pop

    history.append([
        gen,
        scores[:, 0].mean(),
        scores[:, 1].mean(),
        scores[:, 2].mean()
    ])

    if gen % 20 == 0:
        print(
            f"第 {gen} 代 | "
            f"结构相似度均值={scores[:,0].mean():.3f}, "
            f"性质多样性均值={scores[:,1].mean():.3f}, "
            f"活性多样性均值={scores[:,2].mean():.3f}"
        )


# =========================
# 7. 提取 Pareto 前沿
# =========================

all_df = pd.DataFrame(all_records)
all_df = all_df.drop_duplicates(subset=["Individual"]).reset_index(drop=True)


def is_dominated(row, others):
    """
    判断某个方案是否被其他方案支配。
    结构分数越小越好；
    性质分数越大越好；
    活性分数越大越好。
    """

    cond1 = others["Structure_Score"] <= row["Structure_Score"]
    cond2 = others["Property_Score"] >= row["Property_Score"]
    cond3 = others["Activity_Score"] >= row["Activity_Score"]

    strict = (
        (others["Structure_Score"] < row["Structure_Score"]) |
        (others["Property_Score"] > row["Property_Score"]) |
        (others["Activity_Score"] > row["Activity_Score"])
    )

    return ((cond1 & cond2 & cond3 & strict).sum() > 0)


pareto_mask = []

for i, row in all_df.iterrows():
    pareto_mask.append(not is_dominated(row, all_df.drop(i)))

pareto = all_df[pareto_mask].copy().reset_index(drop=True)

print("\nPareto 前沿方案数量：", len(pareto))


# =========================
# 8. 从 Pareto 前沿中选一个代表方案
# =========================

p1 = normalize_score(pareto["Structure_Score"], reverse=True)
p2 = normalize_score(pareto["Property_Score"], reverse=False)
p3 = normalize_score(pareto["Activity_Score"], reverse=False)

pareto["Composite_Score"] = 2 * p1 + p2 + p3

best = pareto.sort_values("Composite_Score", ascending=False).iloc[0]

best_indices = list(map(int, best["Individual"].split(",")))

selected = df.iloc[best_indices].copy()
selected["Selected_Index"] = best_indices

print("\n最终选择的代表性分子索引：")
print(best_indices)

print("\n最终方案三个目标：")
print(best[[
    "Structure_Score",
    "Property_Score",
    "Activity_Score",
    "Composite_Score"
]])


# =========================
# 9. 保存结果
# =========================

all_df.to_csv(
    os.path.join(out_dir, "Q3_all_generated_solutions.csv"),
    index=False,
    encoding="utf-8-sig"
)

pareto.to_csv(
    os.path.join(out_dir, "Q3_pareto_solutions.csv"),
    index=False,
    encoding="utf-8-sig"
)

selected.to_csv(
    os.path.join(out_dir, "Q3_selected_representative_molecules.csv"),
    index=False,
    encoding="utf-8-sig"
)


# =========================
# 10. 绘制遗传算法收敛图
# =========================

history = pd.DataFrame(
    history,
    columns=[
        "Generation",
        "Avg_Structure_Score",
        "Avg_Property_Score",
        "Avg_Activity_Score"
    ]
)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(history["Generation"], history["Avg_Structure_Score"], color="blue")
plt.xlabel("Generation")
plt.ylabel("Structure Score")
plt.title("结构相似度变化\n越低越好")

plt.subplot(1, 3, 2)
plt.plot(history["Generation"], history["Avg_Property_Score"], color="orange")
plt.xlabel("Generation")
plt.ylabel("Property Score")
plt.title("理化性质多样性变化\n越高越好")

plt.subplot(1, 3, 3)
plt.plot(history["Generation"], history["Avg_Activity_Score"], color="green")
plt.xlabel("Generation")
plt.ylabel("Activity Score")
plt.title("活性多样性变化\n越高越好")

plt.tight_layout()
plt.savefig(
    os.path.join(out_dir, "Q3_GA_convergence.png"),
    dpi=400
)
plt.show()


# =========================
# 11. 绘制 Pareto 目标空间图
# =========================

plt.figure(figsize=(8, 6))

plt.scatter(
    all_df["Structure_Score"],
    all_df["Property_Score"],
    c=all_df["Activity_Score"],
    cmap="viridis",
    s=15,
    alpha=0.35,
    label="Generated solutions"
)

plt.scatter(
    pareto["Structure_Score"],
    pareto["Property_Score"],
    c=pareto["Activity_Score"],
    cmap="viridis",
    s=45,
    edgecolor="black",
    label="Pareto front"
)

plt.scatter(
    best["Structure_Score"],
    best["Property_Score"],
    color="red",
    s=120,
    marker="*",
    label="Selected solution"
)

plt.colorbar(label="Activity Diversity")
plt.xlabel("Structure Score，越低越好")
plt.ylabel("Property Score，越高越好")
plt.title("第三题：多目标分子选择设计空间")

plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(out_dir, "Q3_pareto_space.png"),
    dpi=400
)
plt.show()


# =========================
# 12. 绘制所选分子的活性分布
# =========================

plt.figure(figsize=(8, 5))

plt.hist(
    df[target_col],
    bins=30,
    alpha=0.6,
    label="All molecules",
    color="lightgray"
)

plt.hist(
    selected[target_col],
    bins=15,
    alpha=0.8,
    label="Selected molecules",
    color="red"
)

plt.xlabel("Bioactivity Score")
plt.ylabel("Frequency")
plt.title("所选代表性分子的活性覆盖情况")
plt.legend()

plt.tight_layout()
plt.savefig(
    os.path.join(out_dir, "Q3_selected_activity_distribution.png"),
    dpi=400
)
plt.show()


print("\n第三题完成，结果保存在文件夹：", out_dir)