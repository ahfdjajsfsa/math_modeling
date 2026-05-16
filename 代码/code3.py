from pathlib import Path
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEED = 42
K = 8
POP_SIZE = 180
GENERATIONS = 260
CROSSOVER_RATE = 0.85
MUTATION_RATE = 0.28
FORCE_RECOMPUTE = False

random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "数据"
OUT_DIR = BASE_DIR / "数据处理"
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def read_csv_robust(path):
    last_error = None
    for encoding in ["utf-8-sig", "gbk", "gb18030", "latin1"]:
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as exc:
            last_error = exc
    raise last_error


def clean_columns(df):
    df = df.copy()
    df.columns = [str(c).replace("﻿", "").strip() for c in df.columns]
    return df


def keep_numeric_id_rows(df, id_col="ID"):
    df = clean_columns(df)
    df = df[pd.to_numeric(df[id_col], errors="coerce").notna()].copy()
    df[id_col] = pd.to_numeric(df[id_col], errors="coerce").astype(int)
    return df


def to_numeric_columns(df, exclude=("SMILES", "Region")):
    df = df.copy()
    for col in df.columns:
        if col not in exclude:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().any():
                df[col] = converted
    return df


def minmax(values):
    values = np.asarray(values, dtype=float)
    lo = np.nanmin(values)
    hi = np.nanmax(values)
    if math.isclose(hi, lo):
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


def load_data():
    manifest = read_csv_robust(DATA_DIR / "附件1：molecular_interaction_manifest.csv")
    manifest = keep_numeric_id_rows(manifest)
    manifest = to_numeric_columns(manifest)

    q1_path = OUT_DIR / "问题1_分子热点区域标记结果.csv"
    if q1_path.exists():
        q1 = read_csv_robust(q1_path)
        q1 = keep_numeric_id_rows(q1)
        q1 = q1[["ID", "Region", "Hotspot_Label"]]
        manifest = manifest.merge(q1, on="ID", how="left")
    else:
        manifest["Region"] = "Unknown"
        manifest["Hotspot_Label"] = -1

    q2_path = OUT_DIR / "问题2_分子局部敏感度统计.csv"
    if q2_path.exists():
        q2 = read_csv_robust(q2_path)
        q2 = keep_numeric_id_rows(q2)
        keep_cols = ["ID", "Local_Sensitivity", "Max_Local_SALI", "Cliff_Count", "Neighbor_Count"]
        q2 = q2[[col for col in keep_cols if col in q2.columns]]
        manifest = manifest.merge(q2, on="ID", how="left")
    else:
        manifest["Local_Sensitivity"] = np.nan
        manifest["Max_Local_SALI"] = np.nan
        manifest["Cliff_Count"] = np.nan
        manifest["Neighbor_Count"] = np.nan

    edges = read_csv_robust(DATA_DIR / "附件2：knn_graph_edges.csv")
    edges = clean_columns(edges)
    edges = edges[pd.to_numeric(edges["Source"], errors="coerce").notna()].copy()
    for col in ["Source", "Target", "Tanimoto_Similarity"]:
        edges[col] = pd.to_numeric(edges[col], errors="coerce")
    edges = edges.dropna(subset=["Source", "Target", "Tanimoto_Similarity"])
    edges["Source"] = edges["Source"].astype(int)
    edges["Target"] = edges["Target"].astype(int)

    if manifest["Local_Sensitivity"].isna().any():
        manifest = estimate_missing_sensitivity(manifest, edges)

    manifest["Region"] = manifest["Region"].fillna("Unknown")
    manifest["Hotspot_Label"] = manifest["Hotspot_Label"].fillna(-1).astype(int)
    manifest["Local_Sensitivity"] = manifest["Local_Sensitivity"].fillna(manifest["Local_Sensitivity"].median())
    return manifest.sort_values("ID").reset_index(drop=True), edges


def estimate_missing_sensitivity(df, edges):
    df = df.copy()
    activity = dict(zip(df["ID"], df["Bioactivity_Score"]))
    values = {int(i): [] for i in df["ID"]}
    for row in edges.itertuples(index=False):
        s = int(row.Source)
        t = int(row.Target)
        sim = float(row.Tanimoto_Similarity)
        if s in activity and t in activity:
            sali = abs(activity[s] - activity[t]) / max(1 - sim, 1e-6)
            values[s].append(sali)
            values[t].append(sali)
    estimated = {idx: np.mean(v) if v else np.nan for idx, v in values.items()}
    fallback = pd.Series(estimated).median()
    df["Local_Sensitivity"] = df["ID"].map(estimated).fillna(fallback)
    return df


def build_matrices(df, edges):
    ids = df["ID"].to_numpy(dtype=int)
    id_to_pos = {idx: pos for pos, idx in enumerate(ids)}
    n = len(df)

    sim = np.zeros((n, n), dtype=float)
    for row in edges.itertuples(index=False):
        source = int(row.Source)
        target = int(row.Target)
        if source in id_to_pos and target in id_to_pos:
            i = id_to_pos[source]
            j = id_to_pos[target]
            value = float(row.Tanimoto_Similarity)
            sim[i, j] = max(sim[i, j], value)
            sim[j, i] = max(sim[j, i], value)

    coords = df[["Manifold_X", "Manifold_Y"]].to_numpy(dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    spatial_dist = np.sqrt(np.sum(diff * diff, axis=2))

    prop_cols = [
        "LogP",
        "TPSA",
        "MolWt",
        "Dipole_Proxy",
        "Max_Partial_Charge",
        "Balaban_J",
        "Bertz_CT",
    ]
    props = df[prop_cols].to_numpy(dtype=float)
    props = (props - props.mean(axis=0)) / props.std(axis=0, ddof=0)
    props = np.nan_to_num(props)
    prop_diff = props[:, None, :] - props[None, :, :]
    prop_dist = np.sqrt(np.sum(prop_diff * prop_diff, axis=2))
    return sim, spatial_dist, prop_dist


def pair_mean(matrix, individual):
    values = []
    for a in range(len(individual)):
        for b in range(a + 1, len(individual)):
            values.append(matrix[individual[a], individual[b]])
    return float(np.mean(values)) if values else 0.0


def make_evaluator(df, sim_matrix, spatial_matrix, prop_matrix):
    activity = df["Bioactivity_Score"].to_numpy(dtype=float)
    sensitivity = df["Local_Sensitivity"].to_numpy(dtype=float)

    def evaluate(individual):
        ind = np.asarray(individual, dtype=int)
        mean_activity = float(activity[ind].mean())
        mean_sensitivity = float(sensitivity[ind].mean())
        mean_similarity = pair_mean(sim_matrix, ind)
        spatial_diversity = pair_mean(spatial_matrix, ind)
        property_diversity = pair_mean(prop_matrix, ind)
        objectives = np.array(
            [
                mean_activity,
                -mean_sensitivity,
                -mean_similarity,
                spatial_diversity,
            ],
            dtype=float,
        )
        raw = {
            "Mean_Activity": mean_activity,
            "Mean_Local_Sensitivity": mean_sensitivity,
            "Mean_Tanimoto_Similarity": mean_similarity,
            "Spatial_Diversity": spatial_diversity,
            "Property_Diversity": property_diversity,
        }
        return objectives, raw

    return evaluate


def dominates(a, b):
    return np.all(a >= b) and np.any(a > b)


def non_dominated_sort(fitness):
    size = len(fitness)
    dominated_sets = [[] for _ in range(size)]
    domination_count = np.zeros(size, dtype=int)
    fronts = [[]]

    for p in range(size):
        for q in range(size):
            if p == q:
                continue
            if dominates(fitness[p], fitness[q]):
                dominated_sets[p].append(q)
            elif dominates(fitness[q], fitness[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)

    current = 0
    while fronts[current]:
        next_front = []
        for p in fronts[current]:
            for q in dominated_sets[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        current += 1
        fronts.append(next_front)
    return fronts[:-1]


def crowding_distance(front, fitness):
    if not front:
        return {}
    if len(front) <= 2:
        return {idx: float("inf") for idx in front}

    distance = {idx: 0.0 for idx in front}
    values = np.asarray([fitness[idx] for idx in front], dtype=float)
    objective_count = values.shape[1]

    for obj in range(objective_count):
        order = np.argsort(values[:, obj])
        sorted_front = [front[i] for i in order]
        distance[sorted_front[0]] = float("inf")
        distance[sorted_front[-1]] = float("inf")
        span = values[order[-1], obj] - values[order[0], obj]
        if math.isclose(span, 0.0):
            continue
        for k in range(1, len(sorted_front) - 1):
            prev_value = values[order[k - 1], obj]
            next_value = values[order[k + 1], obj]
            distance[sorted_front[k]] += (next_value - prev_value) / span
    return distance


def select_nsga2(population, fitness, target_size):
    fronts = non_dominated_sort(fitness)
    selected = []
    for front in fronts:
        if len(selected) + len(front) <= target_size:
            selected.extend(front)
        else:
            distance = crowding_distance(front, fitness)
            ordered = sorted(front, key=lambda idx: distance[idx], reverse=True)
            selected.extend(ordered[: target_size - len(selected)])
            break
    return [population[idx] for idx in selected]


def individual_key(individual):
    return tuple(sorted(int(x) for x in individual))


def random_individual(n):
    return np.array(random.sample(range(n), K), dtype=int)


def seeded_population(df):
    n = len(df)
    activity_rank = minmax(df["Bioactivity_Score"].to_numpy(dtype=float))
    stability_rank = 1 - minmax(df["Local_Sensitivity"].to_numpy(dtype=float))
    priority = 0.62 * activity_rank + 0.38 * stability_rank
    top_pool = np.argsort(priority)[-max(60, K * 8) :]
    population = []

    for _ in range(POP_SIZE // 3):
        population.append(np.array(random.sample(list(top_pool), K), dtype=int))
    for _ in range(POP_SIZE - len(population)):
        population.append(random_individual(n))
    return population


def crossover(parent1, parent2, n):
    if random.random() > CROSSOVER_RATE:
        return parent1.copy()
    cut = random.randint(2, K - 2)
    child = list(random.sample(list(parent1), cut))
    for item in random.sample(list(parent2), K):
        if item not in child:
            child.append(int(item))
        if len(child) == K:
            break
    while len(child) < K:
        item = random.randrange(n)
        if item not in child:
            child.append(item)
    return np.array(child, dtype=int)


def mutate(individual, n):
    child = individual.copy().tolist()
    if random.random() < MUTATION_RATE:
        replace_count = 1 if random.random() < 0.8 else 2
        for _ in range(replace_count):
            pos = random.randrange(K)
            available = set(range(n)) - set(child)
            child[pos] = random.choice(tuple(available))
    return np.array(child, dtype=int)


def run_ga(df, evaluator):
    n = len(df)
    population = seeded_population(df)
    archive = {}
    history = []

    for generation in range(GENERATIONS + 1):
        evaluated = [evaluator(ind) for ind in population]
        fitness = [item[0] for item in evaluated]
        raw_metrics = [item[1] for item in evaluated]

        for ind, metrics in zip(population, raw_metrics):
            archive.setdefault(individual_key(ind), metrics)

        history.append(
            {
                "Generation": generation,
                "Mean_Activity": np.mean([m["Mean_Activity"] for m in raw_metrics]),
                "Mean_Local_Sensitivity": np.mean([m["Mean_Local_Sensitivity"] for m in raw_metrics]),
                "Mean_Tanimoto_Similarity": np.mean([m["Mean_Tanimoto_Similarity"] for m in raw_metrics]),
                "Spatial_Diversity": np.mean([m["Spatial_Diversity"] for m in raw_metrics]),
            }
        )

        if generation == GENERATIONS:
            break

        parents = select_nsga2(population, fitness, POP_SIZE)
        children = []
        while len(children) < POP_SIZE:
            p1, p2 = random.sample(parents, 2)
            child = crossover(p1, p2, n)
            child = mutate(child, n)
            children.append(child)

        combined = parents + children
        combined_eval = [evaluator(ind) for ind in combined]
        combined_fitness = [item[0] for item in combined_eval]
        population = select_nsga2(combined, combined_fitness, POP_SIZE)

    return archive, pd.DataFrame(history)


def archive_to_dataframe(archive, evaluator):
    rows = []
    for key in archive:
        ind = np.array(key, dtype=int)
        _, metrics = evaluator(ind)
        row = {"Selected_IDs": ";".join(map(str, key))}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def pareto_mask_from_dataframe(df):
    values = df[
        ["Mean_Activity", "Mean_Local_Sensitivity", "Mean_Tanimoto_Similarity", "Spatial_Diversity"]
    ].to_numpy(dtype=float)
    objectives = np.column_stack([values[:, 0], -values[:, 1], -values[:, 2], values[:, 3]])
    mask = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        if not mask[i]:
            continue
        for j in range(len(df)):
            if i != j and dominates(objectives[j], objectives[i]):
                mask[i] = False
                break
    return mask


def add_composite_score(df):
    out = df.copy()
    f1 = minmax(out["Mean_Activity"])
    f2 = minmax(out["Mean_Local_Sensitivity"])
    f3 = minmax(out["Mean_Tanimoto_Similarity"])
    f4 = minmax(out["Spatial_Diversity"])
    out["Composite_Score"] = 0.40 * f1 - 0.30 * f2 - 0.15 * f3 + 0.15 * f4
    return out


def selected_ids_from_row(row):
    return [int(x) for x in str(row["Selected_IDs"]).split(";")]


def solution_metrics(df, ids, sim_matrix, spatial_matrix, prop_matrix):
    positions = df.index[df["ID"].isin(ids)].to_numpy(dtype=int)
    return {
        "Mean_Activity": float(df.loc[positions, "Bioactivity_Score"].mean()),
        "Mean_Local_Sensitivity": float(df.loc[positions, "Local_Sensitivity"].mean()),
        "Mean_Tanimoto_Similarity": pair_mean(sim_matrix, positions),
        "Spatial_Diversity": pair_mean(spatial_matrix, positions),
        "Property_Diversity": pair_mean(prop_matrix, positions),
    }


def plot_convergence(history):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=160)
    plot_items = [
        ("Mean_Activity", "平均活性", "#d62728"),
        ("Mean_Local_Sensitivity", "平均局部敏感性", "#1f77b4"),
        ("Mean_Tanimoto_Similarity", "平均结构相似度", "#9467bd"),
        ("Spatial_Diversity", "空间分散度", "#2ca02c"),
    ]
    for ax, (col, title, color) in zip(axes.ravel(), plot_items):
        ax.plot(history["Generation"], history[col], color=color, linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("迭代代数")
        ax.grid(alpha=0.25)
    fig.suptitle("问题3：遗传算法种群指标收敛趋势", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "问题3_遗传算法收敛趋势.png", bbox_inches="tight")
    plt.close(fig)


def plot_pareto(_all_solutions, pareto, final_row):
    fig, ax = plt.subplots(figsize=(11, 7), dpi=180)

    display = pareto.sort_values("Composite_Score", ascending=False).head(30).copy().reset_index(drop=True)
    x = np.arange(len(display)) + 1

    ax.plot(
        x,
        display["Composite_Score"],
        color="#d62728",
        linewidth=2.4,
        marker="o",
        markersize=5,
        label="综合评分",
    )
    ax_twin = ax.twinx()
    ax_twin.plot(
        x,
        display["Mean_Activity"],
        color="#2ca02c",
        linewidth=1.9,
        marker="s",
        markersize=4.5,
        alpha=0.9,
        label="平均活性",
    )
    ax_twin.plot(
        x,
        display["Mean_Local_Sensitivity"],
        color="#1f77b4",
        linewidth=1.9,
        marker="^",
        markersize=4.5,
        alpha=0.9,
        label="平均局部敏感性",
    )
    ax_twin.plot(
        x,
        display["Mean_Tanimoto_Similarity"],
        color="#9467bd",
        linewidth=1.7,
        marker="D",
        markersize=3.8,
        alpha=0.85,
        label="平均结构相似度",
    )

    final_rank = None
    for idx, ids in enumerate(display["Selected_IDs"]):
        if ids == final_row["Selected_IDs"]:
            final_rank = idx + 1
            break
    if final_rank is not None:
        ax.axvline(final_rank, color="red", linestyle="--", linewidth=1.4, alpha=0.8)
        ax.scatter(
            [final_rank],
            [display.loc[final_rank - 1, "Composite_Score"]],
            marker="*",
            c="red",
            s=260,
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
            label="最终方案",
        )
        ax.text(
            final_rank + 0.4,
            display.loc[final_rank - 1, "Composite_Score"],
            "最终方案",
            color="red",
            va="center",
            fontsize=10,
        )

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax.set_xlabel("Pareto 方案综合评分排名")
    ax.set_ylabel("综合评分")
    ax_twin.set_ylabel("原始指标值")
    ax.set_title("问题3：Top 30 Pareto 方案指标对比")
    ax.set_xticks(x)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "问题3_Pareto前沿分布.png", bbox_inches="tight")
    plt.close(fig)


def plot_selected_on_manifold(df, selected_ids):
    selected = df[df["ID"].isin(selected_ids)].copy()
    fig, ax = plt.subplots(figsize=(10, 8), dpi=160)
    sc = ax.scatter(
        df["Manifold_X"],
        df["Manifold_Y"],
        c=df["Bioactivity_Score"],
        cmap="YlGnBu",
        s=35,
        alpha=0.55,
        edgecolors="none",
    )
    ax.scatter(
        selected["Manifold_X"],
        selected["Manifold_Y"],
        marker="*",
        c="red",
        s=240,
        edgecolors="black",
        linewidths=0.8,
        label="推荐分子",
    )
    for row in selected.itertuples(index=False):
        ax.annotate(
            str(row.ID),
            (row.Manifold_X, row.Manifold_Y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            color="black",
            weight="bold",
        )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Bioactivity Score")
    ax.set_xlabel("Manifold X")
    ax.set_ylabel("Manifold Y")
    ax.set_title("问题3：最终推荐 8 个分子在二维流形空间中的位置")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "问题3_推荐分子二维空间分布.png", bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(df, selected_ids, sim_matrix, spatial_matrix, prop_matrix):
    top_activity_ids = df.sort_values("Bioactivity_Score", ascending=False).head(K)["ID"].tolist()
    selected_metrics = solution_metrics(df, selected_ids, sim_matrix, spatial_matrix, prop_matrix)
    top_metrics = solution_metrics(df, top_activity_ids, sim_matrix, spatial_matrix, prop_matrix)

    all_summary = {
        "Mean_Activity": float(df["Bioactivity_Score"].mean()),
        "Mean_Local_Sensitivity": float(df["Local_Sensitivity"].mean()),
        "Mean_Tanimoto_Similarity": float(sim_matrix[np.triu_indices(len(df), 1)].mean()),
        "Spatial_Diversity": float(spatial_matrix[np.triu_indices(len(df), 1)].mean()),
        "Property_Diversity": float(prop_matrix[np.triu_indices(len(df), 1)].mean()),
    }
    comp = pd.DataFrame(
        [all_summary, top_metrics, selected_metrics],
        index=["全体平均", "仅按活性前8", "本文推荐8个"],
    )
    comp.to_csv(OUT_DIR / "问题3_推荐方案与基准方案指标比较.csv", encoding="utf-8-sig")

    display = pd.DataFrame(index=comp.index)
    display["活性水平"] = minmax(comp["Mean_Activity"])
    display["稳定性"] = 1 - minmax(comp["Mean_Local_Sensitivity"])
    display["结构多样性"] = 1 - minmax(comp["Mean_Tanimoto_Similarity"])
    display["空间分散性"] = minmax(comp["Spatial_Diversity"])
    display["理化性质多样性"] = minmax(comp["Property_Diversity"])

    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    x = np.arange(len(display.columns))
    width = 0.25
    colors = ["#999999", "#1f77b4", "#d62728"]
    for idx, name in enumerate(display.index):
        ax.bar(x + (idx - 1) * width, display.loc[name], width=width, label=name, color=colors[idx])
    ax.set_xticks(x)
    ax.set_xticklabels(display.columns)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("归一化表现（越高越好）")
    ax.set_title("问题3：推荐方案与基准方案的综合比较")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "问题3_推荐方案指标对比.png", bbox_inches="tight")
    plt.close(fig)


def plot_selected_radar(selected_df):
    labels = ["活性", "低敏感", "邻居数", "低悬崖次数"]
    values = np.column_stack(
        [
            minmax(selected_df["Bioactivity_Score"]),
            1 - minmax(selected_df["Local_Sensitivity"]),
            minmax(selected_df["Neighbor_Count"].fillna(0)),
            1 - minmax(selected_df["Cliff_Count"].fillna(0)),
        ]
    )
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"}, dpi=160)
    for row_values, molecule_id in zip(values, selected_df["ID"]):
        row = row_values.tolist() + row_values[:1].tolist()
        ax.plot(angles, row, linewidth=1.2, alpha=0.65, label=str(molecule_id))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("问题3：推荐分子单体指标雷达图")
    ax.legend(loc="upper right", bbox_to_anchor=(1.22, 1.08), title="ID", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "问题3_推荐分子单体雷达图.png", bbox_inches="tight")
    plt.close(fig)


def save_summary(final_row, selected_ids, pareto_count, total_count):
    lines = [
        "问题3：8个优先验证分子筛选结果摘要",
        "",
        f"全部生成的不同候选方案数：{total_count}",
        f"Pareto 前沿候选方案数：{pareto_count}",
        f"最终推荐分子 ID：{', '.join(map(str, selected_ids))}",
        "",
        "最终方案集合指标：",
        f"平均活性：{final_row['Mean_Activity']:.4f}",
        f"平均局部敏感性：{final_row['Mean_Local_Sensitivity']:.4f}",
        f"平均结构相似度：{final_row['Mean_Tanimoto_Similarity']:.4f}",
        f"二维空间分散度：{final_row['Spatial_Diversity']:.4f}",
        f"理化性质分散度：{final_row['Property_Diversity']:.4f}",
        f"综合评分：{final_row['Composite_Score']:.4f}",
        "",
        "输出文件：",
        "问题3_最终推荐8个分子.csv",
        "问题3_Pareto候选方案.csv",
        "问题3_所有生成候选方案.csv",
        "问题3_遗传算法收敛趋势.png",
        "问题3_Pareto前沿分布.png",
        "问题3_推荐分子二维空间分布.png",
        "问题3_推荐方案指标对比.png",
        "问题3_推荐分子单体雷达图.png",
    ]
    (OUT_DIR / "问题3_结果摘要.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    df, edges = load_data()
    sim_matrix, spatial_matrix, prop_matrix = build_matrices(df, edges)
    evaluator = make_evaluator(df, sim_matrix, spatial_matrix, prop_matrix)

    all_path = OUT_DIR / "问题3_所有生成候选方案.csv"
    pareto_path = OUT_DIR / "问题3_Pareto候选方案.csv"
    history_path = OUT_DIR / "问题3_遗传算法收敛数据.csv"

    if (not FORCE_RECOMPUTE) and all_path.exists() and pareto_path.exists() and history_path.exists():
        all_solutions = read_csv_robust(all_path)
        pareto = read_csv_robust(pareto_path)
        history = read_csv_robust(history_path)
        print("检测到已有问题3结果，跳过遗传算法搜索，仅重新生成图表。")
    else:
        archive, history = run_ga(df, evaluator)
        all_solutions = archive_to_dataframe(archive, evaluator)
        all_solutions = add_composite_score(all_solutions)
        all_solutions = all_solutions.sort_values("Composite_Score", ascending=False).reset_index(drop=True)
        all_solutions.to_csv(all_path, index=False, encoding="utf-8-sig")

        pareto = all_solutions[pareto_mask_from_dataframe(all_solutions)].copy()
        pareto = add_composite_score(pareto).sort_values("Composite_Score", ascending=False).reset_index(drop=True)
        pareto.to_csv(pareto_path, index=False, encoding="utf-8-sig")
        history.to_csv(history_path, index=False, encoding="utf-8-sig")

    final_row = pareto.iloc[0]
    selected_ids = selected_ids_from_row(final_row)
    selected_df = df[df["ID"].isin(selected_ids)].copy()
    selected_df["Selection_Order"] = selected_df["ID"].map({mid: i + 1 for i, mid in enumerate(selected_ids)})
    selected_df = selected_df.sort_values("Selection_Order")

    priority_base = selected_df[["Bioactivity_Score", "Local_Sensitivity"]].copy()
    selected_df["Single_Molecule_Priority"] = (
        0.6 * minmax(priority_base["Bioactivity_Score"]) - 0.4 * minmax(priority_base["Local_Sensitivity"])
    )
    selected_df.to_csv(OUT_DIR / "问题3_最终推荐8个分子.csv", index=False, encoding="utf-8-sig")

    plot_convergence(history)
    plot_pareto(all_solutions, pareto, final_row)
    plot_selected_on_manifold(df, selected_ids)
    plot_metric_comparison(df, selected_ids, sim_matrix, spatial_matrix, prop_matrix)
    plot_selected_radar(selected_df)
    save_summary(final_row, selected_ids, len(pareto), len(all_solutions))

    print("问题3筛选完成。")
    print(f"最终推荐分子 ID：{', '.join(map(str, selected_ids))}")
    print(f"结果已保存到：{OUT_DIR}")


if __name__ == "__main__":
    main()
