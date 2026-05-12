import glob
import os

import numpy as np
import pandas as pd
from scipy import stats


def get_file_name(folder_name, subfolder_name, size, policy, cluster):
    file_name_rule = "reward_sim_eval_log*_" + str(size) + "_cluster_" + str(cluster) + "_" + policy + "_0.csv"
    file_list = glob.glob(os.path.join("output", folder_name, subfolder_name, file_name_rule))
    if len(file_list) == 0:
        raise Exception("Error: file name rule %s not found" % file_name_rule)
    elif len(file_list) > 1:
        raise Exception("Error: multiple files found with name rule %s" % file_name_rule)
    else:
        return file_list[0]

def get_reward_list(file_name):
    df = pd.read_csv(file_name, sep=';')
    not_error_df = df.loc[~df['traces'].str.contains('ERRORE')]
    reward_list = not_error_df['reward']
    count = len(not_error_df)
    avg = reward_list.mean()
    stdev = reward_list.std()
    return reward_list, count, avg, stdev

def create_bar_plot(df, output_file=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    # Make a copy to avoid modifying the original
    df = df.copy()

    # Ensure cluster and avg are numeric
    df["cluster"] = df["cluster"].astype(int)
    df["avg"] = pd.to_numeric(df["avg"], errors='coerce')  # convert to float
    df["as_best"] = df["as_best"].astype(int)

    # Sort by policy and cluster (numeric)
    df = df.sort_values(["policy", "cluster"])

    policies = df["policy"].unique()
    clusters = np.sort(df["cluster"].unique())

    n_policies = len(policies)
    n_clusters = len(clusters)

    # Color map: green → yellow → red
    cmap = LinearSegmentedColormap.from_list(
        "cluster_cmap", ["green", "yellow", "red"]
    )

    # Map cluster → color
    cluster_to_color = {
        c: cmap(i / (n_clusters - 1 if n_clusters > 1 else 1))
        for i, c in enumerate(clusters)
    }

    bar_width = 0.8 / n_clusters
    x = np.arange(n_policies)

    fig, ax = plt.subplots(figsize=(1.5 * n_policies + 4, 6))

    for i, cluster in enumerate(clusters):
        heights = []
        stars = []

        for p in policies:
            row = df[(df["policy"] == p) & (df["cluster"] == cluster)]
            if row.empty:
                heights.append(0)
                stars.append(False)
            else:
                heights.append(row["avg"].iloc[0])
                stars.append(row["as_best"].iloc[0] == 1)

        positions = x - 0.4 + i * bar_width + bar_width / 2

        bars = ax.bar(
            positions,
            heights,
            bar_width,
            color=cluster_to_color[cluster],
            label=f"{cluster}"
        )

        # Add stars
        for bar, is_best in zip(bars, stars):
            if is_best:
                ax.plot(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    "*",
                    markersize=10,
                    color="black",
                    zorder=5
                )

    # --- Legend labels (LaTeX) ---
    policy_labels = {
        "none": r"$\pi^{MDP}_{0}$",
        "expQ1": r"$\pi^{MDP}_{sigQ1}$",
        "expQ2": r"$\pi^{MDP}_{sigQ2}$",
        "expQ3": r"$\pi^{MDP}_{sigQ3}$",
        "stepQ1": r"$\pi^{MDP}_{stepQ1}$",
        "stepQ2": r"$\pi^{MDP}_{stepQ2}$",
        "stepQ3": r"$\pi^{MDP}_{stepQ3}$",
    }

    # --- Title ---
    if 'rare' in output_file:
        ax.set_title("Synthetic logs rare", pad=12)
    else:
        ax.set_title("Synthetic logs", pad=12)

    # --- Axes ---
    ax.set_xticks(x)
    ax.set_xticklabels([policy_labels.get(p, p) for p in policies])
    ax.set_xlabel("policy")
    ax.set_ylabel("avg KPI")

    # --- Grid ---
    ax.yaxis.grid(
        True,
        color="lightgrey",
        linestyle="--",
        linewidth=0.7,
        alpha=0.7,
    )
    ax.set_axisbelow(True)

    # --- Legend outside ---
    ax.legend(
        title="Cluster",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
    )

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()


def main():
    folder_name = "IS26_cluster_exp"
    subfolder_name = "sim_log_rare_evolution"
    # size list
    size_list = [4000]
    # policy list
    policy_type_list = ['none', 'stepQ1', 'stepQ2', 'stepQ3', 'expQ1', 'expQ2', 'expQ3']
    cluster_list = [40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
    # number_of_policies = len(policy_type_list)
    # number_of_clusters = len(cluster_list)
    # define output file name
    output_file = os.path.join("output", folder_name, subfolder_name, subfolder_name + "_ptest_cluster.csv")
    # define dataframe
    header = ['folder_name', 'subfolder_name', 'size', 'policy', 'best_cluster', 'cluster', 'count', 'avg', 'stdev', 'pvalue', 'as_best']
    create_plot = True
    include_errors = True

    output_df = pd.DataFrame(columns=header)

    result_csv = os.path.join("output", folder_name, subfolder_name, subfolder_name + "_results.csv")

    if include_errors:
        results_df = pd.read_csv(result_csv)[['policy_size', 'policy_type', 'policy_type_new', 'avg_reward_all']]
        results_df = results_df.rename(columns={"avg_reward_all": "avg_reward"})
    else:
        results_df = pd.read_csv(result_csv)[['policy_size', 'policy_type', 'policy_type_new', 'avg_reward']]

    results_df[['cluster', 'policy']] = results_df['policy_type'].str.split("_", expand=True).iloc[:, 1:3]
    results_df = results_df.drop(columns="policy_type")
    results_df["rank"] = (
        results_df
        .groupby(["policy_size", "policy"])["avg_reward"]
        .rank(ascending=False, method="dense")
        .astype(int)
    )

    for size in size_list:
        for policy in policy_type_list:
            cluster_ordered_list = results_df.loc[(results_df["policy_size"] == size) & (results_df["policy"] == policy)].sort_values("rank")["cluster"].tolist()
            best_cluster = cluster_ordered_list[0]
            file1 = get_file_name(folder_name, subfolder_name, size, policy, best_cluster)
            reward_list1, count1, avg1, stdev1 = get_reward_list(file1)
            for i, cluster2 in enumerate(cluster_ordered_list):
                rank2 = i+1
                file2 = get_file_name(folder_name, subfolder_name, size, policy, cluster2)
                reward_list2, count2, avg2, stdev2 = get_reward_list(file2)
                statistic, pvalue = stats.ttest_ind(reward_list1, reward_list2, equal_var=False)
                as_best = 1 if pvalue >= 0.05 else 0
                # write result
                value_list = [folder_name, subfolder_name, size, policy, best_cluster,
                              cluster2, count2, avg2, stdev2, pvalue, as_best]
                new_row = pd.DataFrame(np.array(value_list).reshape(1,11), columns=header)
                output_df = pd.concat([output_df, new_row], ignore_index=True)              # write file  # append
    output_df.to_csv(output_file, index=False)

    if create_plot:
        output_plot = os.path.join("output", folder_name, subfolder_name, subfolder_name + "_cluster_plot.png")
        create_bar_plot(output_df, output_plot)


if __name__ == "__main__":
    # this script performs statistical difference test between different policies and computes the corresponding p-value
    # the analysis is performed at different prefix lengths corresponding to the policy activation
    main()