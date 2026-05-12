import glob
import os

import numpy as np
import pandas as pd
from scipy import stats


def get_file_name(folder_name, subfolder_name, size, policy):
    rare = 'rare_' if 'rare' in subfolder_name else ''
    # file_name_rule = "reward_sim_eval_log_" + rare + str(size) + "_" + policy + ".csv"
    # file_list = glob.glob(os.path.join("output", folder_name, subfolder_name, file_name_rule))

    base = os.path.join("output", folder_name, subfolder_name)
    pattern1 = "reward_sim_eval_log_" + rare + str(size) + "_" + policy + ".csv"
    pattern2 = "reward_sim_eval_log_" + rare + str(size) + "_" + policy + "_0.csv"
    file_list = glob.glob(os.path.join(base, pattern1)) + \
                glob.glob(os.path.join(base, pattern2))

    if len(file_list) == 0:
        raise Exception("Error: file name rule %s not found" % {base, pattern1, pattern2})
    elif len(file_list) > 1:
        raise Exception("Error: multiple files found with name rule %s" % {base, pattern1, pattern2})
    else:
        return file_list[0]

def get_reward_list(file_name, include_errors):
    df = pd.read_csv(file_name, sep=';')
    if not include_errors: # removes errors
        df = df.loc[~df['traces'].str.contains('ERRORE')]
    reward_list = df['reward']
    count = len(df)
    avg = reward_list.mean()
    stdev = reward_list.std()
    return reward_list, count, avg, stdev

def create_bar_plot(df, output_file=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # --- Copy & type safety ---
    df = df.copy()
    df["size"] = df["size"].astype(int)
    df["avg"] = pd.to_numeric(df["avg"], errors="coerce")
    df["as_best"] = df["as_best"].astype(int)

    # --- Policy order ---
    policy_order = [
        "syn",
        "baseline",
        "cluster_none",
        "cluster_expQ1",
        "cluster_expQ2",
        "cluster_expQ3",
        "cluster_stepQ1",
        "cluster_stepQ2",
        "cluster_stepQ3",
        "cql_50",
        "cql_100",
    ]

    policies = [p for p in policy_order if p in df["policy"].unique()]

    # --- Legend labels (LaTeX) ---
    policy_labels = {
        "syn": r"$\pi_{customary}$",
        "baseline": r"$\pi_{informed}$",
        "cluster_none": r"$\pi^{MDP}_{0}$",
        "cluster_expQ1": r"$\pi^{MDP}_{sigQ1}$",
        "cluster_expQ2": r"$\pi^{MDP}_{sigQ2}$",
        "cluster_expQ3": r"$\pi^{MDP}_{sigQ3}$",
        "cluster_stepQ1": r"$\pi^{MDP}_{stepQ1}$",
        "cluster_stepQ2": r"$\pi^{MDP}_{stepQ2}$",
        "cluster_stepQ3": r"$\pi^{MDP}_{stepQ3}$",
        "cql_50": r"$\pi^{CQL}_{50}$",
        "cql_100": r"$\pi^{CQL}_{100}$",
    }

    # --- Hex pastel colors ---
    policy_colors = {
        # green gradient
        "syn": "#b6d7a8",
        "baseline": "#6aa84f",

        # light yellow
        "cluster_none": "#ffe599",

        # blue gradient
        "cluster_expQ1": "#9fc5e8",
        "cluster_expQ2": "#6fa8dc",
        "cluster_expQ3": "#3d85c6",

        # magenta gradient
        "cluster_stepQ1": "#d5a6bd",
        "cluster_stepQ2": "#c27ba0",
        "cluster_stepQ3": "#a64d79",

        # orange gradient
        "cql_50": "#f9cb9c",
        "cql_100": "#e69138",
    }

    # --- Data prep ---
    sizes = np.sort(df["size"].unique())
    n_sizes = len(sizes)
    n_policies = len(policies)

    x = np.arange(n_sizes)

    fig, ax = plt.subplots(figsize=(1.5 * n_sizes + 4, 6))

    # --- Bars ---
    for i, policy in enumerate(policies):
        heights = []
        stars = []

        for s in sizes:
            row = df[(df["size"] == s) & (df["policy"] == policy)]
            if row.empty:
                heights.append(0)
                stars.append(False)
            else:
                heights.append(row["avg"].iloc[0])
                stars.append(row["as_best"].iloc[0] == 1)

        # bar shape
        gap = 0.01
        bar_width = (0.8 - gap * (n_policies - 1)) / n_policies
        positions = x - 0.4 + i * (bar_width + gap) + bar_width / 2


        bars = ax.bar(
            positions,
            heights,
            bar_width,
            color=policy_colors[policy],
            label=policy_labels.get(policy, policy),
        )

        # Stars for best policy
        for bar, is_best in zip(bars, stars):
            if is_best:
                ax.plot(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    "*",
                    markersize=10,
                    color="black",
                    zorder=5,
                )

    # --- Title ---
    if 'rare' in output_file:
        ax.set_title("Synthetic logs rare", pad=12)
    else:
        ax.set_title("Synthetic logs", pad=12)

    # --- Axes ---
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_xlabel("log size")
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
        title="Policy",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
    )

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, bbox_inches="tight")
    else:
        plt.show()



def main():
    folder_name = "IS26_NEW"
    subfolder_name = "sim_event_log_rare"
    # size list
    size_list = [2000, 4000, 8000, 16000]
    # policy list
    policy_type_list = ['syn', 'baseline', 'none', 'stepQ1', 'stepQ2', 'stepQ3', 'expQ1', 'expQ2', 'expQ3', 'cql_50', 'cql_100']
    # define output file name
    output_file = os.path.join("output", folder_name, subfolder_name, subfolder_name + "_ptest_for_plot.csv")
    # define dataframe
    header = ['folder_name', 'subfolder_name', 'size', 'best_policy', 'policy', 'count', 'avg', 'stdev', 'pvalue', 'as_best']
    create_plot = True
    include_errors = True  # should include errors in reward avg

    output_df = pd.DataFrame(columns=header)



    result_csv = os.path.join("output", folder_name, subfolder_name, subfolder_name + "_results.csv")
    if include_errors:
        results_df = pd.read_csv(result_csv)[['policy_size', 'policy_type', 'policy_type_new', 'avg_reward_all']]
        results_df = results_df.rename(columns={"avg_reward_all": "avg_reward"})
    else:
        results_df = pd.read_csv(result_csv)[['policy_size', 'policy_type', 'policy_type_new', 'avg_reward']]

    results_df["rank"] = (
        results_df
        .groupby(["policy_size"])["avg_reward"]
        .rank(ascending=False, method="dense")
        .astype(int)
    )

    for size in size_list:
            df_ordered = (
                results_df.loc[results_df["policy_size"] == size]
                .sort_values("rank")
            )

            policy_ordered_list = df_ordered["policy_type"].tolist()
            policy_ordered_list_new = df_ordered["policy_type_new"].tolist()

            best_policy = policy_ordered_list[0]
            best_policy_name = policy_ordered_list_new[0]
            file1 = get_file_name(folder_name, subfolder_name, size, best_policy)
            reward_list1, count1, avg1, stdev1 = get_reward_list(file1, include_errors)
            for i, policy in enumerate(policy_ordered_list):
                policy_name = policy_ordered_list_new[i]
                rank2 = i+1
                file2 = get_file_name(folder_name, subfolder_name, size, policy)
                reward_list2, count2, avg2, stdev2 = get_reward_list(file2, include_errors)
                statistic, pvalue = stats.ttest_ind(reward_list1, reward_list2, equal_var=False)
                as_best = 1 if pvalue >= 0.05 else 0
                # write result
                value_list = [folder_name, subfolder_name, size, best_policy_name,
                              policy_name, count2, avg2, stdev2, pvalue, as_best]
                new_row = pd.DataFrame(np.array(value_list).reshape(1,10), columns=header)
                output_df = pd.concat([output_df, new_row], ignore_index=True)              # write file  # append
    output_df.to_csv(output_file, index=False)

    if create_plot:
        output_plot = os.path.join("output", folder_name, subfolder_name, subfolder_name + "_bar_plot.png")
        create_bar_plot(output_df, output_plot)


if __name__ == "__main__":
    main()