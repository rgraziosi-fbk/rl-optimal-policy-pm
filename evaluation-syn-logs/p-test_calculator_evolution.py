import glob
import os

import numpy as np
import pandas as pd
from pm4py.algo.clustering.trace_attribute_driven.variants.act_dist_calc import act_sim_percent_avg_actset
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def get_file_name_evolution(folder_name, subfolder_name, size, policy, starting_at):
    file_name_rule = "reward_sim_eval_log*_" + str(size) + "*" + policy + "_" + str(starting_at) + ".csv"
    file_list = glob.glob(os.path.join("output", folder_name, subfolder_name, file_name_rule))
    if len(file_list) == 0:
        raise Exception("Error: file name rule %s not found" % file_name_rule)
    elif len(file_list) > 1:
        raise Exception("Error: multiple files found with name rule %s" % file_name_rule)
    else:
        return file_list[0]

def get_reward_list(file_name, starting_at):
    df = pd.read_csv(file_name, sep=';')
    not_error_df = df.loc[~df['traces'].str.contains('ERRORE')]
    not_error_df = not_error_df[not_error_df['len_trace'] >= starting_at]
    count = len(not_error_df)
    reward_list = not_error_df['reward']
    avg = reward_list.mean()
    stdev = reward_list.std()
    return reward_list, count, avg, stdev

def define_winners(df):
    # tie is NOT consider winning
    df['winner1'] = np.where(((df['avg1'] > df['avg2']) & (df['pvalue'] < 0.05)), True, False)
    df['winner2'] = np.where(((df['avg1'] < df['avg2']) & (df['pvalue'] < 0.05)), True, False)
    return df

def compute_delta_avg(df):
    # compute delta avg KPI wrt to customary policy
    df['delta_avg1'] = df['avg1'] - df['customary_avg']
    df['delta_avg2'] = df['avg2'] - df['customary_avg']
    return df

def create_plot_subdf(folder_name, subfolder_name, size, starting_at, policy1, policy2, reward_list1, reward_list2):
    l1 = len(reward_list1)
    l2 = len(reward_list2)
    ltot = l1 + l2
    df_dict = dict()
    df_dict['folder_name'] = [folder_name for _ in range(ltot)]
    df_dict['subfolder_name'] = [subfolder_name for _ in range(ltot)]
    df_dict['size'] = [size for _ in range(ltot)]
    df_dict['starting_at'] = [starting_at for _ in range(ltot)]
    df_dict['policy'] = [policy1 for _ in range(l1)] + [policy2 for _ in range(l2)]
    df_dict['reward'] = reward_list1.to_list() + reward_list2.to_list()
    return pd.DataFrame.from_dict(df_dict)


def main():
    folder_name = "folder1"
    subfolder_name = "sim_log_rare_evolution"
    do_plot = True
    # size list
    size_list = [2000, 4000, 8000, 16000]
    # policy list
    policy_type_list = ['expQ1', 'cql_100']  # compare only two policies
    # starting_at list
    staring_at_list = [0, 5, 10, 15, 20, 25]
    number_of_policies = len(policy_type_list)
    # define output file name
    output_file = os.path.join("output", folder_name, subfolder_name, subfolder_name + "_ptest.csv")
    # define dataframe
    header = ['folder_name', 'subfolder_name', 'size', 'starting_at', 'policy1', 'policy2', 'count1', 'avg1', 'stdev1', 'count2', 'avg2', 'stdev2', 'pvalue']
    output_df = pd.DataFrame(columns=header)
    # header_plot = ['folder_name', 'subfolder_name', 'size', 'starting_at', 'policy', 'reward']
    for size in size_list:
        # plot_df = pd.DataFrame(columns=header_plot)  # other try to do the plots
        for starting_at in staring_at_list:
            policy1 = policy_type_list[0]
            policy2 = policy_type_list[1]
            file1 = get_file_name_evolution(folder_name, subfolder_name, size, policy1, starting_at)
            file2 = get_file_name_evolution(folder_name, subfolder_name, size, policy2, starting_at)
            reward_list1, count1, avg1, stdev1 = get_reward_list(file1, starting_at)
            reward_list2, count2, avg2, stdev2 = get_reward_list(file2, starting_at)
            statistic, pvalue = stats.ttest_ind(reward_list1, reward_list2, equal_var=False)
            # write result
            value_list = [folder_name, subfolder_name, size, starting_at, policy1, policy2,
                          count1, avg1, stdev1, count2, avg2, stdev2, pvalue]
            # plot_df = pd.concat([plot_df, create_plot_subdf(folder_name, subfolder_name, size, starting_at, policy1, policy2, reward_list1, reward_list2)], ignore_index=True)
            new_row = pd.DataFrame(np.array(value_list).reshape(1,13), columns=header)
            output_df = pd.concat([output_df, new_row], ignore_index=True)  # write file  # append
        # if do_plot:
        #     g = sns.catplot(
        #         data=plot_df, kind="bar",
        #         x="starting_at", y="reward", hue="policy",
        #         errorbar="sd", palette="deep", alpha=1  # , height=6
        #     )
        #     plt.show()
    cols = output_df.columns.drop(['folder_name', 'subfolder_name', 'policy1', 'policy2'])
    output_df[cols] = output_df[cols].apply(pd.to_numeric, errors='coerce')
    output_df = define_winners(output_df)
    output_df.to_csv(output_file, index=False)
    # plot
    if do_plot:
        create_plot(output_df, size_list, output_file)


# for main_all
def evaluate_policy(folder_name, subfolder_name, size, policy_type, starting_at):
    """Compute statistics for a single policy."""
    policy_file = get_file_name_evolution(
        folder_name, subfolder_name, size, policy_type, starting_at
    )

    reward_list, count, avg, stdev = get_reward_list(policy_file, starting_at)

    return {
        'policy_type': policy_type,
        'reward_list': reward_list,
        'count': count,
        'avg': avg,
        'stdev': stdev
    }

# for main_all
def define_winners_all(policies):
    """Given a list of policy dicts, assign winner flags."""
    # Find best policy
    best = max(policies, key=lambda x: x['avg'])
    best_rewards = best['reward_list']

    results = []
    for p in policies:
        if p['policy_type'] == best['policy_type']:
            winner = True
            p_value = None
        else:
            _, p_value = stats.ttest_ind(
                best_rewards,
                p['reward_list'],
                equal_var=False
            )
            winner = p_value > 0.05

        results.append({
            **p,
            'p_value_vs_best': p_value,
            'winner': winner,
            'is_best_avg': p['policy_type'] == best['policy_type']
        })

    return results



def main_all():
    """Second version to compare 2 more policies, with also customary and baseline"""
    folder_name = "IS26_NEW"
    subfolder_name = "sim_log_evolution"
    do_plot = True
    # size list
    size_list = [2000, 4000, 8000, 16000]
    # policy list
    policy_type_list = ['baseline', 'expQ1', 'cql_100']
    # starting_at list
    starting_at_list = [0, 5, 10, 15, 20, 25]
    # define output file name
    output_file = os.path.join("output", folder_name, subfolder_name, subfolder_name + "_all_ptest.csv")

    rows = []
    for size in size_list:
        for starting_at in starting_at_list:
            # ---- Evaluate all policies ----
            policies = [
                evaluate_policy(
                    folder_name,
                    subfolder_name,
                    2000 if pt in ['syn', 'baseline'] else size,
                    pt,
                    starting_at
                )
                for pt in policy_type_list
            ]

            # ---- Compute winners ----
            policies = define_winners_all(policies)

            # ---- Add metadata and flatten ----
            for p in policies:
                rows.append({
                    'folder_name': folder_name,
                    'subfolder_name': subfolder_name,
                    'size': size,
                    'starting_at': starting_at,
                    **{k: v for k, v in p.items() if k != 'reward_list'}
                })

    output_df = pd.DataFrame(rows)
    output_df.to_csv(output_file, index=False)

    # plot
    if do_plot:
        create_plot_all(output_df, size_list, output_file)


def main_new():
    """wrt to old version in this one we consider the difference
    between KPI of learned policy and customary one"""
    folder_name = "IS26_NEW"
    subfolder_name = "sim_log_rare_evolution"
    do_plot = True
    # size list
    size_list = [2000, 4000, 8000, 16000]
    # size_list = [2000]
    # policy list
    policy_type_list = ['expQ1', 'stepQ1']
    # policy_type_list = ['expQ2', 'baseline']
    baseline_policy = 'baseline'
    customary_policy = 'syn'
    # starting_at list
    staring_at_list = [0, 5, 10, 15, 20, 25, 30]
    number_of_policies = len(policy_type_list)
    # define output file name
    output_file = os.path.join("output", folder_name, subfolder_name, subfolder_name + "_new_ptest.csv")
    # define dataframe
    header = ['folder_name', 'subfolder_name', 'size', 'starting_at',
              'policy1', 'policy2', 'count1', 'avg1', 'stdev1', 'count2', 'avg2', 'stdev2',
              'pvalue', 'customary_avg', 'customary_stdev', 'customary_count']
    output_df = pd.DataFrame(columns=header)
    # header_plot = ['folder_name', 'subfolder_name', 'size', 'starting_at', 'policy', 'reward']
    for size in size_list:
        # plot_df = pd.DataFrame(columns=header_plot)  # other try to do the plots
        for starting_at in staring_at_list:
            policy1 = policy_type_list[0]
            policy2 = policy_type_list[1]
            file1 = get_file_name_evolution(folder_name, subfolder_name, size, policy1, starting_at)
            file2 = get_file_name_evolution(folder_name, subfolder_name, size, policy2, starting_at)
            file3 = get_file_name_evolution(folder_name, subfolder_name, '2000', customary_policy, starting_at)
            reward_list1, count1, avg1, stdev1 = get_reward_list(file1, starting_at)
            reward_list2, count2, avg2, stdev2 = get_reward_list(file2, starting_at)
            reward_list3, count3, avg3, stdev3 = get_reward_list(file3, starting_at)  # customary policy
            statistic, pvalue = stats.ttest_ind(reward_list1, reward_list2, equal_var=False)
            # write result
            value_list = [folder_name, subfolder_name, size, starting_at, policy1, policy2,
                          count1, avg1, stdev1, count2, avg2, stdev2, pvalue, avg3, stdev3, count3]
            # plot_df = pd.concat([plot_df, create_plot_subdf(folder_name, subfolder_name, size, starting_at, policy1, policy2, reward_list1, reward_list2)], ignore_index=True)
            new_row = pd.DataFrame(np.array(value_list).reshape(1,16), columns=header)
            output_df = pd.concat([output_df, new_row], ignore_index=True)  # write file  # append
        # if do_plot:
        #     g = sns.catplot(
        #         data=plot_df, kind="bar",
        #         x="starting_at", y="reward", hue="policy",
        #         errorbar="sd", palette="deep", alpha=1  # , height=6
        #     )
        #     plt.show()
    cols = output_df.columns.drop(['folder_name', 'subfolder_name', 'policy1', 'policy2'])
    output_df[cols] = output_df[cols].apply(pd.to_numeric, errors='coerce')
    output_df = define_winners(output_df)
    output_df = compute_delta_avg(output_df)
    output_df.to_csv(output_file, index=False)
    # plot
    if do_plot:
        create_plot_differences(output_df, size_list, output_file)


def create_plot(output_df, size_list, output_file):

    plot_title = 'Synthetic Event Logs Rare' if '_rare_' in output_file else 'Synthetic Event Logs'

    header_plot = ['folder_name', 'subfolder_name', 'size', 'starting_at', 'policy', 'avg',
                   'stdev', 'winner']
    header1 = ['folder_name', 'subfolder_name', 'size', 'starting_at', 'policy1', 'avg1',
               'stdev1', 'winner1']
    header2 = ['folder_name', 'subfolder_name', 'size', 'starting_at', 'policy2', 'avg2',
               'stdev2', 'winner2']

    # Legend relabeling (policy name -> LaTeX label)
    policy_labels = {
        "expQ2": r"$\pi^{\text{MDP}}_{\text{sig}Q_2}$",
        "cql_100": r"$\pi^{\text{CQL}}_{100}$",
        "baseline": r"baseline",
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    fig.suptitle(plot_title)

    for index, size in enumerate(size_list):
        df1 = pd.DataFrame()
        df2 = pd.DataFrame()
        df1[header_plot] = output_df[output_df['size'] == size][header1]
        df2[header_plot] = output_df[output_df['size'] == size][header2]
        plot_df = pd.concat([df1, df2], ignore_index=True)

        winning = [df1['winner'].to_list(), df2['winner'].to_list()]
        ax = axes[index // 2, index % 2]

        g = sns.barplot(
            ax=ax,
            data=plot_df,
            x="starting_at", y="avg", hue="policy",
            errorbar=None, palette="tab10",
            alpha=1
        )
        g.set(xlabel="from prefix len", ylabel="mean future KPI", title='size ' + str(size))

        # (1) Horizontal grey grid lines
        ax.yaxis.grid(True, color="lightgrey", linestyle="--", linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)

        # Stars for winners
        for i, container in enumerate(g.axes.containers):
            for j, bar in enumerate(container.patches):
                if winning[i][j]:
                    g.plot(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        "*",
                        markersize=10,
                        color="black"
                    )

        # (2) Rename legend entries using policy_labels
        handles, labels = ax.get_legend_handles_labels()
        new_labels = [policy_labels.get(lbl, lbl) for lbl in labels]
        ax.legend(handles, new_labels, title="policy")

    plt.subplots_adjust(
        left=0.1,
        bottom=0.05,
        right=0.9,
        top=0.92,
        wspace=0.2,
        hspace=0.25
    )

    plt.savefig(output_file.replace('_ptest.csv', '_plot.png'))



def create_plot_all(output_df, size_list, output_file):
    """wrt to the create plot here more than 2 policies can be considered"""

    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_title = 'Synthetic Event Logs Rare' if '_rare_' in output_file else 'Synthetic Event Logs'

    # --- Legend labels ---
    policy_labels = {
        "syn": r"$\pi_{customary}$",
        "baseline": r"$\pi_{informed}$",
        "none": r"$\pi^{MDP}_{0}$",
        "expQ1": r"$\pi^{MDP}_{sigQ1}$",
        "expQ2": r"$\pi^{MDP}_{sigQ2}$",
        "expQ3": r"$\pi^{MDP}_{sigQ3}$",
        "stepQ1": r"$\pi^{MDP}_{stepQ1}$",
        "stepQ2": r"$\pi^{MDP}_{stepQ2}$",
        "stepQ3": r"$\pi^{MDP}_{stepQ3}$",
        "cql_50": r"$\pi^{CQL}_{50}$",
        "cql_100": r"$\pi^{CQL}_{100}$",
    }

    # --- Colors ---
    policy_colors = {
        "syn": "#b6d7a8",
        "baseline": "#6aa84f",
        "none": "#ffe599",
        "expQ1": "#9fc5e8",
        "expQ2": "#6fa8dc",
        "expQ3": "#3d85c6",
        "stepQ1": "#d5a6bd",
        "stepQ2": "#c27ba0",
        "stepQ3": "#a64d79",
        "cql_50": "#f9cb9c",
        "cql_100": "#e69138",
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    fig.suptitle(plot_title)

    for index, size in enumerate(size_list):
        ax = axes[index // 2, index % 2]

        plot_df = output_df[output_df['size'] == size].copy()
        plot_df = plot_df.sort_values("starting_at")

        # keep only policies present in this subset
        present_policies = plot_df["policy_type"].unique().tolist()

        # keep order consistent with your dictionary (important!)
        hue_order = [p for p in policy_colors.keys() if p in present_policies]

        # filter palette accordingly
        palette = {p: policy_colors[p] for p in hue_order}

        g = sns.barplot(
            ax=ax,
            data=plot_df,
            x="starting_at",
            y="avg",
            hue="policy_type",
            hue_order=hue_order,
            palette=palette,
            errorbar=None
        )

        ax.set(
            xlabel="from prefix len",
            ylabel="mean future KPI",
            title=f"size {size}"
        )

        # Grid
        ax.yaxis.grid(True, color="lightgrey", linestyle="--", linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)

        # ---- Stars (aligned with actual plotted order) ----
        for container, policy in zip(g.containers, hue_order):
            policy_data = plot_df[plot_df["policy_type"] == policy]

            for bar, (_, row) in zip(container.patches, policy_data.iterrows()):
                if row["winner"]:
                    ax.plot(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        "*",
                        markersize=10,
                        color="black"
                    )

        # ---- Clean legend (only present policies) ----
        handles, labels = ax.get_legend_handles_labels()
        new_labels = [policy_labels.get(lbl, lbl) for lbl in labels]
        ax.legend(handles, new_labels, title="policy")

    plt.subplots_adjust(
        left=0.1,
        bottom=0.05,
        right=0.9,
        top=0.92,
        wspace=0.2,
        hspace=0.25
    )

    plt.savefig(output_file.replace('_ptest.csv', '_plot.png'))

def create_plot_differences(output_df, size_list, output_file):
    """wrt to creat plot this compute the delta KPI of the learned policies wrt the customary policy"""
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_title = 'Synthetic Event Logs Rare' if '_rare_' in output_file else 'Synthetic Event Logs'

    header_plot = ['folder_name', 'subfolder_name', 'size', 'starting_at', 'policy', 'delta_avg',
                   'stdev', 'winner']
    header1 = ['folder_name', 'subfolder_name', 'size', 'starting_at', 'policy1', 'delta_avg1',
               'stdev1', 'winner1']
    header2 = ['folder_name', 'subfolder_name', 'size', 'starting_at', 'policy2', 'delta_avg2',
               'stdev2', 'winner2']

    # Legend relabeling (policy name -> LaTeX label)
    policy_labels = {
        "expQ2": r"$\pi^{\text{MDP}}_{\text{sig}Q_2}$",
        "cql_100": r"$\pi^{\text{CQL}}_{100}$",
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    fig.suptitle(plot_title)

    for index, size in enumerate(size_list):
        df1 = pd.DataFrame()
        df2 = pd.DataFrame()
        df1[header_plot] = output_df[output_df['size'] == size][header1]
        df2[header_plot] = output_df[output_df['size'] == size][header2]
        plot_df = pd.concat([df1, df2], ignore_index=True)

        winning = [df1['winner'].to_list(), df2['winner'].to_list()]
        ax = axes[index // 2, index % 2]

        g = sns.barplot(
            ax=ax,
            data=plot_df,
            x="starting_at", y="delta_avg", hue="policy",
            errorbar=None, palette="tab10",
            alpha=1
        )
        g.set(xlabel="from prefix len", ylabel="mean future gained KPI", title='size ' + str(size))

        # (1) Horizontal grey grid lines
        ax.yaxis.grid(True, color="lightgrey", linestyle="--", linewidth=0.7, alpha=0.7)
        ax.set_axisbelow(True)

        # Stars for winners
        for i, container in enumerate(g.axes.containers):
            for j, bar in enumerate(container.patches):
                if winning[i][j]:
                    g.plot(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        "*",
                        markersize=10,
                        color="black"
                    )

        # (2) Rename legend entries using policy_labels
        handles, labels = ax.get_legend_handles_labels()
        new_labels = [policy_labels.get(lbl, lbl) for lbl in labels]
        ax.legend(handles, new_labels, title="policy")

    plt.subplots_adjust(
        left=0.1,
        bottom=0.05,
        right=0.9,
        top=0.92,
        wspace=0.2,
        hspace=0.25
    )

    plt.savefig(output_file.replace('_ptest.csv', '_plot.png'))



if __name__ == "__main__":
    # this script performs statistical difference test between different policies and computes the corresponding p-value
    # the analysis is performed at different prefix lengths corresponding to the policy activation
    main_all()