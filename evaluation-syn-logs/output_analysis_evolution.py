import glob
import os

from statistics import mean, median, stdev

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def compute_stats(alist):
    stats = {}
    stats['count'] = len(alist)
    stats['sum'] = sum(alist)
    if stats['count'] > 0:
        stats['min'] = max(alist)
        stats['max'] = max(alist)
        stats['avg'] = mean(alist)
        stats['median'] = median(alist)
        if stats['count'] > 1:
            stats['stdev'] = stdev(alist)
        else:
            stats['stdev'] = 0
    else:
        stats['min'] = 0
        stats['max'] = 0
        stats['avg'] = 0
        stats['median'] = 0
        stats['stdev'] = 0
    stats = {k: str(v) for k, v in stats.items()}
    return stats

def main():
    # policy_type_list = ['syn', 'none', 'step4', 'exp4']
    # folder_name = "sim_event_log"
    # file_name_list = glob.glob("output/" + folder_name + "/reward*.csv")

    policy_type_list = ['cluster_*_expQ2', 'cql_100']
    policy_size_list = ['2000', '4000', '8000', '16000']
    starting_at_list = [0, 1, 5, 10, 15, 20, 25, 30]
    folder_name = "folder1"
    subfolder_name = "sim_log_rare_evolution"
    folder_path = os.path.join("output", folder_name, subfolder_name)
    # file_name_list = glob.glob(os.path.join(folder_path, "reward*.csv"))

    file_name_list = []
    for policy_type in policy_type_list:
        file_name_list += list(glob.glob(os.path.join(folder_path, "reward*" + policy_type + "*.csv")))
    file_name_list = [file_name[file_name.rfind('\\')+1:] for file_name in file_name_list]

    output_file = os.path.join("output", folder_name, subfolder_name, subfolder_name + "_results.csv")

    boxplot_df = pd.DataFrame()
    for file_name in file_name_list:

        file = os.path.join("output", folder_name, subfolder_name, file_name)

        df = pd.read_csv(file, sep=';')

        agent_policy_name_list = file_name.replace('reward_sim_eval_log_', '').replace('rare_', '').replace(".csv", "").split("_")
        policy_size = agent_policy_name_list[0]
        agent_model_name = "_".join(agent_policy_name_list[:-1])
        policy_type = "_".join(agent_policy_name_list[1:-1])
        policy_type_new = policy_type[:policy_type.find("_")] + policy_type[policy_type.rfind("_"):] if "cluster" in policy_type else policy_type
        agent_starting_at = agent_policy_name_list[-1]
        environment_model_name = "original_simulator"

        all_reward_list = df['reward'].to_list()
        all_length_list = df['len_trace'].to_list()

        not_error_df = df.loc[~df['traces'].str.contains('ERRORE')]  # ~ is the invert operator!
        not_error_reward_list = not_error_df['reward'].to_list()
        not_error_length_list = not_error_df['len_trace'].to_list()

        if policy_size in policy_size_list and int(agent_starting_at) in starting_at_list:
            # and policy_type in policy_type_list
            tmp_df = not_error_df['reward'].to_frame()
            tmp_df['policy_size'] = policy_size
            tmp_df['policy_type'] = policy_type
            tmp_df['starting_at'] = int(agent_starting_at)
            boxplot_df = pd.concat([boxplot_df, tmp_df])

        all_reward_stats = compute_stats(all_reward_list)
        not_error_reward_stats = compute_stats(not_error_reward_list)
        all_length_stats = compute_stats(all_length_list)
        not_error_length_stats = compute_stats(not_error_length_list)

        header = 'folder_name,agent_model_name,policy_size,policy_type,policy_type_new,agent_starting_at,environment_model_name,stochastic_environment,force_environment,total_runs,completed_runs,avg_reward,min_reward,max_reward,median_reward,stdev_reward,avg_reward_all,min_reward_all,max_reward_all,median_reward_all,stdev_reward_all,avg_length,min_length,max_length,median_length,stdev_length,avg_length_all,min_length_all,max_length_all,median_length_all,stdev_length_all,timestamp'
        value_list = [subfolder_name, agent_model_name, policy_size, policy_type, policy_type_new, agent_starting_at, environment_model_name, '', '', all_reward_stats['count'], not_error_reward_stats['count'],
                 not_error_reward_stats['avg'], not_error_reward_stats['min'], not_error_reward_stats['max'], not_error_reward_stats['median'], not_error_reward_stats['stdev'],
                 all_reward_stats['avg'], all_reward_stats['min'], all_reward_stats['max'], all_reward_stats['median'], all_reward_stats['stdev'],
                 not_error_length_stats['avg'], not_error_length_stats['min'], not_error_length_stats['max'], not_error_length_stats['median'], not_error_length_stats['stdev'],
                 all_length_stats['avg'], all_length_stats['min'], all_length_stats['max'], all_length_stats['median'], all_length_stats['stdev'],
                 '']
        result_str = ','.join(value_list)

        # write file  # append
        with open(output_file, 'a+') as file:
            if file.tell() == 0:
                file.write(header + '\n')
            file.write(result_str + '\n')

    do_plots_sns(boxplot_df, folder_path)
    print('')


def do_plots_pyplot(boxplot_df, folder_path):
    policy_size_list = boxplot_df["policy_size"].unique()
    policy_type_list = boxplot_df["policy_type"].unique()
    fig, axs = plt.subplots(nrows=len(policy_size_list), ncols=len(policy_type_list), figsize=(6, 6), sharey=True)
    for i, size in enumerate(policy_size_list):
        for j, policy_type in enumerate(policy_type_list):
            tmp_df = boxplot_df[(boxplot_df['policy_size'] == size) & (boxplot_df['policy_type'] == policy_type)]
            tmp_df = tmp_df.pivot(columns='starting_at', values='reward')
            if len(policy_size_list) == 1:
                # axs[j].boxplot(tmp_df, labels=tmp_df.columns, showmeans=True, meanline=True)
                set_axis_style(axs[j], labels=tmp_df.columns)
                parts = axs[j].violinplot(tmp_df, showmeans=True, showmedians=True)
                parts['cmeans'].set_edgecolor('black')
                if j == 0:  # tmp: should work for the moment
                    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
                        parts[partname].set_edgecolor('#3D85C6')
                    for piece in parts['bodies']:
                        piece.set_facecolor('#6FA8DC')
                else:
                    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
                        parts[partname].set_edgecolor('#E69138')
                    for piece in parts['bodies']:
                        piece.set_facecolor('#F6B26B')  # cql

                    if j == 0:
                        piece.set_facecolor('#6FA8DC')  # cluster
                    else:
                        piece.set_facecolor('#F6B26B')  # cql
                axs[j].set_title(size + ' - ' + policy_type, fontsize=10)


            else:
                # axs[i, j].boxplot(tmp_df, labels=tmp_df.columns, showmeans=True, meanline=True)
                axs[i, j].violinplot(tmp_df, labels=tmp_df.columns, showmeans=True, showmedians=False)
                axs[i, j].set_title(size + ' - ' + policy_type, fontsize=10)

    plt.show()

def do_plots_sns(boxplot_df, folder_path):
    policy_size_list = boxplot_df["policy_size"].unique()
    policy_type_list = boxplot_df["policy_type"].unique()
    for size in policy_size_list:
        sns.set(rc={'figure.figsize': (10, 7)})
        sub_df = boxplot_df[boxplot_df["policy_size"] == size]
        ax1 = sns.violinplot(sub_df, x="starting_at", y="reward", hue="policy_type", inner="box", density_norm="count")
        ax1.axes.set_title('Violin plot: ' + size, fontsize=20)
        sns.move_legend(ax1, "lower right", bbox_to_anchor=(1, 1))
        ax1.get_figure().savefig(os.path.join(folder_path, "violinplot_" + size + ".png"))
        plt.clf()
        ax2 = sns.boxplot(sub_df, x="starting_at", y="reward", hue="policy_type", medianprops={"color": "r", "linewidth": 1}, showmeans=True , meanprops={'marker':'o', 'markerfacecolor':'white', 'markeredgecolor':'black', 'markersize':'8'}, flierprops={"marker": "x"})
        ax2.axes.set_title('Box plot: ' + size, fontsize=20)
        sns.move_legend(ax2, "lower right", bbox_to_anchor=(1, 1))
        ax2.get_figure().savefig(os.path.join(folder_path, "boxplot_" + size + ".png"))
        plt.clf()


def set_axis_style(ax, labels):
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Sample name')


if __name__ == "__main__":
    # perform an aggregated analysis on the evaluation simulations
    # the analysis considers policy being activated at different prefix lenghts
    main()