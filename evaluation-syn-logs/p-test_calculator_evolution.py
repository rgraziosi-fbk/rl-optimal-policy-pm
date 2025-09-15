import glob
import os

import numpy as np
import pandas as pd
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

def get_reward_list(file_name):
    df = pd.read_csv(file_name, sep=';')
    not_error_df = df.loc[~df['traces'].str.contains('ERRORE')]
    reward_list = not_error_df['reward']
    count = len(not_error_df)
    avg = reward_list.mean()
    stdev = reward_list.std()
    return reward_list, count, avg, stdev

def define_winners(df):
    # tie is NOT consider winning
    df['winner1'] = np.where(((df['avg1'] > df['avg2']) & (df['pvalue'] < 0.05)), True, False)
    df['winner2'] = np.where(((df['avg1'] < df['avg2']) & (df['pvalue'] < 0.05)), True, False)
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
    policy_type_list = ['expQ2', 'cql_100']  # compare only two policies
    # starting_at list
    staring_at_list = [0, 5, 10, 15, 20, 25, 30]
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
            reward_list1, count1, avg1, stdev1 = get_reward_list(file1)
            reward_list2, count2, avg2, stdev2 = get_reward_list(file2)
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


def create_plot(output_df, size_list, output_file):
    plot_title = 'Synthetic Event Logs Rare' if '_rare_' in output_file else 'Synthetic Event Logs'
    header_plot = ['folder_name', 'subfolder_name', 'size', 'starting_at', 'policy', 'avg',
              'stdev', 'winner']
    header1 = ['folder_name', 'subfolder_name', 'size', 'starting_at', 'policy1', 'avg1',
              'stdev1', 'winner1']
    header2 = ['folder_name', 'subfolder_name', 'size', 'starting_at', 'policy2', 'avg2',
              'stdev2', 'winner2']
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    fig.suptitle(plot_title)
    # fig.tight_layout(pad=5.0)
    for index, size in enumerate(size_list):
        df1 = pd.DataFrame()
        df2 = pd.DataFrame()
        df1[header_plot] = output_df[output_df['size'] == size][header1]
        df2[header_plot] = output_df[output_df['size'] == size][header2]
        plot_df = pd.concat([df1, df2], ignore_index=True)
        winning = [df1['winner'].to_list(), df2['winner'].to_list()]
        g = sns.barplot(
            ax=axes[index // 2, index % 2],
            data=plot_df,
            x="starting_at", y="avg", hue="policy",
            errorbar=None, palette="tab10",
            alpha=1
        )
        g.set(xlabel="from prefix len", ylabel="mean future reward", title='size ' + str(size) )
        # Iterating over the bars one-by-one
        for i, container in enumerate(g.axes.containers):
            for j, bar in enumerate(container.patches):
                if winning[i][j]:
                    g.plot(bar.get_x() + bar.get_width() / 2, bar.get_height(), "*",
                           markersize=10, color="black")
    # plt.subplot_tool()  # to set the adjustment
    plt.subplots_adjust(left=0.1,
                        bottom=0.05,
                        right=0.9,
                        top=0.92,
                        wspace=0.2,
                        hspace=0.25)
    # plt.show()
    plt.savefig(output_file.replace('_ptest.csv', '_plot.png'))


if __name__ == "__main__":
    # this script performs statistical difference test between different policies and computes the corresponding p-value
    # the analysis is performed at different prefix lengths corresponding to the policy activation
    main()