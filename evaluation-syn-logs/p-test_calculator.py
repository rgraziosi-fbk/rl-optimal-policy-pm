import glob
import os

import numpy as np
import pandas as pd
from scipy import stats


def get_file_name(folder_name, subfolder_name, size, policy):
    file_name_rule = "reward_sim_eval_log*_" + str(size) + "*" + policy + ".csv"
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

def main():
    folder_name = "folder1"
    subfolder_name = "sim_event_log"
    # size list
    size_list = [2000, 4000, 8000, 16000]
    # policy list
    policy_type_list = ['none', 'stepQ1', 'stepQ2', 'stepQ3', 'expQ1', 'expQ2', 'expQ3', 'cql_50', 'cql_100']
    number_of_policies = len(policy_type_list)
    # define output file name
    output_file = os.path.join("output", folder_name, subfolder_name, subfolder_name + "_ptest.csv")
    # define dataframe
    header = ['folder_name', 'subfolder_name', 'size', 'policy1', 'policy2', 'count1', 'avg1', 'stdev1', 'count2', 'avg2', 'stdev2', 'pvalue']
    output_df = pd.DataFrame(columns=header)

    for size in size_list:
        for p1 in range(number_of_policies):
            policy1 = policy_type_list[p1]
            file1 = get_file_name(folder_name, subfolder_name, size, policy1)
            reward_list1, count1, avg1, stdev1 = get_reward_list(file1)
            for p2 in range(p1+1, number_of_policies):
                policy2 = policy_type_list[p2]
                file2 = get_file_name(folder_name, subfolder_name, size, policy2)
                reward_list2, count2, avg2, stdev2 = get_reward_list(file2)
                statistic, pvalue = stats.ttest_ind(reward_list1, reward_list2, equal_var=False)
                # write result
                value_list = [folder_name, subfolder_name, size, policy1, policy2,
                              count1, avg1, stdev1, count2, avg2, stdev2, pvalue]
                new_row = pd.DataFrame(np.array(value_list).reshape(1,12), columns=header)
                output_df = pd.concat([output_df, new_row], ignore_index=True)              # write file  # append
    output_df.to_csv(output_file, index=False)





if __name__ == "__main__":
    # this script performs statistical difference test between different policies and computes the corresponding p-value
    main()