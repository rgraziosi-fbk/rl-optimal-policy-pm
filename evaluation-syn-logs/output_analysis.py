import glob
import os

from statistics import mean, median, stdev
import pandas as pd


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
    policy_type_list = ['syn', 'none', 'stepQ1', 'stepQ2', 'stepQ3', 'expQ1', 'expQ2', 'expQ3', 'cql_50', 'cql_100']
    folder_name = "folder2"
    subfolder_name = "sim_event_log"
    file_name_list = glob.glob(os.path.join("output", folder_name, subfolder_name, "reward*.csv"))

    file_name_list = [file_name.replace(os.path.join("output", folder_name, subfolder_name, ""), "") for file_name in file_name_list if any(policy_type in file_name for policy_type in policy_type_list)]

    output_file = os.path.join("output", folder_name, subfolder_name, subfolder_name + "_results.csv")

    for file_name in file_name_list:

        file = os.path.join("output", folder_name, subfolder_name, file_name)

        df = pd.read_csv(file, sep=';')

        agent_policy_name_list =  file_name.replace('reward_sim_eval_log_', '').replace('rare_', '').replace(".csv", "").split("_")
        policy_size = agent_policy_name_list[0]
        agent_model_name = "_".join(agent_policy_name_list)
        policy_type = "_".join(agent_policy_name_list[1:])
        policy_type_new = policy_type[:policy_type.find("_")] + policy_type[policy_type.rfind("_"):] if "cluster" in policy_type else policy_type
        environment_model_name = "original_simulator"

        all_reward_list = df['reward'].to_list()
        all_length_list = df['len_trace'].to_list()

        not_error_df = df.loc[~df['traces'].str.contains('ERRORE')]  # ~ is the invert operator!
        not_error_reward_list = not_error_df['reward'].to_list()
        not_error_length_list = not_error_df['len_trace'].to_list()

        all_reward_stats = compute_stats(all_reward_list)
        not_error_reward_stats = compute_stats(not_error_reward_list)
        all_length_stats = compute_stats(all_length_list)
        not_error_length_stats = compute_stats(not_error_length_list)

        header = 'folder_name,agent_model_name,policy_size,policy_type,policy_type_new,environment_model_name,stochastic_environment,force_environment,total_runs,completed_runs,avg_reward,min_reward,max_reward,median_reward,stdev_reward,avg_reward_all,min_reward_all,max_reward_all,median_reward_all,stdev_reward_all,avg_length,min_length,max_length,median_length,stdev_length,avg_length_all,min_length_all,max_length_all,median_length_all,stdev_length_all,timestamp'
        value_list = [subfolder_name, agent_model_name, policy_size, policy_type, policy_type_new, environment_model_name, '', '', all_reward_stats['count'], not_error_reward_stats['count'],
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


if __name__ == "__main__":
    # perform an aggregated analysis on the evaluation simulations
    main()