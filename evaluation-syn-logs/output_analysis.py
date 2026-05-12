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

        value_dict = {'folder_name': subfolder_name,
                      'agent_model_name': agent_model_name,
                      'policy_size': policy_size,
                      'policy_type': policy_type,
                      'policy_type_new': policy_type_new,
                      'environment_model_name': environment_model_name,
                      'stochastic_environment': '',
                      'force_environment': '',
                      'total_runs': all_reward_stats['count'],
                      'completed_runs': not_error_reward_stats['count'],
                      'avg_reward': not_error_reward_stats['avg'],
                      'min_reward': not_error_reward_stats['min'],
                      'max_reward': not_error_reward_stats['max'],
                      'median_reward': not_error_reward_stats['median'],
                      'stdev_reward': not_error_reward_stats['stdev'],
                      'avg_reward_all': all_reward_stats['avg'],
                      'min_reward_all': all_reward_stats['min'],
                      'max_reward_all': all_reward_stats['max'],
                      'median_reward_all': all_reward_stats['median'],
                      'stdev_reward_all': all_reward_stats['stdev'],
                      'avg_length': not_error_length_stats['avg'],
                      'min_length': not_error_length_stats['min'],
                      'max_length': not_error_length_stats['max'],
                      'median_length': not_error_length_stats['median'],
                      'stdev_length': not_error_length_stats['stdev'],
                      'avg_length_all': all_length_stats['avg'],
                      'min_length_all': all_length_stats['min'],
                      'max_length_all': all_length_stats['max'],
                      'median_length_all': all_length_stats['median'],
                      'stdev_length_all': all_length_stats['stdev'],
                      'timestamp': ''
                      }

        header_list = value_dict.keys()
        value_list = value_dict.values()

        header = ','.join(header_list)
        result_str = ','.join(value_list)

        # write file  # append
        with open(output_file, 'a+') as file:
            if file.tell() == 0:
                file.write(header + '\n')
            file.write(result_str + '\n')


if __name__ == "__main__":
    # perform an aggregated analysis on the evaluation simulations
    main()