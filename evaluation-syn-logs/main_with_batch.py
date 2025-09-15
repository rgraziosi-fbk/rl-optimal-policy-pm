""""
this script launches all the experiments
"""
from main import main_explicit_arguments

def main_cluster():
    folder = "folder1"
    last_position = True

    type = 'cluster'
    n_trace = 2000

    cluster_experiments_list = ['log_2000_cluster_100', 'log_4000_cluster_140',
                                'log_8000_cluster_120', 'log_16000_cluster_120',
                                'log_rare_2000_cluster_120', 'log_rare_4000_cluster_120',
                                'log_rare_8000_cluster_100', 'log_rare_16000_cluster_100', ]
    policy_list = ['none', 'stepQ1', 'stepQ2', 'stepQ3', 'expQ1', 'expQ2', 'expQ3']
    # starting point defines after how many activities performed under the standard simulation model the policy is activated
    starting_at_list = [0, 5, 10, 15, 20, 25, 30]
    for e in cluster_experiments_list:
        for policy in policy_list:
            for starting_at in starting_at_list:
                name_experiments = 'sim_eval_' + e + '_' + policy + '_' + str(starting_at)
                print('experiment:', name_experiments)
                path_model = 'cluster_models/' + folder + '/' + e.replace('_cluster', '') + '/'
                rare = 'rare' in e
                main_explicit_arguments(type, path_model, name_experiments, n_trace, rare, policy, last_position,
                                        starting_at)

def main_cql():
    last_position = True

    type = 'cql'
    n_trace = 2000

    cql_experiments_list = ['log_2000_e100', 'log_rare_2000_e100',
                            'log_4000_e100', 'log_rare_4000_e100',
                            'log_8000_e100', 'log_rare_8000_e100',
                            'log_16000_e100', 'log_rare_16000_e100',
                            'log_2000_e50', 'log_rare_2000_e50',
                            'log_4000_e50', 'log_rare_4000_e50',
                            'log_8000_e50', 'log_rare_8000_e50',
                            'log_16000_e50', 'log_rare_16000_e50',]
    policy = None

    # starting point defines after how many activities performed under the standard simulation model the policy is activated
    starting_at_list = [0, 5, 10, 15, 20, 25, 30]
    for e in cql_experiments_list:
        for starting_at in starting_at_list:
            name_experiments = 'sim_eval_' + e.replace('_e100', '_cql_100') + '_' + str(starting_at)
            print('experiment:', name_experiments)
            path_model = 'cql_recommender/output_data/' + e
            rare = 'rare' in e
            main_explicit_arguments(type, path_model, name_experiments, n_trace, rare, policy, last_position,
                                    starting_at)


if __name__ == "__main__":
    main_cluster()
    # main_cql()

