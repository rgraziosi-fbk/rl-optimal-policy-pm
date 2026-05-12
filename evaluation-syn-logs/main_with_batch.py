""""
this script launches all the experiments
"""
from main import main_explicit_arguments


def main_training():
    """generates the training log using the simulator"""
    folder = "folder1"
    last_position = True

    type = 'synthetic'
    n_trace_list = [2000, 4000, 8000, 16000]
    rare_list = ['', 'rare_']

    starting_at = 0

    for rare in rare_list:
        for n_trace in n_trace_list:
            name_experiments = 'event_log_' + rare + str(n_trace) + '_training'
            print('experiment:', name_experiments)
            path_model = None
            policy = None
            rare_flag = 'rare' in rare
            main_explicit_arguments(type, path_model, name_experiments, n_trace, rare_flag, policy, last_position,
                                    starting_at)



def main_baseline():
    """test the informed heuristic policy using the simulator"""
    type = 'baseline'
    n_trace = 2000
    experiments_list = ['2000', 'rare_2000']
    # experiments_list = ['rare_2000']
    policy = 'baseline'
    starting_at_list = [5, 10, 15, 20, 25, 30]
    # starting_at_list = [0]

    # useless parameters
    path_model = None
    last_position = True

    for e in experiments_list:
        for starting_at in starting_at_list:
            name_experiments = 'sim_eval_log_' + e + '_' + policy + '_' + str(starting_at)
            print('experiment:', name_experiments)
            rare = 'rare' in e
            main_explicit_arguments(type, path_model, name_experiments, n_trace, rare, policy, last_position,
                                    starting_at)


def main_cluster():
    """test the MDP-based policies using the simulator"""
    folder = "folder1"
    last_position = True

    type = 'cluster'
    n_trace = 2000


    # experiments
    experiments_list = ['log_2000_cluster_100', 'log_4000_cluster_100',
                        'log_8000_cluster_100', 'log_16000_cluster_80',
                        'log_rare_2000_cluster_100', 'log_rare_4000_cluster_100',
                        'log_rare_8000_cluster_80', 'log_rare_16000_cluster_100', ]



    policy_list = ['none',  'stepQ2', 'stepQ3', 'expQ1', 'expQ2', 'expQ3']


    starting_at_list = [0]
    # starting_at_list = [5, 10, 15, 20, 25]
    for e in experiments_list:
        for policy in policy_list:
            for starting_at in starting_at_list:
                name_experiments = 'sim_eval_' + e + '_' + policy + '_' + str(starting_at)
                print('experiment:', name_experiments)
                path_model = 'cluster_models/' + folder + '/' + e.replace('_cluster', '') + '/'
                rare = 'rare' in e
                main_explicit_arguments(type, path_model, name_experiments, n_trace, rare, policy, last_position,
                                        starting_at)

def main_cql():
    """test the CQL-based policies using the simulator"""
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
    starting_at_list = [0, 5, 10, 15, 20, 25]
    for e in cql_experiments_list:
        for starting_at in starting_at_list:
            name_experiments = 'sim_eval_' + e.replace('_e100', '_cql_100') + '_' + str(starting_at)
            print('experiment:', name_experiments)
            path_model = 'cql_recommender/output_data/' + e
            rare = 'rare' in e
            main_explicit_arguments(type, path_model, name_experiments, n_trace, rare, policy, last_position,
                                    starting_at)


if __name__ == "__main__":
    # main_training()
    # main_baseline()
    main_cluster()
    # main_cql()


