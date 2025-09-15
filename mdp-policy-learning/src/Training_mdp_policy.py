import time
from clusterer.Clusterer_splitted_without_last_act import Path_variables, Clusterer_main
from mdp_creator.MDPCreator_BPI_cluster import mdp_creator_main
from training.DP_methods import DP_training_main

# contains the full pipeline to construct the MDP and train the policy with Dynamic Programming methods
if __name__ == "__main__":

    t1 = time.time()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t1)))

    # best cluster number from wss analysis
    best_n_clusters_dict = {'BPI_2012_log_eng_clean': 140,
                          'BPI_2017_log_strip_newcredit': 100,
                          'event_log_2000': 100,
                          'event_log_4000': 140,
                          'event_log_8000': 120,
                          'event_log_16000': 120,
                          'event_log_rare_2000': 120,
                          'event_log_rare_4000': 120,
                          'event_log_rare_8000': 100,
                          'event_log_rare_16000': 100}

    # define working parameters
    folder_name = "folder1"
    file = 'event_log_2000'
    scale_factor = "expQ2" # select the scale factor to use in the DP training between ["none", "expQ1", "expQ2", "expQ3", "stepQ1", "stepQ2", "stepQ3",]

    # automatically select the best cluster number derived in the wss analysis
    n_clusters = best_n_clusters_dict[file]

    mode = 'training'
    print("file: %s, cluster_number: %d" % (file, n_clusters))

    # select the scale factor to use in the DP training between ["none", "expQ1", "expQ2", "expQ3", "stepQ1", "stepQ2", "stepQ3",]
    scale_factor = "expQ2"

    # define RL parameters
    threshold = 0.00001  # threshold at which algorithm stops
    gamma = 1  # discount rate, discount near to 1 avoids loops
    normalize_reward = True  # minmaxscaler is used in reward
    change_zero_reward = False  # if minmaxscaler on reward is used, apply also to zero reward?


    print("\nPhase 1: prefix clustering\n")

    # the denominator in the normalization of the number of event in the trace, could be avg, median or max
    # in the paper we have selected "max", "max"
    encoding_type_dict = {
        "all_minmax" : False,
        "last_position" : True,
        "frequency_normalization_type" : "max",
        "position_normalization_type" : "max",}
    monitor = False

    # define path variables used in the clusterer method
    path_variables_dict, single_reward = Path_variables(folder_name, file, n_clusters, mode)

    # cluster prefixes and export the annotaded log in ./cluster_data/folder1/output_logs/ together with pkl files
    Clusterer_main(mode, path_variables_dict, encoding_type_dict, n_clusters, single_reward, monitor)

    print("\nPhase 2: MDP creation\n")

    # create the mdp and export it in ./cluster_data/folder1/output_mdps/ (also export a preprocessed event log in ./cluster_data/folder1/output_logs/ with suffix "_preprocessed_wloops"
    mdp_creator_main(folder_name, mode, file, n_clusters)

    print("\nPhase 3: policy learning with DP\n")
    # Learn the optimal policy with DP method and export it in ./cluster_data/folder1/output_policies/

    # start the DP training pipeline to learn the optimal policy
    file_name = file + "_" + str(n_clusters)
    states_list, state_action_dict, q_table, policy_Q = DP_training_main(folder_name, file_name, scale_factor, threshold, gamma, normalize_reward, change_zero_reward)

    t2 = time.time()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t2)))
    print("Total execution time:", t2 - t1)


