import os
import time
import pickle
import numpy as np
import pandas as pd
from pm4py import read_xes, write_xes
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from statistics import mean, median, stdev



def create_data_frequency_positional(log, path_variables_dict, encoding_type_dict, single_reward, monitor, events_set=None, max_trace_length=None, len_log=None, mms=None, testing=False):
    all_minmax =  encoding_type_dict["all_minmax"]  # default False
    last_position = encoding_type_dict["last_position"]  # default True
    print("Start encoding dataframe")
    # Computing the set of the different events for the columns of the dataframe
    # dict: for each event (key) in the log it contains the average number of occurence in the traces (value)
    if not testing:
        events_set_list = dict()
        trace_length = []
        len_log = len(log)
        for i, trace in enumerate(log):
            if monitor:
                print("Computing trace stats: trace %s/%s" % (i+1, len_log))
            trace_event_count = dict()
            trace_length += [len(trace)]
            for j, e in enumerate(trace):
                event_name = e["concept:name"]
                if event_name not in events_set_list.keys():
                    events_set_list[event_name] = []
                if event_name not in trace_event_count.keys():
                    trace_event_count[event_name] = 1
                else:
                    trace_event_count[event_name] += 1
            for event_name, count in trace_event_count.items():
                events_set_list[event_name] += [count]

        # compute trace stats
        trace_length_aggr = stats(trace_length)
        events_set_aggr = {e: stats(v) for e, v in events_set_list.items()}
        if monitor:
            print('Events statistics:')
            print('trace length: ', trace_length_aggr)
            for item in events_set_aggr.items():
                print(item)
        # position_normalization
        max_trace_length = int(trace_length_aggr[encoding_type_dict["position_normalization_type"]])
        # frequency_normalization
        events_set = {e: v[encoding_type_dict["frequency_normalization_type"]] for e, v in events_set_aggr.items()}
        # this is a further normalization intra events that could be ignored
        events_set = {e: max(list(events_set.values())) for e in events_set.keys()}

    last_act_to_cluster = {x: set() for x in events_set}
    data_matrix = list()
    for i, trace in enumerate(log):
        trace_id = trace.attributes['concept:name']
        if monitor:
            print("Computing df: trace %s/%s" % (i + 1, len_log))
        # Initializing a dictionary to store the number of occurrences for each event
        events_dict = {e: {'count': 0, 'last_position': 0} for e in events_set}
        events_dict["START"]["count"] = 1
        events_dict["START"]["last_position"] = 1
        for j, event in enumerate(trace[1:]):
            if all_minmax:
                # normalization happens later with minmaxscaler
                to_add = [i, j + 1] + [d['count']for e, d in events_dict.items()] + [
                    d['last_position'] for e, d in events_dict.items()]
            else:
                # normalization used in the paper
                to_add = [i, j+1] + [float(d['count'] / events_set[e]) for e, d in events_dict.items()] + [
                    float(d['last_position'] / max_trace_length) for e, d in events_dict.items()]
            if event["concept:name"] == "END" or single_reward:
                to_add.append(event["kpi:reward"])
            else:
                to_add.append(0.0)
            data_matrix.append(to_add)
            # Updating the dictionary
            events_dict[event["concept:name"]]['count'] += 1
            if last_position:
                events_dict[event["concept:name"]]['last_position'] = j + 2
            elif events_dict[event["concept:name"]]['last_position'] == 0:
                # in this case is the first position and not the last position
                events_dict[event["concept:name"]]['last_position'] = j + 2
    data = pd.DataFrame(data_matrix)
    # Initializing a MinMaxScaler for the continuous column: reward
    if not testing:
        mms = MinMaxScaler(feature_range=(0, 1))
        # Normalizing the reward for the clustering procedure!
        if all_minmax:
            data[list(range(2, len(data.transpose())))] = mms.fit_transform(data[list(range(2, len(data.transpose())))])
        else:
            data[len(data.transpose()) - 1] = mms.fit_transform(data[[len(data.transpose()) - 1]])
        file=open(path_variables_dict["PICKLE_SAVE"],'wb')
        pickle.dump(events_set, file)
        pickle.dump(max_trace_length, file)
        pickle.dump(mms, file)
        file.close()
    else:
        data[len(data.transpose()) - 1] = mms.transform(data[[len(data.transpose()) - 1]])

    if testing:
        return data
    else:
        return data, mms, events_set, max_trace_length, len_log, last_act_to_cluster


def stats(list):
   stdev_v = stdev(list) if len(list) > 1 else np.nan
   stats = {'max': max(list), 'avg': mean(list), 'median': median(list), 'stdev': stdev_v,
                         'Q3': np.percentile(list, 75), 'D9': np.percentile(list, 90)}
   return stats


def cluster(df, log, mms, last_a_to_cluster, path_variables_dict, encoding_type_dict, n_clusters):
    print("Number of items: " + str(len(df)))
    np.nan_to_num(df)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(df[[x for x in range(2, len(df.transpose()))]])
    print("Finished Clustering, now starting exporting results")
    df.apply(iterTrainingDf, args=(log, last_a_to_cluster, mms, kmeans, encoding_type_dict), axis=1)
    file = open(path_variables_dict["PICKLE_SAVE"], 'ab')
    pickle.dump(last_a_to_cluster, file)
    file.close()
    write_xes(log, path_variables_dict["PATH_CLUSTERED_TRAINING"])
    print("Finished writing training log")
    pickle.dump(kmeans, open(path_variables_dict["MODEL_CLUSTER_SAVE"], 'wb'), protocol=4)

    return kmeans


def iterTrainingDf(row, log, last_a_to_cluster, mms, kmeans, encoding_type_dict):
    # row is the encoded prefix, where first and second entries are trace and event indeces, then there is the normalized frequency encoding and the last entry should be cumulated reward
    event = log[int(row[0])][int(row[1])]
    if event["concept:name"] == "A_PREACCEPTED":  #debug??
        pass
    label = kmeans.labels_[row.name]  # row.name should be indentifier of the prefix in the log, thery are 1:1 relations with the labels_
    event["cluster:prefix"] = label
    event["stato"] = event["concept:name"] + " | " + str(label)
    if encoding_type_dict["all_minmax"]:
        event["cluster:reward"] = mms.inverse_transform([kmeans.cluster_centers_[label]])[0][-1]
    else:
        event["cluster:reward"] = mms.inverse_transform([[kmeans.cluster_centers_[label][-1]]])[0][0]
    last_a_to_cluster[event["concept:name"]].add(label)  # this should be a dict where for every cluster is given the set of the activities that are last activities for all the prefixes in that cluster


def iterTestingDf(row, log, last_a_to_cluster, mms, kmean, labels):
    event = log[int(row[0])][int(row[1])]
    best_cluster_value, best_cluster = min([(labels[row.name][x], x) for x in last_a_to_cluster[event["concept:name"]]],
                                           key=lambda x: x[0])
    event["cluster:prefix"] = best_cluster
    event["stato"] = event["concept:name"] + " | " + str(best_cluster)
    event["cluster:reward"] = mms.inverse_transform([[kmean.cluster_centers_[best_cluster][-1]]])[0][0]



def Clusterer_main(mode, path_variables_dict, encoding_type_dict, n_clusters, single_reward, monitor):
    # this part creates the clustering for training the agent
    if mode == 'training':
        training_log = read_xes(path_variables_dict["TRAINING_PATH"])
        training_data, training_mms, eventsset, avgtracelength, lenlog, last_act_to_cluster = create_data_frequency_positional(training_log, path_variables_dict, encoding_type_dict, single_reward, monitor)
        kmeans = cluster(training_data, training_log, training_mms, last_act_to_cluster, path_variables_dict, encoding_type_dict, n_clusters)
    # this part creates the clustering for evaluating the policy using a test set MDP
    # this generates a different test set with different clusters number w.r.t. training
    elif mode == 'testing':
        testing_log = read_xes(path_variables_dict["TESTING_PATH"])
        testing_data, testing_mms, eventsset, avgtracelength, lenlog, last_act_to_cluster = create_data_frequency_positional(testing_log, path_variables_dict, encoding_type_dict, single_reward, monitor)
        kmeans = cluster(testing_data, testing_log, testing_mms, last_act_to_cluster, path_variables_dict, encoding_type_dict, n_clusters)
    else:
        raise Exception("Error: mode not correct!")

def Path_variables(folder_name, file):
    # define project root relative to this file
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    path_variables_dict = {}

    path_variables_dict["TRAINING_PATH"] = os.path.join(ROOT, "data", "logs", folder_name, f"{file}_training.xes")
    path_variables_dict["PICKLE_PATH"]   = os.path.join(ROOT, "data", "logs", folder_name, f"{file}.pkl")
    # add more paths as needed

    return path_variables_dict


def Path_variables(folder_name, file, n_clusters, mode):
    # define a dict with path variables where to save output log and pickle files
    path_variables_dict = dict()
    # find project root (2 levels up from this file: src/subsrc1 → src → project_root)
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if mode == 'training':
        path_variables_dict["TRAINING_PATH"] = os.path.join(ROOT, "data", "logs", folder_name, file + "_training_cumulative_rewards.xes")
        path_variables_dict["TESTING_PATH"] = "not_used"
        path_variables_dict["PATH_CLUSTERED_TRAINING"] = os.path.join(ROOT, "cluster_data", folder_name, "output_logs", file + "_" + str(
            n_clusters) + "_training.xes")
        path_variables_dict["PATH_CLUSTERED_TESTING"] = "not_used"
        path_variables_dict["MODEL_CLUSTER_SAVE"] = os.path.join(ROOT, "cluster_data", folder_name, "output_logs", file + "_" + str(
            n_clusters) + "_training_model.pkl")
        path_variables_dict["PICKLE_SAVE"] = os.path.join(ROOT, "cluster_data", folder_name, "output_logs", file + "_" + str(
            n_clusters) + "_training_pickle.pkl")
        single_reward = False
    elif mode == 'testing':
        # for testing we use non-cumulative reward, this provides more fair evaluation
        path_variables_dict["TRAINING_PATH"] = "not_used"
        path_variables_dict["TESTING_PATH"] = os.path.join(ROOT, "data", "logs", folder_name, file + "_ForTEST_single_rewards.xes")
        path_variables_dict["PATH_CLUSTERED_TRAINING"] = os.path.join(ROOT, "cluster_data", folder_name, "output_logs", file + "_" + str(
            n_clusters) + "_ForTEST.xes")  # this name is correct, the variable is called TRAINING to reuse already written code
        path_variables_dict["PATH_CLUSTERED_TESTING"] = "not used"
        path_variables_dict["MODEL_CLUSTER_SAVE"] = os.path.join(ROOT, "cluster_data", folder_name, "output_logs", file + "_" + str(
            n_clusters) + "_ForTEST_model.pkl")
        path_variables_dict["PICKLE_SAVE"] = os.path.join(ROOT, "cluster_data", folder_name, "output_logs", file + "_" + str(
            n_clusters) + "_ForTEST_pickle.pkl")
        single_reward = True
    else:
        raise Exception("Error: mode not correct!")
    return path_variables_dict, single_reward


def main():
    # script to apply clustering to all the event log considered in the paper
    folder_name = "folder1"

    mode = 'training'

    # the denominator in the normalization of the number of event in the trace, could be avg, median or max
    # in the paper we have selected "max", "max"
    encoding_type_dict = {
        "all_minmax" : False,
        "last_position" : True,
        "frequency_normalization_type" : "max",
        "position_normalization_type" : "max",}
    monitor = False

    training_file_list = [('BPI_2012_log_eng_clean', '140'),
                          ('BPI_2017_log_strip_newcredit', '100'),
                          ('event_log_2000', '100'),
                          ('event_log_4000', '140'),
                          ('event_log_8000', '120'),
                          ('event_log_16000', '120'),
                          ('event_log_rare_2000', '120'),
                          ('event_log_rare_4000', '120'),
                          ('event_log_rare_8000', '100'),
                          ('event_log_rare_16000', '100')
                          ]
    training_file_list = [('event_log_2000', '100')]
    test_file_list = [('BPI_2012_log_eng_clean', '100'),
                      ('BPI_2017_log_strip_newcredit', '100'),
                      ('event_log_2000', '100'),
                      ('event_log_rare_2000', '120'),
                      ]

    if mode == 'training':
        file_list = training_file_list
    elif mode == 'testing':
        file_list = test_file_list
    else:
        raise Exception("Error: mode not correct!")

    for file, n_clusters in file_list:
        n_clusters = int(n_clusters)
        print(file)

        path_variables_dict, single_reward = Path_variables(folder_name, file, n_clusters, mode)

        t1 = time.time()
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t1)))

        Clusterer_main(mode, path_variables_dict, encoding_type_dict, n_clusters, single_reward, monitor)

        t2 = time.time()
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t2)))
        print(t2 - t1)


if __name__ == "__main__":
    main()

