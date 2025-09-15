import glob
import json
import os
import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pm4py import read_xes, write_xes
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from statistics import mean, median, stdev
from kneed import KneeLocator

def create_data_frequency_positional(log, events_set=None, max_trace_length=None, len_log=None, mms=None, testing=False, single_reward=False, all_minmax=False, last_position=True):
    print("Start encoding dataframe")
    # Computing the set of the different events for the columns of the dataframe
    # dict: for each event (key) in the log it contains the average number of occurences in the traces (value)
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
        max_trace_length = int(trace_length_aggr[position_normalization_type])
        # frequency_normalization
        events_set = {e: v[frequency_normalization_type] for e, v in events_set_aggr.items()}
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
                to_add = [i, j + 1] + [d['count']for e, d in events_dict.items()] + [
                    d['last_position'] for e, d in events_dict.items()]
            else:
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
        # Normalizing the reward for the clustering procedure
        if all_minmax:
            data[list(range(2, len(data.transpose())))] = mms.fit_transform(data[list(range(2, len(data.transpose())))])
        else:
            data[len(data.transpose()) - 1] = mms.fit_transform(data[[len(data.transpose()) - 1]])
    else:
        data[len(data.transpose()) - 1] = mms.transform(data[[len(data.transpose()) - 1]])

    if testing:
        return data
    else:
        return data, mms, events_set, max_trace_length, len_log, last_act_to_cluster


def get_event_set(log, event_label):
    # get the activity labels from an event log
    event_set = set()
    for t in log:
        for e in t:
            event_set.add(e[event_label])
    return event_set


def stats(list, zeros=True):
    # computes several statistics for the list
    if not zeros:
        list = [x for x in list if x != 0]
    stats = {'min': min(list), 'max': max(list), 'avg': mean(list), 'median': median(list), 'stdev': stdev(list),
             'Q3': np.percentile(list, 75), 'D9': np.percentile(list, 90), '1': 1}
    return stats

def compute_silhoutte_and_wss(row, kmeans, silhouette_scores, wss, k):
    # Compute silhouette scores and WSS (the paper uses the latter)
    x = row.to_numpy()
    distances = np.sort(kmeans.transform([x]))
    # select the closest 2 clusters: the distance from the closest one is a, the distance from the second one is b
    a, b = distances[0, :2]
    # Simplified Silhouette which used the distances from the cluster centers
    silhouette_scores[k] += (b - a) / max(a, b)
    # Within-Cluster Sum of Squares (WSS scores)
    wss[k] += a


def Silhouette_search(df, output_path, do_plot=False):
    # logging time spent
    t1 = time.time()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t1)))
    print("Number of items: " + str(len(df)))
    np.nan_to_num(df)
    # initialize both silhouette and wss scores dictionaries
    silhouette_scores = dict()
    wss = dict()
    # encoded vectors
    scaled_features = df[[x for x in range(2, len(df.transpose()))]]
    print('Start searching k space:')
    output_file = output_path + frequency_normalization_type + position_normalization_type + "_%d_%d_%d" % (kmin, kmax, step)
    for k in rangex:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(scaled_features)
        silhouette_scores[k] = 0
        wss[k] = 0
        for i in range(iterations):
            print('k: %s, iteration: %s/%s' % (k, i + 1, iterations))
            scaled_features.apply(compute_silhoutte_and_wss, args=(kmeans, silhouette_scores, wss, k), axis=1)
        # normalize by the number of data points and number of iterations (if iterations>1)
        silhouette_scores[k] = silhouette_scores[k] / len(scaled_features) / iterations
        wss[k] = wss[k] / len(scaled_features) / iterations
        with open(output_file + '.txt', 'w') as f:
            f.write("Silhouette: %s\n" % silhouette_scores)
            f.write("WSS: %s\n" % wss)
            t2 = time.time()
            f.write("Execution time: %s\n" % (t2 - t1))

    print('Silhouette:', silhouette_scores)
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t2)))
    print('execution time (sec):', t2 - t1)
    if do_plot:
        # silhouette plot
        s_list = silhouette_scores.items()
        s_list = sorted(s_list)
        x, y = zip(*s_list)
        # plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c='black', s=50)
        plt.clf()
        plt.plot(x, y)
        plt.savefig(output_file + '.png')
        # wss plot
        s_list = wss.items()
        s_list = sorted(s_list)
        x, y = zip(*s_list)
        # plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c='black', s=50)
        plt.clf()
        plt.plot(x, y)
        plt.savefig(output_file.replace('_sil_', '_wss_') + '.png')


def Silhouette_analysis(training_path, output_path, all_minmax, last_position, single_reward, do_plot):
    training_log = read_xes(training_path)
    training_data, *_ = create_data_frequency_positional(training_log, all_minmax=all_minmax, last_position=last_position, single_reward=single_reward)
    Silhouette_search(training_data, output_path, do_plot)



def get_json_dict(line):
    line_dict = {}
    line.replace(' ','').replace('{', ',').replace('}', '')
    items_list = line[1:-1].replace(' ','').split(',')
    for item in items_list:
        key, value = item.split(":")
        line_dict[key] = float(value)
    return line_dict



def knee_analysis(folder_name, frequency_normalization_type, position_normalization_type, kmin, kmax, step):
    # define output paths
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    encoding_type = frequency_normalization_type + position_normalization_type
    path_prefix = os.path.join(ROOT, "cluster_data", folder_name, "silhouette_analysis")
    pattern = "*_sil_%s_%d_%d_%d.txt" % (encoding_type, kmin, kmax, step)
    file_name_list = glob.glob(os.path.join(path_prefix, pattern))
    output_path_sil = os.path.join(path_prefix, "sil_%s_%d_%d_%d_summary.csv" % (encoding_type, kmin, kmax, step))
    output_path_wss = os.path.join(path_prefix, "wss_%s_%d_%d_%d_summary.csv" % (encoding_type, kmin, kmax, step))
    output_path_knee = os.path.join(path_prefix, "knees_%s_%d_%d_%d_summary.csv" % (encoding_type, kmin, kmax, step))

    # initialize dataframes
    output_sil_df = pd.DataFrame()
    output_wss_df = pd.DataFrame()
    output_knee_df = pd.DataFrame()
    for file_name in file_name_list:
        # read first 2 lines of the file
        file = open(file_name, "r")
        line_sil = file.readline()
        line_wss = file.readline()
        file.close()
        # define dictionaries for sil and wss values
        sil_dict = {}
        wss_dict = {}
        knee_dict = {}
        # clean lines from file
        sil_dict['file_name'] = file_name.replace("_sil_%d_%d_%d.txt" % (kmin, kmax, step), "").replace(path_prefix, "")
        wss_dict['file_name'] = sil_dict['file_name']
        knee_dict['file_name'] = sil_dict['file_name']
        line_sil_clean = line_sil.replace('Silhouette: ', '').replace('\n', '')
        line_wss_clean = line_wss.replace('WSS: ', '').replace('\n', '')
        # update the dicts
        sil_dict.update(get_json_dict(line_sil_clean))
        wss_dict.update(get_json_dict(line_wss_clean))
        # define y lists
        y_sil = [sil_dict[str(k)] for k in rangex]
        y_wss = [wss_dict[str(k)] for k in rangex]
        # find knees
        knee_dict['sil_knee1'] = compute_knee(rangex, y_sil, s=1.0, curve="concave", direction="increasing")
        knee_dict['wss_knee1'] = compute_knee(rangex, y_wss, s=1.0, curve="convex", direction="decreasing")
        # create df
        sil_df = pd.DataFrame([sil_dict])
        wss_df = pd.DataFrame([wss_dict])
        knee_df = pd.DataFrame([knee_dict])
        output_sil_df = pd.concat([output_sil_df, sil_df], ignore_index=True)
        output_wss_df = pd.concat([output_wss_df, wss_df], ignore_index=True)
        output_knee_df = pd.concat([output_knee_df, knee_df], ignore_index=True)
        if do_plot:
            # create sil plot
            k_list = ['sil_knee1']
            colors = ["red"]
            plt.clf()
            plt.plot(rangex, y_sil)
            for c, k in zip(colors, k_list):
                plt.vlines(knee_dict[k], min(y_sil) - 0.1, max(y_sil) + 0.1, linestyles="--", colors=c, label=f"{k}")
            plt.legend()
            plt.savefig(file_name.replace('.txt', '.png'))
            # create wss plot
            k_list = ['wss_knee1']
            colors = ["blue"]
            plt.clf()
            plt.plot(rangex, y_wss)
            for c, k in zip(colors, k_list):
                plt.vlines(knee_dict[k], min(y_wss) - 0.1, max(y_wss) + 0.1, linestyles="--", colors=c, label=f"{k}")
            plt.legend()
            plt.savefig(file_name.replace('_sil_', '_wss_').replace('.txt', '.png'))
    # export to csv
    output_sil_df.to_csv(output_path_sil, index=False)
    output_wss_df.to_csv(output_path_wss, index=False)
    output_knee_df.to_csv(output_path_knee, index=False)


def compute_knee(x, y, s, curve, direction):
    kneedle = KneeLocator(x, y, S=s, curve=curve, direction=direction)
    return kneedle.knee

# perform the silhouette, wss and knee analyses
if __name__ == "__main__":

    do_plot = True  # create plots for the silhouette analyses
    do_Silhouette_analysis = False  # perform the silhouette analyses
    do_knee_analysis = True  # do the knee analysis starting from the silhouette and wss analyses

    # the denominator in the normalization of the number of event in the trace, could be avg, median or max or '1'
    # in our paper we selected "max" for both
    frequency_normalization_type = "max"
    position_normalization_type = "max"

    # several settings
    monitor = False  # verbose monitor on the construction of the encoding df
    all_minmax = False
    last_position = True
    clean_encoding = False
    last_activity_importance = 0
    # range of search for cluster numbers
    kmin = 20
    kmax = 300
    step = 20
    rangex = range(kmin, kmax + 1, step)
    # number of clustering iterations for each cluster number
    iterations = 1

    folder_name = 'folder1'

    mode = 'training'

    if do_Silhouette_analysis:
        if mode == 'training':
            single_reward = False
            file_list = ["BPI_2012_log_eng_clean", "BPI_2017_log_strip_newcredit",
                                  "event_log_2000", "event_log_4000",
                                  "event_log_8000", "event_log_16000",
                                  "event_log_rare_2000", "event_log_rare_4000",
                                  "event_log_rare_8000", "event_log_rare_16000",
                         ]
        elif mode == 'testing':
            single_reward = True
            file_list = ["BPI_2012_log_eng_clean", "BPI_2017_log_strip_newcredit",
                         "event_log_2000", "event_log_rare_2000",]
        for file in file_list:
            print("file:", file)
            if mode == 'training':
                file_name = file + "_training_cumulative_rewards"
            elif mode == 'testing':
                file_name = file + "_ForTEST_single_rewards"

            output_file_name = file_name.replace("cumulative_rewards", "sil_").replace("single_rewards", "sil_")

            ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            TRAINING_PATH = os.path.join(ROOT, "data", "logs", folder_name, file_name + ".xes")
            OUTPUT_PATH = os.path.join(ROOT, "cluster_data", folder_name, "silhouette_analysis", output_file_name)
            # perform silhouette analysis
            Silhouette_analysis(TRAINING_PATH, OUTPUT_PATH, all_minmax, last_position, single_reward, do_plot)

    if do_knee_analysis:
        # detect knee on the silhouette and wss graphs
        knee_analysis(folder_name, frequency_normalization_type, position_normalization_type, kmin, kmax, step)


