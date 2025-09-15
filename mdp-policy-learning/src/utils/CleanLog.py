import pandas as pd
import copy
import pm4py
import functools
from itertools import combinations
from pm4py.objects.log.importer.xes import importer as xes_importer
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from pm4py.objects.log.util.sampling import sample_log

import kosaraju
from pymining import seqmining
from pm4py.objects.log.obj import EventLog, Trace
from pm4py.algo.filtering.log.variants import variants_filter

environment_action = ["O_ACCEPTED", "A_CANCELLED", "O_SENT_BACK"]
order_of_actions_custom = [["O_SELECTED", "A_FINALIZED", "O_CREATED", "O_SENT"], ["O_CANCELLED", "O_SELECTED", "O_CREATED", "O_SENT"],
                           ["O_ACCEPTED", "A_APPROVED", "A_REGISTERED", "A_ACTIVATED"], ["A_CANCELLED", "O_CANCELLED"]]
min_support = 3

def compare_actions_custom(a, b):
    a_name = a["concept:name"]
    b_name = b["concept:name"]
    for list_tmp in order_of_actions_custom:
        if a_name in list_tmp and b_name in list_tmp:
            if list_tmp.index(a_name) > list_tmp.index(b_name):
                return 1
            else:
                return -1
    return 0


def compare_actions(a, b):
    a_name = a["concept:name"]
    b_name = b["concept:name"]
    pair = tuple(sorted((a_name, b_name)))
    if pair in order_dict.keys():
        score_a = order_dict[pair][a_name]
        score_b = order_dict[pair][b_name]
        if score_a > 0 and score_b == 0:
            return -1
        elif score_a == 0 and score_b == 0:
            return 1
    return 0


def main(path_in):
    log = xes_importer.apply(path_in)
    matrix = []
    for trace in log:
        matrix.append([e["concept:name"] for e in trace])
    a = TransactionEncoder()
    a_data = a.fit(matrix).transform(matrix)
    df = pd.DataFrame(a_data, columns=a.columns_)
    df = df.replace(False, 0)
    df_association_rules = apriori(df, min_support=0.6, use_colnames=True, verbose=1)
    df_association_rules = df_association_rules.sort_values('support')
    print(df_association_rules)


def seqDiscovery(log):
    seqs = [[x["concept:name"] for x in trace] for trace in log[:10]]
    report = seqmining.freq_seq_enum(seqs, 10)
    print(report)


def removeComponents(log):
    #filtered_log = variants_filter.filter_variants_by_coverage_percentage(log, min_coverage_percentage=0.0001)
    filtered_log = log
    G = kosaraju.DiGraph()
    events_set = set()
    vertex_dict = dict()
    connections_set = set()
    connections_count = dict()
    for trace in filtered_log:
        last_event = None
        for e in trace:
            events_set.add(e["concept:name"])
            if last_event:
                key = (last_event, e["concept:name"])
                if key in connections_count.keys():
                    connections_count[key] += 1
                else:
                    connections_count[key] = 1
                connections_set.add((last_event, e["concept:name"]))
            last_event = e["concept:name"]

    for e in events_set:
        vertex_dict[e] = kosaraju.Vertex(e)

    G.add_vertices([v for e, v in vertex_dict.items()])

    #for (e1, e2) in connections_set:
    #    G.add_edge(vertex_dict[e1], vertex_dict[e2])

    for k, c in connections_count.items():
        if c > min_support:
            G.add_edge(vertex_dict[k[0]], vertex_dict[k[1]])

    y = kosaraju.kosaraju(G)

    for j in range(len(y)):
        if y[j] != []:
            print("Component:", j + 1, " ", end=" ")
            for v in y[j]:
                print(v.value, end=" ")
            print()


    print(len(filtered_log) / len(log))


def orderEventsOnTimestamps_custom(log_path):
    log = pm4py.read_xes(log_path)
    output_log = EventLog()
    for name, value in log.attributes.items():
        output_log.attributes[name] = value

    for trace in log:
        trace_timestamps_set = set()
        events_by_timestamp = dict()
        for event in trace[1:-1]:
            ev = copy.deepcopy(event)
            timestamp = ev["time:timestamp"]
            # timestamp = event["time:timestamp"].strftime('%Y-%m-%d %H:%M:%S.%f')[:-4]
            trace_timestamps_set.add(timestamp)
            if timestamp in events_by_timestamp.keys():
                events_by_timestamp[timestamp].append(ev)
            else:
                events_by_timestamp[timestamp] = [ev]
        new_trace = Trace()
        for name, value in trace.attributes.items():
            new_trace.attributes[name] = value
        new_trace.append(trace[0])
        for timestamp in sorted(list(trace_timestamps_set)):
            [new_trace.append(x) for x in sorted(events_by_timestamp[timestamp], key=functools.cmp_to_key(compare_actions_custom))]
        new_trace.append(trace[-1])
        output_log.append(new_trace)

    pm4py.write_xes(output_log, log_path.replace(".xes", "_hand_ordered.xes"))


def orderEventsOnTimestamps(log_path):
    log = pm4py.read_xes(log_path)
    output_log = EventLog()
    for name, value in log.attributes.items():
        output_log.attributes[name] = value

    for trace in log:
        trace_timestamps_set = set()
        events_by_timestamp = dict()
        for event in trace[1:-1]:
            ev = copy.deepcopy(event)
            timestamp = ev["time:timestamp"]
            # timestamp = event["time:timestamp"].strftime('%Y-%m-%d %H:%M:%S.%f')[:-4]
            trace_timestamps_set.add(timestamp)
            if timestamp in events_by_timestamp.keys():
                events_by_timestamp[timestamp].append(ev)
            else:
                events_by_timestamp[timestamp] = [ev]
        new_trace = Trace()
        for name, value in trace.attributes.items():
            new_trace.attributes[name] = value
        new_trace.append(trace[0])
        for timestamp in sorted(list(trace_timestamps_set)):
            [new_trace.append(x) for x in sorted(events_by_timestamp[timestamp], key=functools.cmp_to_key(compare_actions))]
        new_trace.append(trace[-1])
        output_log.append(new_trace)

    pm4py.write_xes(output_log, log_path.replace(".xes", "_clean.xes"))


def extractSameTimeEvents(log_path, micro=False):
    log = pm4py.read_xes(log_path)

    same_timestamps_events_set_counter = dict()
    for trace in log:
        events_by_timestamp = dict()
        for event in trace[1:-1]:
            timestamp = get_event_timestamp(event, micro)
            if timestamp in events_by_timestamp.keys():
                events_by_timestamp[timestamp].add(event["concept:name"])
            else:
                events_by_timestamp[timestamp] = {event["concept:name"]}
        for k, v in events_by_timestamp.items():
            if len(list(v)) > 1:
                if tuple(v) in same_timestamps_events_set_counter.keys():
                    same_timestamps_events_set_counter[tuple(v)] += 1
                else:
                    same_timestamps_events_set_counter[tuple(v)] = 1

    same_timestamps_event_pairs_counter = {}
    for k, v in same_timestamps_events_set_counter.items():
        event_pair_list = list(combinations(k,2))
        for event_pair in event_pair_list:
            event_pair = tuple(sorted(event_pair))
            if event_pair in same_timestamps_event_pairs_counter.keys():
                same_timestamps_event_pairs_counter[event_pair] += v
            else:
                same_timestamps_event_pairs_counter[event_pair] = v

    # look in the log if there are cases in which this pairs actually have different timestamps, and if so it counts how many times each activity come first
    same_timestamps_event_priority_order = {k: {e: 0 for e in k} for k, v in same_timestamps_event_pairs_counter.items()}
    for trace in log:
        previous_timestamp = get_event_timestamp(trace[0], micro)
        previous_event = trace[0]["concept:name"]
        for event in trace[1:-1]:
            current_timestamp = get_event_timestamp(event, micro)
            current_pair = tuple(sorted((previous_event, event["concept:name"])))
            # debug
            # if current_pair == tuple({'A_REGISTERED', 'O_ACCEPTED'}):
            #     print('here')
            if current_pair in same_timestamps_event_priority_order.keys():
                if previous_timestamp < current_timestamp:
                    same_timestamps_event_priority_order[current_pair][previous_event] += 1
                elif previous_timestamp > current_timestamp:
                    same_timestamps_event_priority_order[current_pair][event["concept:name"]] += 1
            previous_event = event["concept:name"]
            previous_timestamp = current_timestamp

    return same_timestamps_events_set_counter, same_timestamps_event_priority_order

def get_event_timestamp(event, micro):
    if micro:
        timestamp = event["time:timestamp"].replace(microsecond=0)
    else:
        timestamp = event["time:timestamp"]
    return timestamp



if __name__ == "__main__":
    #main("../../cluster_data/output_logs/BPI2012_log_eng_positional_cumulative_squashed_training_80.xes")
    #removeComponents(pm4py.read_xes("../../data/logs/BPI_2012/BPI_2012_log_eng_rewards_cumulative_durations.xes"))
    #seqDiscovery(pm4py.read_xes("../../data/logs/BPI_2012/BPI_2012_log_eng_rewards_cumulative_durations.xes"))
    orderEventsOnTimestamps("../../data/logs/BPI_2012/BPI_2012_log_eng.xes")
    #same_timestamps_events_set_counter = extractSameTimeEvents("../../data/logs/BPI_2012/BPI_2012_log_eng_rewards_cumulative_durations.xes")
    #same_timestamps_micro_events_set_counter = extrateSameTimeEvents("../../data/logs/BPI_2012/BPI_2012_log_eng_rewards_cumulative_durations.xes", True)
    #print(set(same_timestamps_events_set_counter).intersection(set(same_timestamps_micro_events_set_counter)))

    # automatic pipeline which finds same timestamp events and sees if actually there is a consistent (always the same) ordering elsewhere in the log
    same_timestamps_events_set_counter, order_dict = extractSameTimeEvents("../../data/logs/KR23/BPI_2017_log.xes")
    orderEventsOnTimestamps("../../data/logs/KR23/BPI_2017_log.xes")