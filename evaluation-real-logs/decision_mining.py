import pm4py
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import json
import pickle
from sklearn.metrics import f1_score
from sklearn import tree
from hyperopt import tpe
import numpy as np
from hyperopt import Trials, hp, fmin
from hyperopt.pyll import scope
from pm4py.algo.conformance.alignments.dfg import algorithm as dfg_alignment
from pm4py.objects import log as log_lib
from pm4py.algo.conformance.alignments.dfg.variants.classic import Parameters
from pm4py import util


def extract_event_from_trace(trace, activity):
    i = len(trace)-1
    while i >= 0:
        if trace[i]['concept:name'] == activity:
            return trace[i]
        else:
            i -= 1
    return None


def find_max_length(prefix_traces):
    maximum = []
    for key in prefix_traces:
        list_len = [len(i) for i in prefix_traces[key]]
        maximum.append(max(list_len) if len(list_len) > 0 else 0)
    return max(maximum)


def del_other_traces(trace_log, caseid):
    index_to_delete = []
    for trace in trace_log:
        if trace.attributes['concept:name'] != caseid:
            index_to_delete.append(trace.attributes['concept:name'])
    del trace_log[index_to_delete[0]]

###############################################################################################

def define_encoding_simple_for_trace(elements2n, encoded_traces, trace, WINDOW_SIZE):
    if len(trace) > WINDOW_SIZE:
        trace = trace[-WINDOW_SIZE:]
    for idx, e in enumerate(trace):
        encoded_traces[idx] = elements2n[e]
    return encoded_traces

##### simple_index, con window 10 #####
def simple_index_encoding(elements2n, aligned_traces):
    WINDOW_SIZE = 30
    PAD = len(elements2n)
    log_encoded = []
    columns = ['pr_' + str(i) for i in range(0, WINDOW_SIZE)]
    columns += ['amount', 'label']
    for trace in aligned_traces:
        encoded_traces = [PAD]*WINDOW_SIZE + [trace["amount"]] + [trace["label"]]
        if trace["trace"].count(trace["label"]) > 1:
            all_indexes = [i for i, val in enumerate(trace["trace"]) if val == trace["label"]]
            for index in all_indexes:
                log_encoded.append(define_encoding_simple_for_trace(elements2n, encoded_traces, trace["trace"][:index], WINDOW_SIZE))
        else:
            log_encoded.append(define_encoding_simple_for_trace(elements2n, encoded_traces, trace["trace"][:trace["trace"].index(trace["label"])], WINDOW_SIZE))
    df = pd.DataFrame(log_encoded, columns=columns)
    return df, columns


def define_encoding_frequency_for_trace(encoded_traces, trace, encoding):
    for e in encoding:
        if trace.count(e) > 0:
            encoded_traces[encoding.index(e)] += 1
    return encoded_traces


###### usa questo, aggiungi solo l'amount come attributo di traccia
def frequency_encoding(elements2n, aligned_traces):
    encoding = list(elements2n.keys())
    log_encoded = []
    columns = encoding + ['amount', 'label']
    for trace in aligned_traces:
        encoded_traces = [0]*len(encoding) + [trace["amount"]] + [trace["label"]]
        if trace["trace"].count(trace["label"]) > 1:
            all_indexes = [i for i, val in enumerate(trace["trace"]) if val == trace["label"]]
            for index in all_indexes:
                log_encoded.append(define_encoding_frequency_for_trace(encoded_traces, trace["trace"][:index], encoding))
        else:
            log_encoded.append(define_encoding_frequency_for_trace(encoded_traces, trace["trace"][:trace["trace"].index(trace["label"])], encoding))
    df = pd.DataFrame(log_encoded, columns=columns)
    return df, columns


def trace_filter_decision_point(aligned_log, list_next_elements):
    traces = []
    for trace in aligned_log:
        for element in list_next_elements:
            if element in trace['trace']:
                trace_filter = dict(trace)
                trace_filter["label"] = element
                traces.append(trace_filter)
    return traces


def retrieve_aligned_trace(trace, dfg_pm4py, start_node, end_node):
    aligned_traces = dfg_alignment.apply(trace, dfg_pm4py, start_node, end_node)[0]
    new_trace = []
    for event in aligned_traces['alignment']:
        if event[1] != '>>':
            new_trace.append(event[1])
    return new_trace


def retrieve_aligned_trace_customize(trace, dfg_pm4py, start_node, end_node, BPMN_ELEMENTS):
    #### define the cost for the alignments ####
    model_cost_function = dict()
    #sync_cost_function = dict()
    log_cost_function = dict()
    internal_cost_function = dict()
    for node in BPMN_ELEMENTS:
        if BPMN_ELEMENTS[node] != 'task':
            model_cost_function[node] = 1
            log_cost_function[node] = 1
            internal_cost_function[node] = 1
        else:
            model_cost_function[node] = 10000
            log_cost_function[node] = 10000
            internal_cost_function[node] = 10000
    params = dict()
    params[util.constants.PARAMETER_CONSTANT_ACTIVITY_KEY] = log_lib.util.xes.DEFAULT_NAME_KEY
    params[Parameters.MODEL_MOVE_COST_FUNCTION] = model_cost_function
    #params[Parameters.SYNC_COST_FUNCTION] = sync_cost_function
    params[Parameters.LOG_MOVE_COST_FUNCTION] = log_cost_function
    params[Parameters.INTERNAL_LOG_MOVE_COST_FUNCTION] = internal_cost_function
    aligned_traces = dfg_alignment.apply(trace, dfg_pm4py, start_node, end_node, parameters=params)[0]
    new_trace = []
    for event in aligned_traces['alignment']:
        if event[1] != '>>':
            new_trace.append(event[1])
    return new_trace



def retrieve_decision_points_DFG(DFG, BPMN_ELEMENTS):
    decision_points = []
    for key in DFG:
        if BPMN_ELEMENTS[key] == 'exclusiveGateway' and len(DFG[key]) > 1: #### keep just the "Diverging" gateway
            decision_points.append(key)
    return decision_points


PATH_LOG = 'input/BPIC17_parallel/BPI_2017_log_strip_newcredit_100_ForTEST_fixed_times.csv' ##### path for the log give as input
path_dfg = 'input/BPIC17_parallel/BPIC17_parallel_DFG.json' ### path of the DFG retrieve in the previous step with "preprocessing_bpmn.py"


with open(path_dfg) as file:
    data = json.load(file)
    ACTIVITIES = data['activities']
    BPMN_ELEMENTS = data["elements_bpmn"]
    DFG = data["DFG"]
    start_node = data["start_node"]
    end_node = data["end_node"]
    ACT_TO_NODE = data["act_node"]
    SEQUENCE_FLOW_TARGET = data["sequenceFlow_target"]
    DFG_only_task = data["DFG_only_task"]

###### retrieve dfg for alignment #####
sa = {start_node: 1}
ea = {end_node: 1}
dfg_pm4py = {}
for key in DFG:
    for next in DFG[key]:
        dfg_pm4py[(key,next)] = 1
dfg_pm4py = tuple(dfg_pm4py)
log = pd.read_csv(PATH_LOG)
log = pm4py.format_dataframe(log, case_id='case:concept:name', activity_key='concept:name', timestamp_key='time:timestamp')
log['time:timestamp'] = pd.to_datetime(log['time:timestamp'])
log['start:timestamp'] = pd.to_datetime(log['start:timestamp'])
decision_points = retrieve_decision_points_DFG(DFG, BPMN_ELEMENTS)
aligned_log = []
caseid_list = list(log['case:concept:name'].unique())
for caseid in caseid_list:
    group_case = log[log['case:concept:name'] == caseid].sort_values(by='start:timestamp')
    group_case['concept:name'] = group_case['concept:name'].map(ACT_TO_NODE)
    #group_case['concept:name'] = group_case['concept:name'].map(ACT_TO_NODE)
    new_trace = {'trace': retrieve_aligned_trace_customize(group_case, dfg_pm4py, sa, ea, BPMN_ELEMENTS), "amount": group_case.iloc[0]['amount']}
    aligned_log.append(new_trace)
print('######################## LOG aligned ########################')
data_to_save = {}
elements_for_prefix = []
for key in BPMN_ELEMENTS:
    if BPMN_ELEMENTS[key] not in ['startEvent', 'endEvent']:
        elements_for_prefix.append(key)
elements2n = {a: n for n, a in enumerate(elements_for_prefix)}
n2elements = {n: a for n, a in enumerate(elements_for_prefix)}
data_to_save['elements2n'] = elements2n
data_to_save['n2elements'] = n2elements
data_to_save['PAD'] = len(elements2n)

for decision in decision_points:
    print("############## Compute training for", decision, "decision point ##############")
    list_next_elements = DFG[decision] ### possible path from the decision point
    traces_filter = trace_filter_decision_point(aligned_log, list_next_elements)
    #df, columns = frequency_encoding(elements2n, traces_filter)
    df, columns = simple_index_encoding(elements2n, traces_filter)

    data_to_save[decision] = {}
    data_to_save[decision]['transitions'] = list_next_elements
    data_to_save[decision]['#traces'] = len(traces_filter)
    data_to_save[decision]["encoding"] = list(columns)[:-1]
    y = df.label  # Target variable
    X = df.drop(['label'], axis=1)
    if len(df) > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

        space = {'max_depth': scope.int(hp.quniform('max_depth', 1, 4, 1)),
                 'min_samples_split': scope.int(hp.uniform('min_samples_split', 2, 11)),
                 'min_samples_leaf': scope.int(hp.uniform('min_samples_leaf', 3, 26)),
                 'criterion': hp.choice('criterion', ['gini'])}

        criterion_dict = {0: 'gini', 1: 'entropy', 2: 'log_loss'}

        def objective(params):
            clf = DecisionTreeClassifier(**params)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            return -f1_score(y_val, y_pred, average='macro')


        trials = Trials()
        best = fmin(fn=objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=100,
                    trials=trials)

        print("Best Hyperparameters:", best)
        clf = DecisionTreeClassifier(criterion=criterion_dict[best['criterion']],
                                     max_depth=int(best['max_depth']),
                                     min_samples_leaf=int(best['min_samples_leaf']),
                                     min_samples_split=int(best['min_samples_split']))
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        data_to_save[decision]['best_parameters'] = str(best)
        data_to_save[decision]['Accuracy'] = metrics.accuracy_score(y_test, y_pred)
        data_to_save[decision]['f1'] = f1_score(y_test, y_pred, average=None).tolist()
        data_to_save[decision]['f1_score_macro'] = f1_score(y_test, y_pred, average='macro')
        data_to_save[decision]['f1_score_micro'] = f1_score(y_test, y_pred, average='micro')
        data_to_save[decision]['f1_score_weighted'] = f1_score(y_test, y_pred, average='weighted')
        data_to_save[decision]['confusion_matrix'] = confusion_matrix(y_test, np.array(y_pred)).tolist()
        data_to_save[decision]['features_importances'] = {list(df.columns)[i]: v for i, v in
                                                          enumerate(clf.feature_importances_)}
        data_to_save[decision]['tree'] = tree.export_text(clf)
        data_to_save[decision]['encoding2target'] = n2elements
        pickle.dump(clf, open(decision +'.pkl', 'wb'))
        data_to_save[decision]['prediction'] = True
    else:
        data_to_save[decision]['prediction'] = False


with open('BPIC17_parallel' + "_decision_points.json", "w") as outfile:
    json.dump(data_to_save, outfile, indent=len(data_to_save))
