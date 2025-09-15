import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math
import random

# global variables def: header label of the MDP csv
state_label = "s"
action_label = "a"
next_state_label = "s\'"
probability_label = "p_r"
reward_label = "reward"
n_occurrences_label = "number_occurrences"
q_value_label = "q"
scale_factor_label = "scale_factor"
policy_label = "policy"


def compute_policy(df):
    # compute the greedy policy from the q-values
    state_set = set(df[state_label])
    max_q_dict = {}
    for s in state_set:
        filtered_df = df.loc[df[state_label] == s,[state_label, q_value_label]]
        max_q_dict[s] = max(filtered_df[q_value_label])
    df['max_q'] = df[state_label].map(max_q_dict)
    df[policy_label] = np.where(df[q_value_label] == df['max_q'], 1, 0)
    df = df.drop(columns=['max_q'])
    return df


def total_occurrences(group, label):
    # perform aggregation
    g = group[n_occurrences_label].agg('sum')
    group[label] = g
    return group

def preprocess_df(df, normalize_reward, change_zero_reward):
    # add q-value with zeros
    df[q_value_label] = 0

    # compute total number of occurrences per action
    df = df.groupby([state_label, action_label], group_keys=False).apply(total_occurrences, 'sum_n_occurrences')

    # normalize reward column
    if normalize_reward:
        minmax_reward = MinMaxScaler(feature_range=(0, 1))
        df['scaled_reward'] = minmax_reward.fit_transform(df[[reward_label]])
        if change_zero_reward:
            df['new_reward'] = df['scaled_reward']
        else:
            df['new_reward'] = np.where(df[reward_label] == 0, 0, df['scaled_reward'])
        df[reward_label] = df['new_reward']  # for debug
        df = df.drop(columns=['scaled_reward', 'new_reward'])
    return df

def add_scale_factor(df, scale_factor):
    # define scale factor (h in the paper), exp stands for the sigmoid function
    # redefine scale factor label for the df to include the type of scaling
    # scale_factor_label = scale_factor_label + "_" + scale_factor_type
    if scale_factor == "none":
        df[scale_factor_label] = 1
    elif scale_factor == "expQ1":
        state_action_occurences_list = (df[['s', 'a', 'number_occurrences']].groupby(['s', 'a']).sum())['number_occurrences'].to_list()
        q1 = np.quantile(state_action_occurences_list, 0.25)
        df[scale_factor_label] = df["sum_n_occurrences"].apply(lambda x: -2 * (math.exp(-x / q1) / (1 + math.exp(-x / q1))) + 1)
    elif scale_factor == "expQ2":
        state_action_occurences_list = (df[['s', 'a', 'number_occurrences']].groupby(['s', 'a']).sum())['number_occurrences'].to_list()
        q2 = np.quantile(state_action_occurences_list, 0.5)
        df[scale_factor_label] = df["sum_n_occurrences"].apply(lambda x: -2 * (math.exp(-x / q2) / (1 + math.exp(-x / q2))) + 1)
    elif scale_factor == "expQ3":
        state_action_occurences_list = (df[['s', 'a', 'number_occurrences']].groupby(['s', 'a']).sum())[
            'number_occurrences'].to_list()
        q3 = np.quantile(state_action_occurences_list, 0.75)
        df[scale_factor_label] = df["sum_n_occurrences"].apply(
            lambda x: -2 * (math.exp(-x / q3) / (1 + math.exp(-x / q3))) + 1)
    elif scale_factor == "stepQ1":
        state_action_occurences_list = (df[['s', 'a', 'number_occurrences']].groupby(['s', 'a']).sum())[
            'number_occurrences'].to_list()
        q1 = np.quantile(state_action_occurences_list, 0.25)
        df[scale_factor_label] = df["sum_n_occurrences"].apply(lambda x: 0 if x <= q1 else 1)
    elif scale_factor == "stepQ2":
        state_action_occurences_list = (df[['s', 'a', 'number_occurrences']].groupby(['s', 'a']).sum())[
            'number_occurrences'].to_list()
        q2 = np.quantile(state_action_occurences_list, 0.5)
        df[scale_factor_label] = df["sum_n_occurrences"].apply(lambda x: 0 if x <= q2 else 1)
    elif scale_factor == "stepQ3":
        state_action_occurences_list = (df[['s', 'a', 'number_occurrences']].groupby(['s', 'a']).sum())[
            'number_occurrences'].to_list()
        q3 = np.quantile(state_action_occurences_list, 0.75)
        df[scale_factor_label] = df["sum_n_occurrences"].apply(lambda x: 0 if x <= q3 else 1)

    df = df.drop(columns=['sum_n_occurrences'])
    return df

def create_next_state_dict(next_state_df):
    # create dict {next_state: {"p": p, "r": r, "n_occ": n_occ}}
    next_state_df.set_index(next_state_label, inplace=True)
    next_state_dict = next_state_df.to_dict('index')
    return next_state_dict

def generate_q_table(MDP_df):
    """
    q-table is a dict (state) of dict (action)
    each values is a couple: the first value is the q-value, the second value is a dict
    the dict is {next_state: {"p": p, "r": r}}
    in total {s: {a: {"q": 0, "next_state_dict": {s': {"p": p, "r": r}}, "scale_factor": scale_factor}}}
    """
    #extract states
    states_list = np.unique(MDP_df[state_label])
    # define action dictionary and q_table
    q_table = {}
    state_action_dict = {}
    for s in states_list:
        state_action_dict[s] = np.unique(MDP_df.loc[MDP_df[state_label] == s, [action_label]])
        q_table[s] = {}

    # build q_table
    for s, a_list in state_action_dict.items():
        for a in a_list:
            next_state_df = MDP_df.loc[(MDP_df[state_label] == s) & (MDP_df[action_label] == a),
                                       [next_state_label, probability_label, reward_label, scale_factor_label]]
            scale_factor = next_state_df[scale_factor_label].to_numpy()[0]
            next_state_df = next_state_df.drop(columns=[scale_factor_label])
            next_state_dict = create_next_state_dict(next_state_df)
            q_table[s][a] = {"q": 0, "next_state_dict": next_state_dict, scale_factor_label: scale_factor}

    return states_list, state_action_dict, q_table


def Q_evaluation(state_action_dict, q_table, threshold, gamma, verbose=True):
    # direct policy evaluation and improvement of the q-values
    # this method implicitly combine policy evaluation and policy iteration
    max_delta = threshold + 1
    runs = 0
    keys = [[(s,a) for a in a_list] for s, a_list in state_action_dict.items()]
    keys = sum(keys, [])
    # state_action dict initialized with q_table
    Q = {(s, a): q_table[s][a][q_value_label] for s, a in keys}
    while max_delta > threshold:
        Q_temp = {(s, a): 0 for s, a in keys}
        for state, action in keys:
            next_state_dict = q_table[state][action]['next_state_dict']
            scale_factor = q_table[state][action][scale_factor_label]
            for next_state, info in next_state_dict.items():
                proba = info['p_r']
                reward = info['reward']
                if 'END' in next_state:
                    max_Q = 0
                else:
                    next_actions_values = [Q[(next_state, a)] for a in state_action_dict[next_state]]
                    max_Q = max(next_actions_values)
                Q_temp[(state, action)] += scale_factor * proba * (reward + gamma * max_Q)
        Q_diff = [abs(Q[s_a] - Q_temp[s_a]) for s_a in keys]
        max_delta = max(Q_diff)
        runs += 1
        Q = Q_temp.copy()
        if runs % 1 == 0:
            start_q_value = Q[keys[0]]
            if verbose:
                print("Run:", runs, ", Delta:", max_delta, ", Start Value:", start_q_value)
    print("Run:", runs, ", Delta:", max_delta, ", Start Value:", start_q_value)
    return Q


def update_q_table(q_table, Q):
    # uopdate the q-values in the q-table
    for s, a in Q.keys():
        q_table[s][a][q_value_label] = Q[(s,a)]
    return q_table

def update_mdp_df(MDP_df, Q):
    # update mdp using the q-values stored in the q-table
    for (s, a), q in Q.items():
        MDP_df.loc[(MDP_df[state_label] == s) & (MDP_df[action_label] == a), [q_value_label]] = q
    return MDP_df

def extract_policy_from_mdp(MDP_df):
    # extract policy as dict from the mdp
    policy_df = MDP_df[MDP_df['policy']==1][['s', 'a']].drop_duplicates()
    policy_df.set_index('s', inplace=True)
    policy = policy_df['a'].to_dict()
    return policy


def policy_evaluation(states_list, state_action_dict, q_table, policy, threshold, gamma):
    # perform policy evaluation (update V-values according to the current policy)
    V = {s: 0 for s in states_list}  # initialized as 0
    max_delta = threshold + 1
    while max_delta > threshold:
        V_temp = {s: 0 for s in state_action_dict.keys()}
        for state in states_list:
            action = policy[state]
            next_state_dict = q_table[state][action]['next_state_dict']
            scale_factor = q_table[state][action][scale_factor_label]
            for next_state, info in next_state_dict.items():
                proba = info['p_r']
                reward = info['reward']
                if 'END' in next_state:
                    V_temp[state] += scale_factor * proba * (reward)
                else:
                    V_temp[state] += scale_factor * proba * (reward + V[next_state] * gamma)
        V_diff = [abs(V[s] - V_temp[s]) for s in states_list]
        max_delta = max(V_diff)
        V = V_temp.copy()
    return V

def policy_improvement(states_list, state_action_dict, q_table, V, gamma):
    # perform policy improvement (take the greedy policy wrt to the current V-values)
    policy = {}
    for state in states_list:
        action_values = {a: 0 for a in state_action_dict[state]}
        for action in action_values.keys():
            next_state_dict = q_table[state][action]['next_state_dict']
            scale_factor = q_table[state][action][scale_factor_label]
            for next_state, info in next_state_dict.items():
                proba = info['p_r']
                reward = info['reward']
                if 'END' in next_state:
                    action_values[action] += scale_factor * proba * (reward)
                else:
                    action_values[action] += scale_factor * proba * (reward + V[next_state] * gamma)
        policy[state] = max(action_values, key=action_values.get)
    return policy

def policy_iteration(states_list, state_action_dict, q_table, threshold, gamma):
    # perform policy iteration
    # initialize policy randomly
    policy = {s: random.choice(a_list) for s, a_list in state_action_dict.items()}
    old_policy = {}
    iteration = 0
    while iteration <= 20:
        V = policy_evaluation(states_list, state_action_dict, q_table, policy, threshold, gamma)
        new_policy = policy_improvement(states_list, state_action_dict, q_table, V, gamma)
        iteration += 1
        if iteration % 1 == 0:
            start_value = V[states_list[0]]
            print("Run:", iteration, ", Start Value:", start_value)
        if new_policy == policy or new_policy == old_policy:
            return V, new_policy
        old_policy = policy.copy()  # add this to avoid oscillations between 2 policies
        policy = new_policy.copy()

    return V, policy


def compare_DP_methods(policy_Q, policy_PI):
    # check two policies
    print('The two policies computed respectively with Q evaluation and policy iteration are the same?')
    if policy_Q == policy_PI:
        print('Yes!')
    elif policy_Q.keys() == policy_PI.keys():
        print('They have the same keys but they differ by these terms')
        policy_diff = {s: {'Q': policy_Q[s], 'PI': policy_PI[s]} for s in policy_Q.keys() if
                       policy_Q[s] != policy_PI[s]}
        for item in policy_diff.items():
            print(item)
    else:
        print("No! They even don't have the same keys!")
        keys1 = set(policy_Q.keys())
        keys2 = set(policy_PI.keys())
        # symmetric difference
        print("Missing keys :", keys1 ^ keys2)


def DP_training_policy(input_file, output_file, scale_factor, threshold, gamma, normalize_reward, change_zero_reward):
    print("Importing the MDP model")
    # load the mdp model
    MDP_df = pd.read_csv(input_file)
    # add columns to df: q-value, summed number of occurences, and normalize_reward
    MDP_df = preprocess_df(MDP_df, normalize_reward, change_zero_reward)
    # add scale_factor
    MDP_df = add_scale_factor(MDP_df, scale_factor)
    # Q-matrix and co.
    states_list, state_action_dict, q_table = generate_q_table(MDP_df)
    # Q evaluation (is it like policy evaluation and policy improvement together?)
    Q = Q_evaluation(state_action_dict, q_table, threshold, gamma, verbose=False)
    # update MDP_df
    MDP_df = update_mdp_df(MDP_df, Q)
    # compute policy
    MDP_df = compute_policy(MDP_df)
    # extract policy as dict
    policy_Q = extract_policy_from_mdp(MDP_df)
    # export to csv
    MDP_df.to_csv(output_file, index=False)
    print("Result exported to: ", output_file)

    return states_list, state_action_dict, q_table, policy_Q


def DP_training_main(folder_name, file_name, scale_factor, threshold, gamma, normalize_reward, change_zero_reward):
    print("file: %s, scale factor: %s" % (file_name, scale_factor))
    mdp_name = file_name + "_training_preprocessed_wloops.csv"
    policy_prefix = mdp_name.replace("_preprocessed_wloops.csv", "_policy_opt_DP_")
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    input_file = os.path.join(ROOT, "cluster_data", folder_name, "output_mdps", mdp_name)
    policy_suffix = ''.join(scale_factor)
    output_file = os.path.join(ROOT, "cluster_data", folder_name, "output_policies",
                               policy_prefix + policy_suffix + ".csv")

    # train the policy with DP method
    states_list, state_action_dict, q_table, policy_Q = DP_training_policy(input_file, output_file, scale_factor,
                                                                           threshold, gamma, normalize_reward,
                                                                           change_zero_reward)

    return states_list, state_action_dict, q_table, policy_Q

if __name__ == "__main__":
    # script to train all the policies for all the scale factor choices and all the event logs considered in the paper
    # global variables def: RL hyperparameters
    threshold = 0.00001  # threshold at which algorithm stops
    gamma = 1  # discount rate, discount near to 1 avoids loops
    normalize_reward = True  # use minmaxscaler on reward?
    change_zero_reward = False  # if minmaxscaler on reward is used, apply also to zero reward?
    compare_with_policy_iteration = False  # alternative DP method to check robusteness of the policy learned

    folder_name = "folder1"
    # list of the scale functions h used in the paper, exp stand for the sigmoid functions
    scale_factor_list = ["none", "expQ1", "expQ2", "expQ3", "stepQ1", "stepQ2", "stepQ3",]
    scale_factor_list = ["expQ2",]
    file_name_list = ['BPI_2012_log_eng_clean_140',
                      'BPI_2017_log_strip_newcredit_100',
                      'event_log_2000_100', 'event_log_rare_2000_120',
                      'event_log_4000_140', 'event_log_rare_4000_120',
                      'event_log_8000_120', 'event_log_rare_8000_100',
                      'event_log_16000_120', 'event_log_rare_16000_100']
    file_name_list = ['event_log_2000_100']
    for file_name in file_name_list:
        for scale_factor in scale_factor_list:
            # start the DP training pipeline to learn the optimal policy
            states_list, state_action_dict, q_table, policy_Q = DP_training_main(folder_name, file_name, scale_factor)

            # compare alternative DP methods to check robustness of the policy learned
            if compare_with_policy_iteration:
                # policy iteration method
                V, policy_PI = policy_iteration(states_list, state_action_dict, q_table, threshold, gamma)
                compare_DP_methods(policy_Q, policy_PI)