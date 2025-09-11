import glob
import random
import pandas as pd
import pickle


class cluster_model():

    def __init__(self, folder_path, policy_name, normalization=True, last_position=True):
        self.folder_path = folder_path
        self.policy_name = policy_name
        self.normalization = normalization
        self.MODEL_CLUSTER_SAVE = glob.glob(self.folder_path + '*model.pkl')[0]
        self.PICKLE_SAVE = glob.glob(self.folder_path + '*pickle.pkl')[0]
        self.POLICY_FILE = glob.glob(self.folder_path + '*' + self.policy_name + '*')[0]
        self.policy_df = pd.read_csv(self.POLICY_FILE)
        file = open(self.PICKLE_SAVE, 'rb')
        self.events_set = pickle.load(file)
        self.max_trace_length = pickle.load(file)
        self.reward_mmscaler = pickle.load(file)
        _ = pickle.load(file) # this was self.last_a_to_cluster but had some cluster not present in MDP (because of following env activities)
        file.close()
        self.extract_last_a_to_cluster()
        self.kmean = pickle.load(open(self.MODEL_CLUSTER_SAVE, 'rb'))
        self.last_position = last_position

    def extract_last_a_to_cluster(self):
        states_series = self.policy_df["s"].copy()
        df_activity_clusters = pd.DataFrame(
            states_series.str.replace("<", "").str.replace(">", "").str.replace("|", "").str.split().to_list(),
            columns=['activity', 'cluster']).drop_duplicates()
        df_activity_clusters = df_activity_clusters[df_activity_clusters['activity'].notnull()]
        dict_activity_clusters = df_activity_clusters.groupby("activity")["cluster"].apply(list).to_dict()
        for v in dict_activity_clusters.values():
            if None in v:
                v.remove(None)
        self.last_a_to_cluster = {k: set(map(int, v)) for k, v in dict_activity_clusters.items()}


    def encode_vector(self, prefix):
        # IMPORTANT NOTE: this prefix should be full but without start
        # the encoder automatically removes last activity
        # then the cluster number is associated to prefix[:-1]
        events_dict = {e: {'count': 0, 'last_position': 0} for e in self.events_set}
        events_dict["START"]["count"] = 1
        events_dict["START"]["last_position"] = 1
        if len(prefix) == 0:
            # should never be the case
            encoded_vector = [float(0) for e, d in events_dict.items()] + \
                             [float(0) for e, d in events_dict.items()]
            encoded_vector.append(self.reward_mmscaler.transform([[0.0]])[0][0])
        else:
            for j, event_name in enumerate(prefix):
                if self.normalization and self.reward_mmscaler.n_features_in_ == 1:
                    # this is the custom normalization of freq and position with minmaxscaler only for reward
                    encoded_vector = [float(d['count'] / self.events_set[e]) for e, d in events_dict.items()] + [
                        float(d['last_position'] / self.max_trace_length) for e, d in events_dict.items()]
                    encoded_vector.append(
                        self.reward_mmscaler.transform([[0.0]])[0][0])  # reward since it will never be at the END
                elif self.normalization and self.reward_mmscaler.n_features_in_ > 1:
                    # this is the normalization with minmaxscaler on all the features
                    encoded_vector = [float(d['count']) for e, d in events_dict.items()] + [
                        float(d['last_position']) for e, d in events_dict.items()]
                    encoded_vector.append(0.0)
                    encoded_vector = self.reward_mmscaler.transform([encoded_vector])[0]
                else:
                    # no normalization at all
                    encoded_vector = [float(d['count']) for e, d in events_dict.items()] + [
                        float(d['last_position']) for e, d in events_dict.items()]
                    encoded_vector.append(0.0)  # reward since it will never be at the END
                # Updating the dictionary
                events_dict[event_name]['count'] += 1
                if self.last_position:
                    events_dict[event_name]['last_position'] = j + 2
                elif events_dict[event_name]['last_position'] == 0:
                    # in this case is the first position and not the last position
                    events_dict[event_name]['last_position'] = j + 2
        return encoded_vector

    def get_state(self, event_name, labels):
        # this method gives the state for the current prefix
        best_cluster_value, best_cluster = min([(labels[x], x) for x in self.last_a_to_cluster[event_name]],
                                               key=lambda x: x[0])
        state = '<' + event_name + " | " + str(best_cluster) + '>'
        return state

    def get_possibilities(self, prefix):
        last_activity = prefix[-1]  # I do not need to remove it from prefix because it is done already in encode_prefix
        encoded_vector = self.encode_vector(prefix)
        labels = self.kmean.transform([encoded_vector])[0]
        state = self.get_state(last_activity, labels)
        filtered_policy_df = self.policy_df.loc[(self.policy_df['policy'] == 1) & (self.policy_df['s'] == state)]
        recommended_actions_list = filtered_policy_df['a'].drop_duplicates().to_list()
        return recommended_actions_list

    def get_recommendation(self, prefix):
        recommended_actions_list = self.get_possibilities(prefix)
        next_action = random.choices(recommended_actions_list)[0]  # in the rare case of multiple action select one randomly
        return next_action


def compute_reward(completed_trace, total_duration, amount):
    # thi method computes the reward based on total trace duration (without waiting times) and amount of the loan
    if 'ERRORE' in completed_trace:
        return -1
    else:
        time_cost_factor = -0.01
        loan_amount_percentage = 0.15
        reward = time_cost_factor * total_duration
        if 'O_ACCEPTED' in completed_trace or 'O_Accepted' in completed_trace:
            reward += amount * loan_amount_percentage
        return reward





