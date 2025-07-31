import numpy as np
import d3rlpy
import pm4py

class DatasetManager:
    PAD_ACTIVITY = 'PADDING'

    def __init__(self, log_path, env_activities=[], activity_key='concept:name',
                 trace_key='case:concept:name', reward_key='kpi:reward') -> None:
        """
        Initialize the DatasetManager object.

        Parameters:
        - log_path (str): The path to the log file.
        - env_activities (list): List of environment activities.
        - activity_key (str): The key for the activity column in the log.
        - trace_key (str): The key for the trace column in the log.
        - reward_key (str): The key for the reward column in the log.
        """
        self.log_path = log_path
        self.env_activities = env_activities
        self.activity_key = activity_key
        self.trace_key = trace_key
        self.reward_key = reward_key

        # load log
        self.log = pm4py.read_xes(log_path)

        # list activities
        self.activities = self.log[activity_key].unique().tolist()
        self.activities.sort()
        self.activities.append(self.PAD_ACTIVITY) # Add PADDING activity

        # move env_activities to the end of the list of activities
        for activity in self.env_activities:
            self.activities.remove(activity)
            self.activities.append(activity)

        self.activity2n = { a: n for n, a in enumerate(self.activities) }


    def build_offline_dataset(self, window_size=20, num_traces=-1, save_to_file=False):
        """
        Builds an offline dataset for reinforcement learning.

        Args:
            window_size (int): The size of the sliding window used to create the dataset. Defaults to 20.
            num_traces (int): The number of traces to include in the dataset. Defaults to -1, which includes all traces.
            save_to_file (bool): Whether to save the dataset to a file. Defaults to False.

        Returns:
            d3rlpy.dataset.MDPDataset: The built offline dataset.
        """
        observations, actions, rewards, terminals = [], [], [], []

        # compute list of traces
        traces = list(self.log.groupby(self.trace_key).groups.values())

        # if specified, keep only the specified number of traces
        if num_traces > 0:
            traces = traces[:num_traces]

        # for each trace
        for t in traces:
            trace_start_i = t[0]

            # for each prefix
            for trace_end_i in t[1:]:
                trace_start_i = max(trace_end_i - window_size, trace_start_i)

                # get prefix
                prefix = self.log.iloc[trace_start_i:trace_end_i][self.activity_key]
                prefix = [self.activity2n[a] for a in prefix.values] # convert to indexes
                prefix = [self.activity2n[self.PAD_ACTIVITY]] * (window_size - len(prefix)) + prefix # prepend padding if needed
                prefix = np.identity(len(self.activities))[prefix].tolist() # convert to one-hot
                
                # get next action
                next_action = self.activity2n[self.log.iloc[trace_end_i][self.activity_key]]

                # get reward for next action
                reward = self.log.iloc[trace_end_i][self.reward_key]

                # add obtained info
                observations.append(prefix)
                actions.append(next_action)
                rewards.append(reward)

                # is this the last action?
                if trace_end_i == t[-1]:
                    terminals.append(1.0)
                    break
                else:
                    terminals.append(0.0)

        
        assert len(observations) == len(actions) == len(rewards) == len(terminals)
        
        # collapse environment activities
        env_activities = [self.activity2n[a] for a in self.env_activities]
        num_transitions = len(observations)

        for i in reversed(range(num_transitions)):
            if actions[i] in env_activities:
                rewards[i-1] += rewards[i] # cumulate rewards
                terminals[i-1] = terminals[i]

                observations.pop(i)
                actions.pop(i)
                rewards.pop(i)
                terminals.pop(i)

        # convert to numpy
        observations = np.array(observations)
        # observations = observations.reshape((observations.shape[0], -1)) # to have a 1D state (for using FC network)
        actions = np.array(actions)
        rewards = np.array(rewards)
        terminals = np.array(terminals)

        # build offline RL dataset
        dataset = d3rlpy.dataset.MDPDataset(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            action_space=d3rlpy.ActionSpace.DISCRETE,
            action_size=len(self.activities)-len(self.env_activities)-1, # env_activities and PADDING are not possible actions
        )

        # dump dataset if needed
        if save_to_file:
            with open('dataset.h5', 'w+b') as f:
                dataset.dump(f)

        return dataset
    

    def get_activity_mapping(self):
        """
        Returns the mapping of activities to indices in the dataset.

        Returns:
            dict: A dictionary mapping activities names to their corresponding indices.
        """
        return self.activity2n
    

    def get_reverse_activity_mapping(self):
        """
            Returns the mapping of indices to activity names in the dataset.

            Returns:
                dict: A dictionary mapping activities names to their corresponding indices.
        """
        return { n: a for a, n in self.activity2n.items() }
    

    def get_action_masks(self, percentile=None):
        """
        Returns a dictionary of action masks based on the directly following activities in the log.

        Parameters:
        - percentile (float): The percentile value used to remove rare transitions. If None, no filtering is applied.

        Returns:
        - actions_masks (dict): A dictionary where the keys are activity indices and the values are lists of activity indices that directly follow the corresponding key activity.
        """
        # compute graph of directly following activities
        dfg, _, _ = pm4py.discover_dfg(self.log)

        # compute, for each activity, number of occurrencies of each next activity
        occurrencies = { a: [] for a in self.activities }
        for k, v in dfg.items():
            occurrencies[k[0]].append(v)

        # remove rare transitions
        if percentile is not None:
            dfg = { k: v for k, v in dfg.items() if v >= np.percentile(occurrencies[k[0]], percentile) }
        
        # build action masks dictionary
        actions_masks = { n: [] for n in self.activity2n.values() if n != self.activity2n[self.PAD_ACTIVITY] }

        for t in list(dfg.keys()):
            actions_masks[self.activity2n[t[0]]].append(self.activity2n[t[1]])

        return actions_masks
