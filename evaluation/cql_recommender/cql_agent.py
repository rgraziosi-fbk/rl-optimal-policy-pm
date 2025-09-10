import glob
import os
import pickle
import d3rlpy
import numpy as np

from cql_recommender.model import LSTMEncoderFactory

class CQL_agent:
    def __init__(self, path) -> None:
        # get config path
        config_path = glob.glob(os.path.join(path, '*.pkl'))[0]

        # unpickle config
        with open(config_path, mode='rb') as config_file:
            config = pickle.load(config_file)
            
            self.window_size = config['window_size']
            self.action_masks = config['action_masks']
            self.activity2n = config['activity2n']
            self.n2activity = { n: a for a, n in self.activity2n.items() }

        # get model path
        model_path = glob.glob(os.path.join(path, '*.d3'))[0]

        # # load model
        # d3rlpy.models.register_encoder_factory(LSTMEncoderFactory)
        # self.model = d3rlpy.load_learnable(model_path)

        # load model
        try:
            d3rlpy.models.register_encoder_factory(LSTMEncoderFactory)
        except AssertionError:
            print("WARNING: LSTMEncoderFactory already registered")
        self.model = d3rlpy.load_learnable(model_path)


    def get_recommendation(self, prefix):
        # prepend START activity to prefix
        prefix.insert(0, 'START')
        # convert from activity names to indices
        prefix = [self.activity2n[a] for a in prefix]

        # trim prefix if longer than window size
        prefix = prefix[-self.window_size:]
        # prepend padding and one-hot encode
        prefix_with_pad = [self.activity2n['PADDING']] * (self.window_size - len(prefix)) + prefix
        prefix_with_pad = np.identity(len(self.activity2n))[prefix_with_pad]
        prefix_with_pad = np.expand_dims(prefix_with_pad, axis=0)

        # get value for each action
        action_values = [self.model.predict_value(prefix_with_pad, np.array([action])) for action in range(0, self.model.action_size)]
        action_values = np.array(action_values)
        action_values = np.squeeze(action_values)

        # apply mask
        prev_action = prefix[-1]
        mask = self.action_masks[prev_action]
        if len(mask) == 0:
            raise Exception(f'No possible actions after action "{self.n2activity[prev_action]}" because of action masking!')
        mask = np.array(mask)
        np.putmask(action_values, np.isin(np.arange(len(action_values)), mask, invert=True), -np.inf)

        # get action with max value
        recommended_action = np.argmax(action_values)

        return self.n2activity[recommended_action]

    def get_possibilities(self, prefix):
        action = self.get_recommendation(prefix)
        return [action]
    