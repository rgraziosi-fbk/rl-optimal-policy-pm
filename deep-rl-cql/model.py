import torch
import torch.nn as nn
import d3rlpy
import dataclasses

# LSTM
class LSTMEncoder(nn.Module):
    def __init__(self, observation_shape, feature_size, batch_size):
        super().__init__()
        self.feature_size = feature_size
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size=observation_shape[-1],
                            hidden_size=feature_size,
                            batch_first=True)

    def reset(self):
        self.train()
        device = next(self.parameters()).device
        h = torch.zeros(self.lstm.num_layers, self.batch_size, self.feature_size, device=device)
        c = torch.zeros(self.lstm.num_layers, self.batch_size, self.feature_size, device=device)
        self.lstm.hidden = (h, c)

    def forward(self, x):
        # Reset hidden states
        self.reset()

        # Forward through LSTM layer
        o, (h, _) = self.lstm(x)
        
        # Reshape output to (batch_size, feature_size)
        h = h.squeeze(0)

        # return h
        return o[:,-1,:]

@dataclasses.dataclass()
class LSTMEncoderFactory(d3rlpy.models.EncoderFactory):
    feature_size: int
    batch_size: int

    def create(self, observation_shape):
        return LSTMEncoder(observation_shape, self.feature_size, self.batch_size)

    @staticmethod
    def get_type() -> str:
        return "custom"


# Fully Connected
class FCEncoder(nn.Module):
    def __init__(self, observation_shape, feature_size):
        super().__init__()
        self.feature_size = feature_size
        self.fc1 = nn.Linear(observation_shape[0], feature_size)
        self.fc2 = nn.Linear(feature_size, feature_size)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))

        return h
    
@dataclasses.dataclass()
class FCEncoderFactory(d3rlpy.models.EncoderFactory):
    feature_size: int

    def create(self, observation_shape):
        return FCEncoder(observation_shape, self.feature_size)

    @staticmethod
    def get_type() -> str:
        return "custom"
