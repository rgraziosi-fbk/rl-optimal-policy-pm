import pickle
import d3rlpy
import torch
from d3rlpy.algos import DQNConfig, DiscreteCQLConfig
from d3rlpy.preprocessing import StandardRewardScaler
from d3rlpy.metrics import TDErrorEvaluator, DiscountedSumOfAdvantageEvaluator, AverageValueEstimationEvaluator, DiscreteActionMatchEvaluator
from d3rlpy.logging import TensorboardAdapterFactory

from model import LSTMEncoderFactory, FCEncoderFactory
from dataset import DatasetManager

# device
DEVICE = 'cuda:0' if torch.cuda.is_available() else None

# set random seed
SEED = 123
d3rlpy.seed(SEED)

# dataset
DATASET = 'event_log_rare_2000_training_single_rewards.xes'
WINDOW_SIZE = 20

dm = DatasetManager(
    log_path=f'datasets/{DATASET}',
    env_activities=['O_ACCEPTED', 'A_CANCELLED', 'O_SENT_BACK', 'O_DECLINED'],
)
dataset = dm.build_offline_dataset(window_size=WINDOW_SIZE)

# save config to file
with open('config.pkl', mode='wb') as config_file:
    config = {
        'window_size': WINDOW_SIZE,
        'action_masks': dm.get_action_masks(percentile=20.0),
        'activity2n': dm.get_activity_mapping(),
    }

    pickle.dump(config, config_file)

# alg
FEATURE_SIZE = 128
BATCH_SIZE = 64

alg = DiscreteCQLConfig(
    encoder_factory=LSTMEncoderFactory(FEATURE_SIZE, BATCH_SIZE),
    batch_size=BATCH_SIZE,
    target_update_interval=32_000,
    gamma=1,
    learning_rate=0.0000625,
    reward_scaler=StandardRewardScaler(),
).create(device=DEVICE)

# init neural networks with the given observation shape and action size
alg.build_with_dataset(dataset)

# print model config
with open('model_config.txt', mode='w') as f:
    print(alg.config, file=f)

# eval metrics
td_error_evaluator = TDErrorEvaluator(episodes=dataset.episodes)
discounted_sum_of_advantage_evaluator = DiscountedSumOfAdvantageEvaluator(episodes=dataset.episodes)
average_value_estimation_evaluator = AverageValueEstimationEvaluator(episodes=dataset.episodes)
discrete_action_match_evaluator = DiscreteActionMatchEvaluator(episodes=dataset.episodes)

# training
NUM_EPOCHS = 100
NUM_STEPS_PER_EPOCH = 150_000
NUM_STEPS = NUM_EPOCHS * NUM_STEPS_PER_EPOCH

SAVE_EVERY_NUM_EPOCHS = 5

def train_callback(alg, epoch, total_step):
    pass

def epoch_callback(algo, epoch, total_step):
    if epoch % SAVE_EVERY_NUM_EPOCHS == 0:
        model_name = f'model_epoch_{epoch}.d3'
        alg.save(model_name)
        print(f'Saved model "{model_name}"')

alg.fit(
    dataset,
    n_steps=NUM_STEPS,
    n_steps_per_epoch=NUM_STEPS_PER_EPOCH,
    evaluators={
        'td_error': td_error_evaluator,
        'discounted_sum_of_advantage': discounted_sum_of_advantage_evaluator,
        'average_value_estimation': average_value_estimation_evaluator,
        'discrete_action_match': discrete_action_match_evaluator,
    },
    callback=train_callback,
    epoch_callback=epoch_callback,
    logger_adapter=TensorboardAdapterFactory(root_dir='logs'),
)

# save final model
alg.save('model_final.d3')
