import os

from src.utils.utils import utils_main

# This script preprocesses the logs with following manipulation

# - Splits real logs into training 80% and testing 20% (test sets are used for the discovery of the process model employed in the BPS simulation for the evaluation)
# - Compute event durations for all logs taking into account the lifecycles:transition dynamics: SCHEDULE, START, COMPLETE, SUSPEND, RESUME, WITHDRAWN
# - Compute and add the reward for each prefix in the log

if __name__ == '__main__':
    folder_name = "folder1"
    utils_main(folder_name)  # add duration and reward to logs
