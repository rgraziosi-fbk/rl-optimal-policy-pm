import csv
import numpy as np
import simpy
from loan_process import Process
from MAINparameters import *
import time
import warnings
import pandas as pd
from merge_token import MergeToken
from cluster_recommender_model_class import cluster_model
from cql_recommender.cql_agent import CQL_agent
from merge_token_dfg2 import MergeTokenDFG
import sys, getopt

def setup(env: simpy.Environment, NAME_EXPERIMENT, N_TRACE, type, rare, path_model, policy, last_position=True, starting_at=0):
    simulation_process = Process(env=env)
    f = open('output/' + NAME_EXPERIMENT + '.csv', 'w')
    writer = csv.writer(f, delimiter=';')
    writer.writerow(['caseid', 'task', 'start_timestamp', 'end_timestamp', 'resource', 'amount'])

    f1 = open('output/reward_' + NAME_EXPERIMENT + '.csv', 'w')
    writer2 = csv.writer(f1, delimiter=';')
    writer2.writerow(['traces', 'total_duration', 'amount', 'reward', 'reward_start_end', 'len_trace'])
    if type == 'cql':
        recommender_model = CQL_agent(path_model)
    elif type == 'cluster':
        recommender_model = cluster_model(path_model, policy, last_position)
    else:
        recommender_model = None
    for i in range(0, N_TRACE):
        interval = np.random.exponential(scale=1003, size=1)[0]
        yield env.timeout(interval)
        if 'synthetic' in type:
            env.process(MergeToken(MERGE, i, simulation_process, rare).simulation(env, writer, writer2))
        else:
            env.process(MergeTokenDFG(i, simulation_process, recommender_model, rare, starting_at).simulation(env, writer, writer2))  # original

def main(argv):
    opts, args = getopt.getopt(argv, "h:t:o:n:m:r:p:")
    N_TRACE = 1000 ## default value
    NAME_EXPERIMENT = 'simulation'
    rare = None
    path_model = None
    policy = None
    last_position = True
    starting_at = 0
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -t <[s,s_rare,dq,c]> -o <output_file_name> -n_trace <total number of traces>')
            sys.exit()
        elif opt == "-t":
            type = arg
        elif opt == "-m":
            path_model = arg
        elif opt == "-o":
            NAME_EXPERIMENT = arg
        elif opt == "-n":
            N_TRACE = int(arg)
        elif opt == "-r":
            rare = bool(arg)
        elif opt == "-p":
            policy = arg
        elif opt == "-l":
            last_position = arg
        elif opt == "-x":
            starting_at = arg

    start_time = time.time()
    # Create an environment and start the setup process
    env = simpy.Environment()
    env.process(setup(env, NAME_EXPERIMENT, N_TRACE, type, rare, path_model, policy, last_position, starting_at))

    # Execute!
    env.run(until=SIM_DURATION)
    print("--- %s seconds ---" % (time.time() - start_time))


def main_explicit_arguments(type, path_model=None, NAME_EXPERIMENT='simulation', N_TRACE=1000, rare=None, policy=None, last_position=True, starting_at=0):
    start_time = time.time()
    # Create an environment and start the setup process
    env = simpy.Environment()
    env.process(setup(env, NAME_EXPERIMENT, N_TRACE, type, rare, path_model, policy, last_position, starting_at))

    # Execute!
    env.run(until=SIM_DURATION)
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    print(sys.argv[1:])
    main(sys.argv[1:])