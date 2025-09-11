import csv
import simpy
from process import SimulationProcess
import time
from cluster_recommender_model_class import cluster_model
from cql_recommender.cql_agent import CQL_agent
from dfg_token import MergeTokenDFG
from parameters import Parameters
from inter_trigger_timer import InterTriggerTimer
from utility import *
import os
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")


def setup(env: simpy.Environment, NAME_EXPERIMENT, N_TRACES, type, rare, path_model, policy, last_position=True, starting_at=0, DT_decision_points=False, i_sim=0):
    f = open(f"output/{NAME_EXPERIMENT}/{NAME_EXPERIMENT}_{type}_{policy}_DT_{DT_decision_points}_{i_sim}.csv", 'w')
    writer_sim = csv.writer(f, delimiter=';')
    writer_sim.writerow(['caseid', 'task', "role", "enabled_time", 'start_timestamp', 'end_timestamp', 'resource', "prefix", 'AMOUNT_REQ', 'loanGoal'])
    f1 = open(f"output/{NAME_EXPERIMENT}/reward_{NAME_EXPERIMENT}_{type}_{policy}_DT_{DT_decision_points}_{i_sim}.csv", "w")
    writer_reward = csv.writer(f1, delimiter=';')
    writer_reward.writerow(['traces', 'total_duration', 'amount', 'reward', 'len_trace', '#errors'])
    if type == 'cql':
        recommender_model = CQL_agent(path_model)
    elif type == 'simulation':
        recommender_model = 'simulation'
    else:
        recommender_model = cluster_model(path_model, policy, last_position)
    path_parameters = 'input/' + NAME_EXPERIMENT + '/input_' + NAME_EXPERIMENT + '.json'
    path_DFG = 'input/' + NAME_EXPERIMENT + '/' + NAME_EXPERIMENT + '_DFG.json'
    params = Parameters(path_parameters, path_DFG, N_TRACES, DT_decision_points)
    simulation_process = SimulationProcess(env=env, params=params)
    interval = InterTriggerTimer(params, simulation_process, params.START_SIMULATION)
    for i in range(0, N_TRACES):
        prefix = Prefix()
        parallel_object = False
        itime = interval.get_next_arrival(env, i)
        yield env.timeout(itime)
        env.process(MergeTokenDFG(i, params, simulation_process, prefix, type, writer_sim, writer_reward, parallel_object, NAME_EXPERIMENT,
                                  recommender_model).simulation(env))


def main_explicit_arguments(type, n_simulation=1, path_model=None, NAME_EXPERIMENT='simulation', N_TRACE=1000, rare=None, policy=None, last_position=True, starting_at=0, DT_decision_points=False):
    start_time = time.time()
    for i in range(0, n_simulation):
        env = simpy.Environment()
        env.process(setup(env, NAME_EXPERIMENT, N_TRACE, type, rare, path_model, policy, last_position, starting_at, DT_decision_points, i))
        env.run()
    print("--- %s seconds ---" % (time.time() - start_time))


'''
main_explicit_arguments:
    type: 'rl' for MDP-based RL or 'cql' for Offline Deep RL
    n_simulation: number of simulations to be performed
    path_model: path for the  MDP-based RL model(RL) and for Offline Deep RL(cql), in this repository the models are in "real_log_models_evaluation" folder
    NAME_EXPERIMENT: 'BPIC12_parallel' or 'BPIC17_parallel'
    N_TRACE: number of traces for each simulation (2618 for BPIC12_parallel and 5624 for BPIC17_parallel)
    rare: parameters for the synthetic logs evaluation
    policy: MDP-based RL model(RL) can be "none", "expQ1", "expQ2", "expQ3", "stepQ1", "stepQ2", "stepQ3" for CQL set to "None"
    starting_at: after which prefix position to start recommending
    DT_decision_points: True for using the Decison Trees to predict the next path from a gateway or False for using the probability setting in the file 'input_<NAME_EXPERIMENT>.json'
    
    Example for rl method (MDP-based RL) with policy expQ2
        main_explicit_arguments(type='rl', n_simulation=1, path_model=path_model_rl, NAME_EXPERIMENT='BPIC12_parallel', N_TRACE=2618, rare=None, policy='expQ2', last_position=True, starting_at=0, DT_decision_points=True)
    Example for cql method (Offline Deep RL)
        main_explicit_arguments(type='cql', n_simulation=1, path_model=path_model_rl, NAME_EXPERIMENT='BPIC12_parallel', N_TRACE=2618, rare=None, policy='None', last_position=True, starting_at=0, DT_decision_points=DT_decision_points)
'''

path_model_rl = os.getcwd() + "/real_log_models_evaluation/bpi17_RL_models/"
path_model_cql = os.getcwd() + "/real_log_models_evaluation/bpi_2017_CQL/"
DT_decision_points = True
main_explicit_arguments(type='rl', n_simulation=1, path_model=path_model_rl, NAME_EXPERIMENT='BPIC17_parallel', N_TRACE=10, rare=None, policy='none', last_position=True, starting_at=0, DT_decision_points=DT_decision_points)