from datetime import timedelta
import simpy
import random
import numpy as np
from scipy.stats import lognorm, truncnorm, expon
from cluster_recommender_model_class import compute_reward
from utility import *
import pickle
from simpy.events import AllOf
import json
from numpy.random import choice


class MergeTokenDFG(object):

    def __init__(self, id, params, process, prefix, type, writer_sim, writer_reward, parallel_object, NAME_EXPERIMENT, recommender_model=None, starting_at=2, start_act='START', end_act='END', amount=None, loanGoal=None):
        self._id = id
        self._params = params
        self._process = process
        self._prefix = prefix
        self._prefix_time = 0
        self._prefix_time_with_agent = 0
        self._type = type
        self._writer_sim = writer_sim
        self._writer_reward = writer_reward
        self._parallel_object = parallel_object
        self._buffer = Buffer(writer_sim, self._params.TRACE_ATTRIBUTES)
        self._start_time = params.START_SIMULATION
        self.more_offer_same_time = False
        self._recommender_model = recommender_model
        self._starting_at = starting_at
        self._errors = 0
        self._start_act = start_act
        self._actual_node = params.ACT_TO_NODE[start_act] if start_act == 'START' else start_act
        self._internal_prefix = [self._actual_node]
        self._end_act = end_act

        self._NAME_EXPERIMENT = NAME_EXPERIMENT
        if self._NAME_EXPERIMENT == 'BPIC12_parallel':
            self._amount = amount if self._parallel_object else self.generate_amount_BPIC2012()
            self._loanGoal = None
        else:
            self._amount = amount if self._parallel_object else self.generate_amount_BPIC2017()
            self._loanGoal = loanGoal if self._parallel_object else self.generate_goal_BPIC2017()

    def generate_amount_BPIC2012(self):
        return round(np.random.exponential(scale=14451, size=1)[0])

    def generate_goal_BPIC2017(self):
        prob = [0.095, 0.243, 0.297, 0.177, 0.033, 0.075, 0.012, 0.02, 0.009, 0.006, 0.026, 0.001, 0.005, 0.001]
        goal = ['Other, see explanation', 'Home improvement', 'Car', 'Existing loan takeover', 'Not speficied',
                'Unknown',
                'Caravan / Camper', 'Extra spending limit', 'Motorcycle', 'Boat', 'Remaining debt home',
                'Business goal',
                'Tax payments', 'Debt restructuring']
        return random.choices(goal, prob)[0]

    def generate_amount_BPIC2017(self):
        return round(np.random.exponential(16224 * 2, 1)[0])

    def define_processing_time(self, act):
        ### call the RF, put all the encoding
        if self._params.PROCESSING_TIME[act]["name"] == 'lognorm':
            mean = self._params.PROCESSING_TIME[act]["parameters"]["mean"]
            variance = self._params.PROCESSING_TIME[act]["parameters"]["std"]
            min_val = self._params.PROCESSING_TIME[act]["parameters"]["min"]
            max_val = self._params.PROCESSING_TIME[act]["parameters"]["max"]
            sigma = np.sqrt(np.log(1 + (variance / mean ** 2)))
            mu = np.log(mean) - 0.5 * sigma ** 2

            def truncated_lognorm(mu, sigma, min_val, max_val, size=1000):
                a, b = (np.log(min_val + 1e-9) - mu) / sigma, (np.log(max_val) - mu) / sigma
                samples = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=size)
                return np.exp(samples)

            duration = truncated_lognorm(mu, sigma, min_val, max_val, size=1)[0]
        elif self._params.PROCESSING_TIME[act]["name"] == "exponential":
            scale = self._params.PROCESSING_TIME[act]["parameters"]["scale"]
            min_val = self._params.PROCESSING_TIME[act]["parameters"]["min"]
            max_val = self._params.PROCESSING_TIME[act]["parameters"]["max"]

            def truncated_exponential_inverse(scale, min_val, max_val, size=1000):
                cdf_min = expon.cdf(min_val, scale=scale)
                cdf_max = expon.cdf(max_val, scale=scale)
                u = np.random.uniform(cdf_min, cdf_max, size=size)
                return expon.ppf(u, scale=scale)

            duration = truncated_exponential_inverse(scale, min_val, max_val, size=1)[0]
        elif self._params.PROCESSING_TIME[act]["name"] == 'norm':
            mean = self._params.PROCESSING_TIME[act]["parameters"]["mean"]
            std = self._params.PROCESSING_TIME[act]["parameters"]["std"]
            min_val = self._params.PROCESSING_TIME[act]["parameters"]["min"]
            max_val = self._params.PROCESSING_TIME[act]["parameters"]["max"]
            a = (min_val - mean) / std
            b = (max_val - mean) / std
            duration = truncnorm.rvs(a, b, loc=mean, scale=std)
        else:
            distribution = self._params.PROCESSING_TIME[act]['name']
            parameters = self._params.PROCESSING_TIME[act]['parameters']
            duration = getattr(np.random, distribution)(**parameters, size=1)[0]
            if duration < 0:
                print("WARNING: Negative processing time", duration)
                duration = 0

        return duration

    def define_role(self, list_roles):
        role = list_roles
        if isinstance(list_roles, list):
            free_role = []
            for res in list_roles:
                if self._process._get_resource(res)._resource_simpy.count < self._process._get_resource(res)._capacity:
                    free_role.append(res)
            if not free_role:
                role = random.choice(list_roles)
            else:
                role = random.choice(free_role)
        return role

    def define_next_activities_simulation_parallel(self, env):
        if self._actual_node == self._end_act:
            return self._actual_node
        else:
            next = self._params.DFG[self._actual_node]
            if len(next) == 1:
                self._actual_node = next[0]  ### update position in DFG
                node_name = self._params.BPMN_ELEMENTS[next[0]]
                if 'Gateway' in node_name:
                    self._actual_node = next[0]  ### update position in DFG
                    self._internal_prefix.append(self._actual_node)
                    return self.define_next_activities_simulation_parallel(env)
                else:
                    node_name = self._params.ACTIVITIES[self._actual_node]
                    self._internal_prefix.append(self._actual_node)
                    return node_name
            else:
                #### decision point with multiple paths
                if self._params.BPMN_ELEMENTS[self._actual_node] == 'exclusiveGateway':
                    if not self._params.DT_decision_points:
                        choice_next = self.compute_probability_decision_point()
                    else:
                        choice_next = self.compute_DT_decision_point()
                    #choice_next = self.compute_probability_decision_point()
                    self._actual_node = choice_next  ### update position in DFG
                    self._internal_prefix.append(self._actual_node)
                    return self._params.ACTIVITIES[self._actual_node] if self._actual_node in self._params.ACTIVITIES else self.define_next_activities_simulation_parallel(env)
                else:
                    paths = self._params.DFG[self._actual_node]
                    next_paths = []
                    end_parallel = self._params.AND_node_termination[paths[0]]
                    for p in paths:
                        token = env.process(MergeTokenDFG(self._id, self._params, self._process, self._prefix, 'simulation',
                                                          self._writer_sim, self._writer_reward, True, self._NAME_EXPERIMENT,
                                                          self._recommender_model, start_act=p,
                                                          end_act=end_parallel, amount=self._amount,
                                                          loanGoal=self._loanGoal).simulation(env))
                        next_paths.append(token)
                    self._actual_node = end_parallel
                    next_paths.append(self.define_next_activities_simulation_parallel(env))
                    return next_paths

    def define_next_activities_parallel(self, env):
        if self._actual_node == self._end_act:
            return self._actual_node
        else:
            next = self._params.DFG[self._actual_node]
            if len(next) == 1:
                self._actual_node = next[0]  ### update position in DFG
                node_name = self._params.ACTIVITIES[next[0]] if next[0] in self._params.ACTIVITIES else 'Gateway'
                if 'Gateway' in node_name:
                    self._actual_node = next[0]  ### update position in DFG
                    self._internal_prefix.append(self._actual_node)
                    return self.define_next_activities_parallel(env)
                elif node_name in self._params.ENV_activties or len(self._prefix.get_prefix()) < self._starting_at:
                    #### activity decide by the env
                    self._internal_prefix.append(self._actual_node)
                    return node_name
                else:
                    #### action decide by agent
                    next_agent = self._recommender_model.get_recommendation(
                        self._prefix.get_prefix().copy())
                    last_act = list(self._prefix.get_prefix())[-1]
                    if next_agent not in self._params.DFG_only_task[last_act]:
                        self._errors += 1
                        print('***************** ERRORE_' + next_agent, 'after', node_name, ' *****************')
                    self._internal_prefix.append(self._actual_node)
                    return node_name
            else:
                #### decision point with multiple paths
                if self._params.BPMN_ELEMENTS[self._actual_node] == 'exclusiveGateway':
                    if not self._params.DT_decision_points:
                        choice_next = self.compute_probability_decision_point()
                    else:
                        choice_next = self.compute_DT_decision_point()
                    self._internal_prefix.append(self._actual_node)
                    if choice_next in self._params.ACTIVITIES:
                        node_name = self._params.ACTIVITIES[choice_next]
                        if node_name in self._params.ENV_activties or len(self._prefix.get_prefix()) < self._starting_at:
                            self._internal_prefix.append(self._actual_node)
                            self._actual_node = choice_next
                            return node_name
                        else:
                            next_agent = self._recommender_model.get_recommendation(self._prefix.get_prefix().copy())
                            last_act = list(self._prefix.get_prefix())[-1]
                            if next_agent not in self._params.DFG_only_task[last_act]:
                                self._errors += 1
                                print('***************** ERRORE ' + next_agent, 'after', last_act, ' *****************')
                                self._internal_prefix.append(self._actual_node)
                                self._actual_node = choice_next
                            else:
                                self._actual_node = self._params.ACT_TO_NODE[next_agent]
                                self._internal_prefix.append(self._actual_node)
                                node_name = next_agent
                            return node_name
                    else:
                        self._actual_node = choice_next
                        return self.define_next_activities_parallel(env)
                else:
                    paths = self._params.DFG[self._actual_node]
                    next_paths = []
                    end_parallel = self._params.AND_node_termination[paths[0]]
                    for p in paths:
                        token = env.process(MergeTokenDFG(self._id, self._params, self._process, self._prefix, 'simulation',
                                                          self._writer_sim, self._writer_reward, True, self._NAME_EXPERIMENT,
                                                          self._recommender_model, start_act=p,
                                                          end_act=end_parallel, amount=self._amount,
                                                          loanGoal=self._loanGoal).simulation(env))
                        next_paths.append(token)
                    self._actual_node = end_parallel
                    next_paths.append(self.define_next_activities_simulation_parallel(env))
                    return next_paths

    def compute_probability_decision_point(self):
        gateway = self._params.PROBABILITY[self._actual_node]
        population = [p['path_id'] for p in gateway]
        weights = [t['value'] for t in gateway]
        choice_next = random.choices(population=population, weights=weights, k=1)[0]
        choice_next = self._params.SEQUENCE_FLOW_TARGET[choice_next]
        return choice_next

    def compute_DT_decision_point(self):
        path_decision_json = self._params.path_decision_tree + self._NAME_EXPERIMENT + '_decision_points.json'
        with open(path_decision_json) as file:
            data = json.load(file)
        if data[self._actual_node]["prediction"]:
            path_model = self._params.path_decision_tree + self._actual_node + '.pkl'
            loaded_dt = pickle.load(open(path_model, 'rb'))
            PAD = data['PAD']
            WINDOW_SIZE = 30
            elements2n = data['elements2n']
            cut_prefix = self._internal_prefix[-WINDOW_SIZE:]
            encoding = [elements2n[e] for e in cut_prefix] + [PAD]*(WINDOW_SIZE - len(cut_prefix)) + [self._amount]
            prob = list(loaded_dt.predict_proba(np.array(encoding).reshape(1, -1)))
            predicted = choice(loaded_dt.classes_, 1, prob)[0]
            return predicted
        else:
            population = list(data[self._actual_node]['probabilities'].keys())
            weights = list(data[self._actual_node]['probabilities'].values())
            choice_next = random.choices(population=population, weights=weights, k=1)[0]
            choice_next = self._params.SEQUENCE_FLOW_TARGET[choice_next]
            return choice_next

    def simulation(self, env: simpy.Environment):
        if self._parallel_object and self._actual_node in self._params.ACTIVITIES:
            trans = self._params.ACTIVITIES[self._actual_node]
        else:
            trans = self.define_next_activities_simulation_parallel(env) if self._type == 'simulation' else self.define_next_activities_parallel(env)
        while trans != self._end_act:
            if type(trans) == list:  ### check parallel
                yield AllOf(env, trans[:-1])
                trans = trans[-1]
            if trans != self._end_act:
                self._buffer.reset()
                self._buffer.set_feature("id_case", self._id)
                self._buffer.set_feature("activity", trans)
                self._buffer.set_feature("prefix", self._prefix.get_prefix())
                name_res = self.define_role(self._params.ACT_TO_ROLE[trans])
                resource = self._process._get_resource(name_res)
                self._buffer.set_feature("role", resource._get_name())

                ### register event in process ###
                resource_task = self._process._get_resource_event(trans)
                self._buffer.set_feature("enabled_time", self._start_time + timedelta(seconds=env.now))

                request_resource = resource.request()
                yield request_resource
                single_resource = self._process._set_single_resource(resource._get_name())
                self._buffer.set_feature("resource", single_resource)

                resource_task_request = resource_task.request()
                yield resource_task_request

                ## calendars
                stop = resource.to_time_schedule(self._start_time + timedelta(seconds=env.now))
                yield env.timeout(stop)
                self._buffer.set_feature("start_time", self._start_time + timedelta(seconds=env.now))
                duration = self.define_processing_time(trans)
                self._prefix_time += duration
                yield env.timeout(duration)
                if len(self._prefix.get_prefix()) >= self._starting_at:
                    # this is the execution time computed only from the point in which the agent is enabled
                    self._prefix_time_with_agent += duration
                self._buffer.set_feature("end_time", self._start_time + timedelta(seconds=env.now))
                self._buffer.set_feature("AMOUNT_REQ", str(self._amount))
                if self._NAME_EXPERIMENT == 'BPIC17_parallel':
                    self._buffer.set_feature("LoanGoal", str(self._loanGoal))
                self._buffer.print_values()
                self._prefix.add_activity(trans)
                resource.release(request_resource)
                self._process._release_single_resource(resource._get_name(), single_resource)
                resource_task.release(resource_task_request)
                trans = self.define_next_activities_simulation_parallel(env) if self._type == 'simulation' else self.define_next_activities_parallel(env)
        if not self._parallel_object:
            self._writer_reward.writerow([self._prefix.get_prefix(), self._prefix_time, self._amount, compute_reward(self._prefix.get_prefix(), self._prefix_time, self._amount), len(self._prefix.get_prefix()), self._errors])







