"""
    Class for reading simulation parameters
"""
import json
import math
import os
from datetime import datetime


class Parameters(object):

    def __init__(self, path_parameters: str, path_DFG: str, traces: int, DT_decision_points: bool):
        self.TRACES = traces
        """TRACES: number of traces to generate"""
        self.PATH_PARAMETERS = path_parameters
        self.read_metadata_file()
        """PATH_PARAMETERS: path of json file for others parameters. """
        self.path_DFG = path_DFG
        self.read_DFG_file()
        """PATH_DFG: path of json file for DFG. """
        self.DT_decision_points = DT_decision_points

    def read_metadata_file(self):
        '''
        Method to read parameters from json file, see *main page* to get the whole list of simulation parameters.
        '''
        if os.path.exists(self.PATH_PARAMETERS):
            with open(self.PATH_PARAMETERS) as file:
                data = json.load(file)
                self.START_SIMULATION = self._check_default_parameters(data, 'start_timestamp')
                self.SIM_TIME = self._check_default_parameters(data, 'duration_simulation')
                self.PROBABILITY = data['gateway_probability'] if 'gateway_probability' in data.keys() else []
                self.WAITING_TIME = data['waiting_time'] if 'waiting_time' in data.keys() else []
                self.INTER_TRIGGER = data["interTriggerTimer"]
                self.PROCESSING_TIME = data['processing_time']
                self.ROLE_ACTIVITY = dict()
                if 'calendar' in data['interTriggerTimer'] and data['interTriggerTimer']['calendar']:
                    self.ROLE_CAPACITY = {'TRIGGER_TIMER': [math.inf, {'days': data['interTriggerTimer']['calendar']['days'], 'hour_min': data['interTriggerTimer']['calendar']['hour_min'], 'hour_max': data['interTriggerTimer']['calendar']['hour_max']}]}
                else:
                    self.ROLE_CAPACITY = {'TRIGGER_TIMER': [math.inf, []]}
                self._define_roles_resources(data['resource'])
                self.ENV_activties = data['ENV_activities']
                self.TRACE_ATTRIBUTES = data['TRACE_ATTRIBUTES']
                self.EVENT_ATTRIBUTES = data['EVENT_ATTRIBUTES']
                self.RES_TO_ROLE = data["resource_to_role"]
                self.ACT_TO_ROLE = data["activity_to_role"]
                self.path_decision_tree = data["path_decision_tree"] if "path_decision_tree" in data else None
                self.AND_node_termination = data["AND_node_termination"]
        else:
            raise ValueError('Parameter file does not exist')

    def _define_roles_resources(self, roles):
        for idx, key in enumerate(roles):
            self.ROLE_CAPACITY[key] = [roles[key]['resources'], {'days': [0, 1, 3, 4, 5, 6], 'hour_min': 0, 'hour_max': 23}]

    def _check_default_parameters(self, data, type):
        if type == 'start_timestamp':
            value = datetime.strptime(data['start_timestamp'], '%Y-%m-%d %H:%M:%S') if type in data else datetime.now()
        elif type == 'duration_simulation':
            value = data['duration_simulation']*86400000 if type in data else 31536000000
        return value

    def read_DFG_file(self):
        with open(self.path_DFG) as file:
            data = json.load(file)
            self.ACTIVITIES = data['activities']
            self.BPMN_ELEMENTS = data["elements_bpmn"]
            self.DFG = data["DFG"]
            self.start_node = data["start_node"]
            self.end_node = data["end_node"]
            self.ACT_TO_NODE = data["act_node"]
            self.SEQUENCE_FLOW_TARGET = data["sequenceFlow_target"]
            self.DFG_only_task = data["DFG_only_task"]