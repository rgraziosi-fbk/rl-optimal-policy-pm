'''
Class for all subprocess to manage the resource and each token
'''
import simpy
from MAINparameters import *


class Process(object):

    def __init__(self, env: simpy.Environment):
        self.env = env
        self.roles = self.role_process()

    def role_process(self):
        roles = dict()
        for key in ROLE_CAPACITY:
            roles[key] = simpy.Resource(self.env, ROLE_CAPACITY[key])
        return roles

    def request_resource(self, role):
        return self.roles[role]

    def release_resource(self, role, request):
        self.roles[role].release(request)

