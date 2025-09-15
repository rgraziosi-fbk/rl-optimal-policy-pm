from datetime import timedelta
from MAINparameters import *
import simpy
import random
import numpy as np
from cluster_recommender_model_class import compute_reward

""""
Management of environment/agent priority
In this version at every step a virtual simulation of the next step is performed:
if the virtual simulation leads to an environment activity then this is performed in the actual simulator,
if the virtual simulation leads to an agent activity then the agent (policy) can decide which agent activity to perform,
in this way the environment response rate is reproduced exactly.
"""

class MergeTokenDFG(object):

    def __init__(self, id, process, recommender_model=None, rare=None, starting_at=0):
        self.id = id
        self.process = process
        self.amount = self.generate_amount()
        self.prefix = []
        self.prefix_time = 0
        self.prefix_time_with_agent = 0
        self.more_offer_same_time = False
        self.actual_position = ''
        self.recommender_model = recommender_model
        self.rare = rare
        self.starting_at = starting_at

    @staticmethod
    def generate_amount():
        return random.choices([10000, 20000], [0.50, 0.50])[0]

    @staticmethod
    def get_processing_time(trans):
        if PROCESSING_TIME_VARIABLE[trans][1] == 'exp':
            return round(np.random.exponential(scale=PROCESSING_TIME_VARIABLE[trans][0], size=1)[0])
        else:
            duration = \
            abs(np.random.normal(PROCESSING_TIME_VARIABLE[trans][0], PROCESSING_TIME_VARIABLE[trans][1],
                                 1))[0]
            return round(duration, 0)


    def check_activity(self, activity):
        if activity == 'O_CREATED':
            self.more_offer_activate()

    def more_offer_activate(self):
        if not self.more_offer_same_time:
            if self.prefix.count('O_CREATED') - self.prefix.count('O_CANCELLED') > 1:
                self.more_offer_same_time = True

    def get_statistic(self):
        # '#OCreated', '#OCreated>1', '#MoreOfferTogether', '#ActiveOffer', '#Osentback', '#'O_ACCEPTED','#Odeclined', '#Ocancelled', '#Aaccepted'
        if self.prefix.count('O_CREATED') - self.prefix.count('O_CANCELLED') == 0:
            active_offer = False
        else:
            active_offer = True
        stat = [self.id, bool(self.prefix.count('O_CREATED')),
                True if self.prefix.count('O_CREATED') > 1 else False, self.more_offer_same_time, active_offer,
                bool(self.prefix.count('O_SENT_BACK')),
                bool(self.prefix.count('O_ACCEPTED')), bool(self.prefix.count('O_DECLINED')),
                bool(self.prefix.count("O_CANCELLED")), bool(self.prefix.count('A_ACCEPTED'))]
        return stat


    def define_prob_sent_back(self):
        #BASE 60% to do SENT_BACK ----> [0.4, 0.6]
        base_prob = 60
        if self.amount <= 10000:
            base_prob += 10
        else:
            base_prob -= 20

        match self.prefix.count('O_CREATED'):
            case 0:
                base_prob -= 100
            case 1:
                base_prob -= 0
            case 2:
                base_prob += 10
            case 3:
                base_prob += 20
            case 4:
                base_prob += 0
            case 5:
                base_prob -= 20
            case _:
                base_prob -= 40

        offerte_attive = self.prefix.count('O_CREATED') - self.prefix.count('O_CANCELLED')
        match offerte_attive:
            case 0:
                base_prob -= 100
            case 1:
                base_prob -= 0
            case 2:
                base_prob += 40
            case 3:
                base_prob += 0
            case _:
                base_prob -= 40

        if self.prefix.count('W_Call_after_offer') == 0:
            base_prob -= 10
        elif 4 <= self.prefix.count('W_Call_after_offer') < 7:
            base_prob += 10
        elif self.prefix.count('W_Call_after_offer') > 10:
            base_prob -= 30

        if 0 < self.prefix.count('W_Call_missing_information') < 6:
            base_prob -= 5
        elif 6 <= self.prefix.count('W_Call_missing_information') < 11:
            base_prob -= 10
        elif self.prefix.count('W_Call_missing_information') >= 11:
            base_prob -= 30

        if self.prefix.count('O_SENT_BACK') > 0:
            base_prob -= 60

        return base_prob/100


    def check_enviroment(self):
        if self.actual_position in ['W_Call_after_offer', 'O_SENT', 'W_Assess_application', 'W_Call_missing_information', 'O_CANCELLED']:
            return True
        else:
            return False

    def compute_prob(self, gateway):
        match gateway:
            case 'exi_Gateway_AccDecCanc':
                if (self.prefix.count('O_CREATED')-self.prefix.count('O_CANCELLED')) == 0: #no offerte attive
                    next = random.choices(['A_CANCELLED', self.compute_prob('exi_Gateway_ODeclined')], [0.25, 0.75])[0]  # 0=O_ACCEPTED, 1=A_CANCELLED, 2=g_declined
                else:
                    if self.prefix.count('O_SENT_BACK') > 0:
                        next = random.choices(['O_ACCEPTED', 'O_DECLINED'], [0.9, 0.1])[0]
                    else:
                        next = random.choices(['O_ACCEPTED', 'O_DECLINED'], [0.01, 0.99])[0]
            case 'exi_Gateway_ODeclined':
                if self.prefix.count('A_PREACCEPTED') == 0:
                    next = 'A_DECLINED'
                else:
                    next = 'O_DECLINED'
            case 'exi_Gateway_OCancelled':
                if self.prefix.count('O_SENT_BACK') > 0:
                    next = random.choices([self.compute_prob('exi_Gateway_SentBackLoop'), 'O_CANCELLED'], [0.85, 0.15])[0]  # 0 = next gateway, 1 = O_CANCELLED
                else:
                    next = random.choices([self.compute_prob('exi_Gateway_SentBackLoop'), 'O_CANCELLED'], [0.98, 0.02])[0]
            case 'exi_Gateway_SentBackLoop':
                if (self.prefix.count('O_CREATED')-self.prefix.count('O_CANCELLED')) == 0: # no_loop
                    next = self.compute_prob('exi_Gateway_AccDecCanc')
                else:
                    p = [0.97, 0.03] if self.prefix.count('O_SENT_BACK') > 0 else [0.45, 0.55]
                    next = random.choices([self.compute_prob('exi_Gateway_AccDecCanc'), 'O_SELECTED'], p)[0]
            case 'exi_Gateway_SentBack':
                if self.prefix.count('O_SENT_BACK') == 0:
                    sent_back = 0.60
                else:
                    sent_back = self.define_prob_sent_back()
                next = random.choices(['O_SENT_BACK', self.compute_prob('exi_Gateway_OCancelled')], [sent_back, 1 - sent_back])[0]
            case 'exi_Gateway_preaccepted':
                if self.rare:
                    gateway = 'exi_Gateway_preaccepted' + '_rare'
                prob = PROBABILITY_DFG[gateway][0]
                value = [self.compute_prob(x) if 'Gateway' in x else x for x in PROBABILITY_DFG[gateway][1]]
                next = random.choices(value, prob)[0]
            case _:
                prob = PROBABILITY_DFG[gateway][0]
                value = [self.compute_prob(x) if 'Gateway' in x else x for x in PROBABILITY_DFG[gateway][1]]
                next = random.choices(value, prob)[0]
        return next

    # def check_gateway(self, actions):
    #     for elem in actions:
    #         if 'GATEWAY' in elem:
    #             return elem
    #     return None

    def check_gateway(self):
        if self.actual_position in ACTIVITY_TO_GATEWAY.keys():
            return ACTIVITY_TO_GATEWAY[self.actual_position]
        return None

    def define_next_activities(self):
        if len(self.prefix) == 0:
            self.actual_position = 'A_SUBMITTED'
            return 'A_SUBMITTED'
        elif self.actual_position in END_ACTIVITIES:
            return 'END'
        else:
            next = self.recommender_model.get_recommendation(self.prefix.copy())

            if next not in POSSIBLE_ACTIONS[self.actual_position]:
                return 'ERRORE_' + next
            elif self.prefix.count('O_SELECTED') >= 10:
                return 'ERRORE_infinity'
            elif self.check_gateway() is not None:  # means that there is a gateway
                result = self.compute_prob(self.check_gateway())
                if len(self.prefix) < self.starting_at:
                    # clause to start with the agent only when prefix is longer than starting_at
                    return result  # environment
                elif result in ENVIRONMENT_ACTIVITIES:
                    return result  # environment
                else:
                    return next  # agent
            else:
                return next

    def simulation(self, env: simpy.Environment, writer, writer2):
        trans = self.define_next_activities()
        start = env.now
        while trans != 'END':
            if 'ERRORE' in trans:
                buffer = [self.id, trans, 'None', 'None', 'None', self.amount]
                self.prefix.append(trans)
                print(*buffer)
                trans = 'END'
            else:
                buffer = [self.id, trans]
                self.check_activity(trans)
                self.prefix.append(trans)
                resource = self.process.request_resource(ROLE_ACTIVITY[trans])
                request_resource = resource.request()
                yield request_resource
                processing_time = self.get_processing_time(trans)
                self.prefix_time += processing_time
                if len(self.prefix) >= self.starting_at:
                    # this is the execution time computed only from the point in which the agent is enabled
                    self.prefix_time_with_agent += processing_time
                buffer.append(START_SIMULATION + timedelta(seconds=env.now))
                yield env.timeout(processing_time)
                self.process.release_resource(ROLE_ACTIVITY[trans], request_resource)
                buffer.append(START_SIMULATION + timedelta(seconds=env.now))
                buffer.append(ROLE_ACTIVITY[trans])
                buffer.append(self.amount)
                writer.writerow(buffer)
                print(*buffer)
                self.actual_position = trans
                trans = self.define_next_activities()
        writer2.writerow([self.prefix, self.prefix_time, self.amount, compute_reward(self.prefix, self.prefix_time_with_agent, self.amount), compute_reward(self.prefix, (env.now) - start, self.amount), len(self.prefix)])







