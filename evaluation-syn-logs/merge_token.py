from datetime import timedelta
from MAINparameters import *
import simpy
import pm4py
from pm4py.objects.petri_net import semantics
import random
import numpy as np
from scipy.stats import truncnorm


class MergeToken(object):

    def __init__(self, path_petrinet, id, process, rare=None, starting_at=0):
        self.net, self.am, self.fm = pm4py.read_pnml(path_petrinet)
        self.id = id
        self.process = process
        self.amount = self.generate_amount()
        self.prefix = []
        self.prefix_time = 0
        self.prefix_time_with_agent = 0
        self.more_offer_same_time = False
        self.skip = None
        self.rare = rare
        self.starting_at = starting_at

    def compute_reward(self, completed_trace, total_duration, amount):
        # this method computes the reward based on total trace duration (without waiting times) and amount of the loan
        if 'ERRORE' in completed_trace:
            return -1
        else:
            time_cost_factor = -0.01
            loan_amount_percentage = 0.15
            reward = time_cost_factor * total_duration
            if 'O_ACCEPTED' in completed_trace:
                reward += amount * loan_amount_percentage
        return reward

    def compute_probability(self, gateway, all_enabled_trans):
        all_enabled_trans.sort(key=lambda x: x.name)
        if gateway == 'exi_Gateway_ODeclined':
            print('here')  # debug
        match gateway:
            case 'exi_Gateway_AccDecCanc':
                if (self.prefix.count('O_CREATED')-self.prefix.count('O_CANCELLED')) == 0: #no offerte attive
                    next = int(random.choices([1, 2], [0.25, 0.75])[0])  # 0=O_ACCEPTED, 1=A_CANCELLED, 2=g_declined
                else:
                    if self.prefix.count('O_SENT_BACK') > 0:
                        next = int(random.choices([0, 2], [0.9, 0.1])[0])
                    else:
                        next = int(random.choices([0, 2], [0.01, 0.99])[0])
            case 'exi_Gateway_ODeclined':
                if (self.prefix.count('O_CREATED')-self.prefix.count('O_CANCELLED')) == 0: #no offerte attive
                    next = 0  # A_DECLINED
                else:
                    next = 1  # O_DECLINED
            case 'exi_Gateway_OCancelled':
                if self.prefix.count('O_SENT_BACK') == 0:  # never done O_SENT_BACK
                    next = int(random.choices([0, 1], [0.85, 0.15])[0])  # 0 = next gateway, 1 = O_CANCELLED
                else:
                    next = int(random.choices([0, 1], [0.98, 0.02])[0])  # is O_SENT_BACK has been executed then canceling an offer is less probable
            case 'exi_Gateway_SentBackLoop':
                if (self.prefix.count('O_CREATED')-self.prefix.count('O_CANCELLED')) == 0: # no_loop
                    next = 0
                else:
                    p = [0.97, 0.03] if self.prefix.count('O_SENT_BACK') > 0 else [0.45, 0.55]
                    next = int(random.choices([0, 1], p)[0])  # {0: towards exi_Gateway_AccDecCanc, 1: towards O_SELECTED}
            case 'exi_Gateway_SentBack':
                sent_back = self.define_prob_sent_back()
                next = int(random.choices([0, 1], [sent_back, 1-sent_back])[0]) # {0: O_SENT_BACK, 1: exi_Gateway_OCancelled}
            case 'exi_Gateway_preaccepted':
                if self.rare:
                    gateway = 'exi_Gateway_preaccepted' + '_rare'
                prob = PROBABILITY[gateway]
                value = [*range(0, len(prob), 1)]
                next = int(random.choices(value, prob)[0])
            case _:
                prob = PROBABILITY[gateway]
                value = [*range(0, len(prob), 1)]
                next = int(random.choices(value, prob)[0])
        return all_enabled_trans[next]

    def define_prob_sent_back(self):
        #BASE abbiamo 60% di fare SENT_BACK ----> [0.4, 0.6]
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

    def next_transition(self):
        all_enabled_trans = semantics.enabled_transitions(self.net, self.am)  # am è la posizione attuale
        all_enabled_trans = list(all_enabled_trans)
        label_element = str(list(self.am)[0])
        if len(all_enabled_trans) == 0:
            return None
        elif len(all_enabled_trans) > 1:
            return self.compute_probability(label_element, all_enabled_trans)
        else:
            return all_enabled_trans[0]

    def update_marking(self, trans):
        self.am = semantics.execute(trans, self.net, self.am)

    @staticmethod
    def get_processing_time(trans):
        if PROCESSING_TIME_VARIABLE[trans.label][1] == 'exp':
            processing_time = round(np.random.exponential(scale=PROCESSING_TIME_VARIABLE[trans.label][0], size=1)[0])
        else:
            duration = abs(np.random.normal(PROCESSING_TIME_VARIABLE[trans.label][0], PROCESSING_TIME_VARIABLE[trans.label][1], 1))[0]
            processing_time = round(duration, 0)
        return processing_time if processing_time > 0 else 5

    @staticmethod
    def generate_amount():
        mean = (DISTRIBUTION_AMOUNT[2] - DISTRIBUTION_AMOUNT[0]) / DISTRIBUTION_AMOUNT[1]
        std = (DISTRIBUTION_AMOUNT[3] - DISTRIBUTION_AMOUNT[0]) / DISTRIBUTION_AMOUNT[1]
        amount = truncnorm(mean, std, loc=DISTRIBUTION_AMOUNT[0], scale=DISTRIBUTION_AMOUNT[1]).rvs()
        amount = round(amount/1000, 0)*1000
        return amount

    def check_activity(self, activity):
        if activity == 'O_CREATED':
            self.more_offer_activate()

    def more_offer_activate(self):
        if not self.more_offer_same_time:
            if self.prefix.count('O_CREATED') - self.prefix.count('O_CANCELLED') > 1:
                self.more_offer_same_time = True
                
    def get_statistic(self):
        #'#OCreated', '#OCreated>1', '#MoreOfferTogether', '#ActiveOffer', '#Osentback', '#'O_ACCEPTED','#Odeclined', '#Ocancelled', '#Aaccepted'
        if self.prefix.count('O_CREATED') - self.prefix.count('O_CANCELLED') == 0:
            active_offer = False
        else:
            active_offer = True
        stat = [self.id, bool(self.prefix.count('A_PREACCEPTED')), bool(self.prefix.count('O_CREATED')), True if self.prefix.count('O_CREATED') > 1 else False, self.more_offer_same_time, active_offer, bool(self.prefix.count('O_SENT_BACK')),
                bool(self.prefix.count('O_ACCEPTED')), bool(self.prefix.count('O_DECLINED')), bool(self.prefix.count("O_CANCELLED")), bool(self.prefix.count('A_ACCEPTED'))]
        return stat

    def simulation(self, env: simpy.Environment, writer, writer2):
        trans = self.next_transition()
        start = env.now
        while trans is not None:
            if trans.label is not None:  # altrimenti non è un attività ma un place (sequenziale o gateway)
                buffer = [self.id, trans.label]
                self.check_activity(trans.label)
                self.prefix.append(trans.label)
                resource = self.process.request_resource(ROLE_ACTIVITY[trans.label])
                request_resource = resource.request()
                time_res = env.now
                yield request_resource
                if env.now == time_res:
                    yield env.timeout(1)
                processing_time = self.get_processing_time(trans)
                self.prefix_time += processing_time
                if len(self.prefix) >= self.starting_at:
                    # this is the execution time computed only from the point in which the agent is enabled
                    self.prefix_time_with_agent += processing_time
                buffer.append(START_SIMULATION + timedelta(seconds=env.now))
                yield env.timeout(processing_time)
                self.process.release_resource(ROLE_ACTIVITY[trans.label], request_resource)
                buffer.append(START_SIMULATION + timedelta(seconds=env.now))
                buffer.append(ROLE_ACTIVITY[trans.label])
                buffer.append(self.amount)
                writer.writerow(buffer)
                print(*buffer)
            self.update_marking(trans)  # udpate actual position
            trans = self.next_transition()  # get next activity, gateway
        writer2.writerow([self.prefix, self.prefix_time, self.amount, self.compute_reward(self.prefix, self.prefix_time, self.amount), self.compute_reward(self.prefix, (env.now) - start, self.amount), len(self.prefix)])





