'''
Principal parameters to run the process

DA AGGIUNGERE: configurazione risorse, tempi per ogni attivita'
'''
from datetime import datetime

# time in seconds between each token arrival
INTER_TRIGGER_TIMER = 3600
SIM_DURATION = 3154**22
START_SIMULATION = datetime(year=2022, month=1, day=1, hour=8, minute=0, second=0)
# Path for each petrinet
MERGE = 'Petrinet/merge_model.pnml'

DISTRIBUTION_AMOUNT = [15163.62, 12325.49, 2000, 1000000]

PROBABILITY = {
    'exi_Gateway_firstSplit': [0.29, 0.7, 0.01],# [3a9971ce-6672-468d-b17d-378d6c1762e4, 68be6ce9-74b4-497d-b657-a234f8ef325e, W_Assess_fraud]
    'exi_Gateway_preaccepted_rare': [0.60, 0.40], #to do preaccepted ok (RARE)   #[71fa5198-3229-4a9f-ba4c-cae44fd46931, A_PREACCEPTED]
    'exi_Gateway_preaccepted': [0.40, 0.60], #(NORMAL)  #[71fa5198-3229-4a9f-ba4c-cae44fd46931, A_PREACCEPTED]
    'exi_Gateway_LoopPreaccepted': [0.60, 0.40], #to not do loop ok  #[471ebfe2-fac2-485e-b47f-7f2d1d793013, 6c1f3a09-da79-4e4a-b6f7-b888afca526f]
    'exi_Gateway_accepted': [0.30, 0.70], #to do Accepted ok  #[3b7814b2-f9f8-42a2-8484-e5a17e22a3be, A_ACCEPTED]
    'exi_Gateway_LoopWFix': [0.15, 0.85],#to do loop WFIX ok  #[2bca6b50-84c0-456a-bed9-88f7d8636773, f72fa75f-db6c-40fa-8c61-33fea2a617bd]
    'exi_Gateway_AccDecCanc': [0.33, 0.33, 0.34], #[O_ACCEPTED, A_CANCELLED, ae7f4b94-8a80-4f04-a15f-b76215516e4b]
    'exi_Gateway_CallMissingLoop': [0.60, 0.40],#to do loop ok  #[1d08b364-dcb3-4572-bf56-3b78c46cbe56, 88ca0fd6-4a19-42d2-bdfa-50e6ae656bd9]
    'exi_Gateway_AssessLoop': [0.59, 0.01, 0.40],#to do loop ok  #[686efecc-3ff3-49b4-b35d-214caf67ac1e, b4124229-a0ab-4cce-8583-3a197c33f592, b65c7c95-c0e1-4579-a0b8-8f783c1032dd]
    'exi_Gateway_CallMissing': [0.30, 0.70],#to exit to loop ok  #[8db84132-520a-4bfb-9e83-8ecfc9f0574a, ba43745f-533a-4568-b1f6-c5a092fd200b]
    'exi_Gateway_OCancelled': [0.50, 0.50],#prob condizionata, 1 per fare O_CANCELLED  #[6ef7191b-3b1c-43e8-85d6-88d7e6060121, O_CANCELLED]
    'exi_Gateway_SentBack': [0.60, 0.40],#do sent back ok  #[O_SENT_BACK, c8f490e0-6efa-42d7-8543-257cd586b1bc]
    'exi_Gateway_CallLoop': [0.60, 0.40],  #[7eb7d5b6-8b61-402e-9e6a-4aea0d6444c1, e275d463-3268-4eca-97f9-0f01d41da6f9]
    'exi_Gateway_EndProcess1': [0.99, 0.01],  #[a18ba001-19fc-4192-9d38-44ee48ad0f21, c5c9730d-dcb7-4282-8eaf-21aa3d65e4a0]
    'exi_Gateway_CallXor': [0.95, 0.05],#to not do call after offer  #[9c97a049-ea71-4f2f-965b-a31cd1a9c094, f4a4df76-803a-44fa-bda0-843df8ba7e5a]
    'exi_Gateway_SentLoop': [0.95, 0.05],#to remain loop ok  #[c73d9a95-427d-4ea7-ba7a-a6e07add5398, d8641656-f608-4cfc-8cf0-d73479c0a728]
    'exi_Gateway_SentBackLoop': [0.50, 0.50],#special, to define  #[80895713-c8d1-4ed4-bd44-4ae830f8a0b4, ad68b65c-da77-4590-b756-4a2ce60ad1dd]
    'exi_Gateway_ODeclined': [0.50, 0.50]#special to define (pos 1 to do DECLINED)  #[35b865b5-b643-4f4f-b737-003a2b7d9b6d, O_DECLINED]
}

ACTIVITY_TO_GATEWAY = {
    'A_PARTLYSUBMITTED': 'exi_Gateway_firstSplit',
    'W_Fix_incomplete_submission': 'exi_Gateway_LoopWFix',
    'W_Assess_fraud': 'exi_Gateway_preaccepted',
    'W_Complete_preaccepted_appl': 'exi_Gateway_LoopPreaccepted',
    'O_SENT': 'exi_Gateway_SentLoop',
    'W_Call_after_offer': 'exi_Gateway_CallLoop',
    'W_Assess_application': 'exi_Gateway_AssessLoop',
    'W_Call_missing_information': 'exi_Gateway_CallMissingLoop',
    'O_CANCELLED': 'exi_Gateway_SentBackLoop'
}

PROBABILITY_DFG = {
    'exi_Gateway_firstSplit': ([0.29, 0.7, 0.01], ['W_Fix_incomplete_submission', 'exi_Gateway_preaccepted', 'W_Assess_fraud']),
    'exi_Gateway_preaccepted_rare': ([0.60, 0.40], ['exi_Gateway_AccDecCanc', 'A_PREACCEPTED']),
    'exi_Gateway_preaccepted': ([0.40, 0.60], ['exi_Gateway_AccDecCanc', 'A_PREACCEPTED']),
    'exi_Gateway_LoopPreaccepted': ([0.60, 0.40], ['exi_Gateway_accepted', 'W_Complete_preaccepted_appl']),
    'exi_Gateway_accepted': ([0.30, 0.70], ['exi_Gateway_AccDecCanc', 'A_ACCEPTED']),
    'exi_Gateway_LoopWFix': ([0.15, 0.85], ['W_Fix_incomplete_submission', 'exi_Gateway_preaccepted']),
    'exi_Gateway_AccDecCanc': ([0.33, 0.33, 0.34], ['O_ACCEPTED', 'A_CANCELLED', 'exi_Gateway_ODeclined']),
    'exi_Gateway_CallMissingLoop': ([0.60, 0.40], ['exi_Gateway_OCancelled', 'W_Call_missing_information']),
    'exi_Gateway_AssessLoop': ([0.59, 0.01, 0.40], ['exi_Gateway_CallMissing', 'END', 'W_Assess_application']),
    'exi_Gateway_CallMissing': ([0.30, 0.70], ['exi_Gateway_OCancelled', 'W_Call_missing_information']),
    'exi_Gateway_OCancelled': ([0.50, 0.50], ['exi_Gateway_SentBackLoop', 'O_CANCELLED']),  #TODO: check this exi_Gateway_SentBackLoop
    'exi_Gateway_SentBack': ([0.60, 0.40], ['O_SENT_BACK', 'exi_Gateway_OCancelled']),
    'exi_Gateway_CallLoop': ([0.60, 0.40], ['exi_Gateway_EndProcess1', 'W_Call_after_offer']),
    'exi_Gateway_EndProcess1': ([0.99, 0.01], ['exi_Gateway_SentBack', 'END']),
    'exi_Gateway_CallXor': ([0.95, 0.05], ['W_Call_after_offer', 'exi_Gateway_EndProcess1']),
    'exi_Gateway_SentLoop': ([0.95, 0.05], ['exi_Gateway_CallXor', 'O_SELECTED']),
    'exi_Gateway_SentBackLoop': ([0.50, 0.50], ['exi_Gateway_AccDecCanc', 'O_SELECTED']), #TODO: check this
    'exi_Gateway_ODeclined': ([0.50, 0.50], ['A_DECLINED', 'O_DECLINED'])
}

PROCESSING_TIME_VARIABLE = {
    'A_SUBMITTED': [700.0, 0.0],
    'A_APPROVED': [700.0, 0.0],
    'O_SENT_BACK': [1000.0, 'exp'],
    'O_DECLINED': [700.0, 0.0],
    'A_REGISTERED': [700.0, 0.0],
    'W_Assess_fraud': [80.03, 591.67],
    'A_FINALIZED': [700.0, 0.0],
    'O_CREATED': [900.0, 'exp'],
    'A_DECLINED': [700.0, 0.0],
    'O_SELECTED': [200.0, 'exp'],
    'W_Fix_incomplete_submission': [323.9, 1112.71],
    'A_PREACCEPTED': [700.0, 0.0],
    'A_CANCELLED': [700.0, 0.0],
    'O_ACCEPTED': [700.0, 0.0],
    'O_CANCELLED': [1200.0, 'exp'],
    'A_ACCEPTED': [200.0, 'exp'],
    'W_Complete_preaccepted_appl': [369.01, 782.82],
    'A_PARTLYSUBMITTED': [700.0, 0.0],
    'W_Call_after_offer': [214.57, 648.94],
    'O_SENT': [2400.0, 'exp'],
    'A_ACTIVATED': [700.0, 0.0],
    'W_Call_missing_information': [372.2, 951.66],
    'W_Assess_application': [989.32, 1042.45],
}

ROLE_CAPACITY = {
    'SYSTEM': float('inf'),
    'ROLE0': 58,
    'ROLE1': 60,
    'ROLE2': 58,
    'ROLE3': 59,
    'ROLE5': 49,
    'ROLE7': 60,
    'ROLE8': 26,
    'ROLE9': 5,
    'ROLE10': 53,
    'ROLE11': 52,
    'ROLE12': 54,
    'ROLE13': 48
}


ROLE_ACTIVITY = {
    'A_PARTLYSUBMITTED': 'SYSTEM',
    'A_SUBMITTED': 'SYSTEM',
    'A_ACCEPTED': 'ROLE0',
    'A_FINALIZED': 'ROLE0',
    'O_CREATED': 'ROLE0',
    'O_SELECTED': 'ROLE0',
    'O_SENT': 'ROLE0',
    'A_ACTIVATED': 'ROLE1',
    'A_APPROVED': 'ROLE1',
    'A_REGISTERED': 'ROLE1',
    'O_ACCEPTED': 'ROLE1',
    'O_DECLINED': 'ROLE1',
    'W_Assess_application': 'ROLE1',
    'A_CANCELLED': 'ROLE2',
    'A_DECLINED': 'ROLE3',
    'A_PREACCEPTED': 'ROLE5',
    'O_CANCELLED': 'ROLE7',
    'O_SENT_BACK': 'ROLE8',
    'W_Assess_fraud': 'ROLE9',
    'W_Call_after_offer': 'ROLE10',
    'W_Call_missing_information': 'ROLE11',
    'W_Complete_preaccepted_appl': 'ROLE12',
    'W_Fix_incomplete_submission': 'ROLE13',
}

END_ACTIVITIES = ['A_DECLINED', 'A_ACTIVATED', 'A_CANCELLED', 'END']

ENVIRONMENT_ACTIVITIES = ['O_SENT_BACK', 'O_DECLINED', 'O_ACCEPTED', 'A_CANCELLED']

POSSIBLE_ACTIONS = {'A_SUBMITTED': ['A_PARTLYSUBMITTED'],
                    'A_PARTLYSUBMITTED': ['W_Fix_incomplete_submission', 'W_Assess_fraud', 'A_PREACCEPTED', 'A_DECLINED', 'A_CANCELLED'],
                    'W_Fix_incomplete_submission': ['W_Fix_incomplete_submission', 'A_PREACCEPTED', 'A_DECLINED', 'A_CANCELLED'],
                    'W_Assess_fraud': ['A_PREACCEPTED', 'A_DECLINED', 'A_CANCELLED'],
                    'A_PREACCEPTED': ['W_Complete_preaccepted_appl'],
                    'W_Complete_preaccepted_appl': ['W_Complete_preaccepted_appl', 'A_ACCEPTED', 'O_DECLINED', 'A_CANCELLED'],
                    'A_ACCEPTED': ['A_FINALIZED'],
                    'A_FINALIZED': ['O_SELECTED'],
                    'O_SELECTED': ['O_CREATED'],
                    'O_CREATED': ['O_SENT'],
                    'O_SENT': ['O_SELECTED', 'W_Call_after_offer', 'END', 'O_SENT_BACK', 'O_CANCELLED', 'O_DECLINED', 'O_ACCEPTED'],  # non c'è A_CANCELLED perché qui almeno un'offerta è attiva #TODO: rompe qualcosa aver messo END qui dentro?
                    'W_Call_after_offer': ['W_Call_after_offer', 'END', 'O_SENT_BACK', 'O_CANCELLED',  'O_SELECTED', 'O_DECLINED', 'O_ACCEPTED'],  #TODO: rompe qualcosa aver messo END qui dentro?
                    'O_SENT_BACK': ['W_Assess_application'],
                    'W_Assess_application': ['W_Assess_application', 'END', 'W_Call_missing_information', 'O_CANCELLED', 'O_SELECTED', 'O_DECLINED', 'O_ACCEPTED'],  #TODO: rompe qualcosa aver messo END qui dentro?
                    'W_Call_missing_information': ['W_Call_missing_information', 'O_CANCELLED', 'O_SELECTED', 'O_DECLINED', 'O_ACCEPTED'],
                    'O_ACCEPTED': ['A_APPROVED'],
                    'A_APPROVED': ['A_REGISTERED'],
                    'A_REGISTERED': ['A_ACTIVATED'],
                    'O_CANCELLED': ['O_SELECTED', 'O_DECLINED', 'O_ACCEPTED', 'A_CANCELLED'],
                    'O_DECLINED': ['A_DECLINED']}


POSSIBLE_ACTIONS2 = {'A_SUBMITTED': ['A_PARTLYSUBMITTED'],
                     'A_PARTLYSUBMITTED': ['W_Fix_incomplete_submission', 'GATEWAY_A_DECLINED_O_DECLINED', 'A_PREACCEPTED', 'W_Assess_fraud'],
                     'W_Fix_incomplete_submission': ['A_PREACCEPTED', 'GATEWAY_A_DECLINED_O_DECLINED', 'W_Fix_incomplete_submission'],
                     'A_PREACCEPTED': ['W_Complete_preaccepted_appl'],
                     'W_Assess_fraud': ['A_PREACCEPTED', 'GATEWAY_A_DECLINED_O_DECLINED', 'W_Fix_incomplete_submission'],  # TODO: perché 'W_Fix_incomplete_submission', non c'è nel modello nè nel log
                     'W_Complete_preaccepted_appl': ['A_ACCEPTED', 'W_Complete_preaccepted_appl', 'GATEWAY_A_DECLINED_O_DECLINED'],
                     'A_ACCEPTED': ['A_FINALIZED'],
                     'A_FINALIZED': ['O_SELECTED'],
                     'O_SELECTED': ['O_CREATED'],
                     'O_CREATED': ['O_SENT'],
                     'O_SENT': ['W_Call_after_offer', 'O_SELECTED', 'GATEWAY_O_SENT_BACK_O_CANCELLED_O_SELECTED_GATEWAY_A_DECLINED_O_DECLINED'],
                     'W_Call_after_offer': ['GATEWAY_O_SENT_BACK_O_CANCELLED_O_SELECTED_GATEWAY_A_DECLINED_O_DECLINED', 'W_Call_after_offer'],
                     'GATEWAY_O_SENT_BACK_O_CANCELLED_O_SELECTED_GATEWAY_A_DECLINED_O_DECLINED': ['O_SELECTED', 'O_SENT_BACK', 'O_CANCELLED', 'GATEWAY_A_DECLINED_O_DECLINED'],
                     'O_SENT_BACK': ['W_Assess_application'],
                     'W_Assess_application': ['GATEWAY_A_DECLINED_O_DECLINED', 'W_Assess_application', 'W_Call_missing_information', 'O_SELECTED', 'O_CANCELLED'],
                     'W_Call_missing_information': ['GATEWAY_A_DECLINED_O_DECLINED', 'O_CANCELLED', 'W_Call_missing_information', 'O_SELECTED'],
                     'O_CANCELLED': ['GATEWAY_A_DECLINED_O_DECLINED', 'O_SELECTED'],
                     'GATEWAY_A_DECLINED_O_DECLINED': ['O_ACCEPTED', 'A_DECLINED', 'A_CANCELLED', 'O_DECLINED'],
                     'O_ACCEPTED': ['A_APPROVED'],
                     'A_APPROVED': ['A_REGISTERED'],
                     'A_REGISTERED': ['A_ACTIVATED'],
                     'O_DECLINED': ['A_DECLINED']}
