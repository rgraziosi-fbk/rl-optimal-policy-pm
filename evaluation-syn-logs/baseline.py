class Baseline:
    TRANSITIONS = {
        'A_SUBMITTED': 'A_PARTLYSUBMITTED',
        'A_PARTLYSUBMITTED': 'A_PREACCEPTED',
        'W_Fix_incomplete_submission': 'A_PREACCEPTED',
        'W_Assess_fraud': 'A_PREACCEPTED',
        'A_PREACCEPTED': 'W_Complete_preaccepted_appl',
        'W_Complete_preaccepted_appl': 'A_ACCEPTED',
        'A_ACCEPTED': 'A_FINALIZED',
        'A_FINALIZED': 'O_SELECTED',
        'O_SELECTED': 'O_CREATED',
        'O_CREATED': 'O_SENT',
        'O_SENT_BACK': 'W_Assess_application',
        'O_DECLINED': 'A_DECLINED',
        'O_ACCEPTED': 'A_APPROVED',
        'A_APPROVED': 'A_REGISTERED',
        'A_REGISTERED': 'A_ACTIVATED',
    }

    @staticmethod
    def o_sent(n_active_offers, **_):
        return 'O_SELECTED' if n_active_offers < 2 else 'W_Call_after_offer'

    @staticmethod
    def w_call_after_offer(n_w_call_after_offer, n_o_created, **_):
        if n_w_call_after_offer <= 6:
            return 'W_Call_after_offer'
        elif n_o_created < 3:
            return 'O_CANCELLED'
        else:
            return 'A_DECLINED'

    @staticmethod
    def o_cancelled(n_o_created, **_):
        return 'O_SELECTED' if n_o_created < 3 else 'A_DECLINED'

    @staticmethod
    def w_assess_or_w_call_missing(n_o_created, **_):
        return 'O_CANCELLED' if n_o_created < 3 else 'A_DECLINED'

    TRANSITIONS.update({
        'O_SENT': o_sent,
        'W_Call_after_offer': w_call_after_offer,
        'O_CANCELLED': o_cancelled,
        'W_Assess_application': w_assess_or_w_call_missing,
        'W_Call_missing_information': w_assess_or_w_call_missing,
    })

    def get_recommendation(self, prefix):
        if not prefix:
            raise ValueError("Prefix cannot be empty")

        last_activity = prefix[-1]

        n_o_created = prefix.count('O_CREATED')
        n_active_offers = n_o_created - prefix.count('O_CANCELLED')
        n_w_call_after_offer = prefix.count('W_Call_after_offer')

        next_action = self.TRANSITIONS.get(last_activity)
        if next_action is None:
            raise ValueError(f"Unknown last_activity: {last_activity}")

        if callable(next_action):
            return next_action(
                n_active_offers=n_active_offers,
                n_o_created=n_o_created,
                n_w_call_after_offer=n_w_call_after_offer,
            )

        return next_action













