
class RewardFunction:
    def __init__(self):
        pass

    # To implement...
    def get_reward(self, o1, a, o2):
        raise NotImplementedError("To be implemented")

    def get_type(self):
        raise NotImplementedError("To be implemented")

    def compare_to(self, other):
        raise NotImplementedError("To be implemented")


class EmpiricalRewardFunction(RewardFunction):
    """
    Defines a reward that is empirically estimated and it also depends on the observations of the current state
    """
    def __init__(self):
        super().__init__()
        self.reward_sum   = {}
        self.reward_count = {}

    def get_type(self):
        return "empirical"

    def compare_to(self, other):
        return False

    def get_reward(self, o1, a, o2):
        if o2 in self.reward_sum:
            return float(self.reward_sum[o2])/float(self.reward_count[o2])
        return 0

    def add_observed_reward(self, o2, r):
        if o2 not in self.reward_sum:
            self.reward_sum[o2]   = 0
            self.reward_count[o2] = 0
        self.reward_sum[o2]   += r
        self.reward_count[o2] += 1

    def get_info(self):
        info = []
        for o2 in self.reward_sum:
            r = self.get_reward(None, None, o2)
            if r != 0:
                info.append("\t%s -> %f"%(o2, r))
        return info


    def show(self):
        for o2 in self.reward_sum:
            r = self.get_reward(None, None, o2)
            if r != 0:
                print("\t%s -> %f"%(o2, r))
