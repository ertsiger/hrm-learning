from baselines.jirp.reward_functions import RewardFunction
from baselines.jirp.dnf_compile import compile_dnf, evaluate_dnf_compiled
import random

def for_display(dnf):
    result = ""
    skip_next = False
    for i in range(0, len(dnf)):
        if dnf[i] == '!':
            skip_next = True
        elif dnf[i].isalnum() and not skip_next:
            result += dnf[i]
        else:
            skip_next = False
    return result

class LabelRewardFunction(RewardFunction):
    """
    Defines a reward function that depends on true_props/events/label
    """
    def __init__(self, label_rewards):
        super().__init__()
        self.label_rewards = label_rewards
        self.compiled_dnfs = dict([(dnf, compile_dnf(dnf)) for dnf in label_rewards])

    def get_type(self):
        return "label"

    def get_means(self):
        result = set()
        for dnf in self.label_rewards:
            result.add(self.label_rewards[dnf])
        return result

    def get_reward(self, s_info):
        if "true_props" not in s_info:
            return 0.0
        true_props = s_info["true_props"]
        for dnf in self.label_rewards:
            if evaluate_dnf_compiled(self.compiled_dnfs[dnf], true_props):
                return self.label_rewards[dnf]
        return 0.0

    def change_for(self, true_props, to):
        for dnf in self.label_rewards:
            if evaluate_dnf_compiled(self.compiled_dnfs[dnf], true_props):
                self.label_rewards[dnf] = to
                return

    def __str__(self):
        return str(dict(map(lambda x: (for_display(x[0]), x[1]), self.label_rewards.items())))

    def __repr__(self):
        return str(dict(map(lambda x: (for_display(x[0]), x[1]), self.label_rewards.items())))

# (3,5,'f|n',LabelRewardFunction({"f": 1.05, "n": 0.0}))

class NoisyContLabelRewardFunction(RewardFunction):
    """
    Defines a constant reward for a 'simple reward machine'
    """
    def __init__(self, label_rewards):
        super().__init__()
        self.label_rewards = label_rewards
        self.compiled_dnfs = dict([(dnf, compile_dnf(dnf)) for dnf in label_rewards])
        self.eps = 0
        for dnf in label_rewards:
            if self.eps < label_rewards[dnf][1]:
                self.eps = label_rewards[dnf][1]

    def get_type(self):
        return "noisy_cont_label"

    def get_means(self):
        result = set()
        for dnf in self.label_rewards:
            result.add(self.label_rewards[dnf][0]) # mean, eps
        return result

    def get_reward(self, s_info):
        if "true_props" not in s_info:
            return 0.0
        true_props = s_info["true_props"]
        for dnf in self.label_rewards:
            if evaluate_dnf_compiled(self.compiled_dnfs[dnf], true_props):
                return self.label_rewards[dnf][0] + random.uniform(-self.label_rewards[dnf][1], self.label_rewards[dnf][1])
        return 0.0

    def __str__(self):
        return f"{self.label_rewards} (eps={self.eps})"

    def __repr__(self):
        return f"{self.label_rewards} (eps={self.eps})"


class NoisyContRewardFunction(RewardFunction):
    """
    Defines a constant reward for a 'simple reward machine'
    """
    def __init__(self, c, eps):
        super().__init__()
        self.c = c
        self.eps = eps

    def get_type(self):
        return "noisy_cont"

    def get_mean(self):
        return self.c

    def get_means(self):
        return set([self.c])

    def get_reward(self, s_info):
        return self.c + random.uniform(-self.eps, self.eps)

    def __str__(self):
        return f"{self.c} (eps={self.eps})"

    def __repr__(self):
        return f"{self.c} (eps={self.eps})"
