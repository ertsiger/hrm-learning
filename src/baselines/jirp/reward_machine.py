from baselines.jirp.reward_functions import *
from baselines.jirp.reward_machine_utils import value_iteration
from baselines.jirp.label_reward_function import LabelRewardFunction, NoisyContRewardFunction, NoisyContLabelRewardFunction
from baselines.jirp.dnf_compile import compile_dnf, evaluate_dnf_compiled
import copy


class RewardMachine:
    def __init__(self, file):
        # <U,u0,delta_u,delta_r>
        self.U  = []         # list of non-terminal RM states
        self.u0 = None       # initial state
        self.delta_u    = {} # state-transition function
        self.delta_u_compiled = {} # compiled dnfs
        self.delta_r    = {} # reward-transition function
        self.terminal_u = -1  # All terminal states are sent to the same terminal state with id *-1*
        self.epsilon_cont = None # non-zero for continuous noise
        self.noise_delta = None # smallest distance between two means
        self._load_reward_machine(file)
        self.known_transitions = {} # Auxiliary variable to speed up computation of the next RM state

    # Public methods -----------------------------------

    def add_reward_shaping(self, gamma, rs_gamma):
        """
        It computes the potential values for shaping the reward function:
            - gamma(float):    this is the gamma from the environment
            - rs_gamma(float): this gamma that is used in the value iteration that compute the shaping potentials
        """
        self.gamma = gamma
        self.potentials = value_iteration(self.U, self.delta_u, self.delta_r, self.terminal_u, rs_gamma)
        for u in self.potentials:
            self.potentials[u] = -self.potentials[u]


    def reset(self):
        # Returns the initial state
        return self.u0

    def _compute_next_state(self, u1, true_props):
        for u2 in self.delta_u[u1]:
            if evaluate_dnf_compiled(self.delta_u_compiled[u1][u2], true_props):
                return u2
        return self.terminal_u # no transition is defined for true_props

    def get_next_state(self, u1, true_props):
        if (u1,true_props) not in self.known_transitions:
            u2 = self._compute_next_state(u1, true_props)
            self.known_transitions[(u1,true_props)] = u2
        return self.known_transitions[(u1,true_props)]

    def step(self, u1, true_props, s_info, add_rs=False, env_done=False):
        """
        Emulates an step on the reward machine from state *u1* when observing *true_props*.
        The rest of the parameters are for computing the reward when working with non-simple RMs: s_info (extra state information to compute the reward).
        """

        # Computing the next state in the RM and checking if the episode is done
        assert u1 != self.terminal_u, "the RM was set to a terminal state!"
        u2 = self.get_next_state(u1, true_props)
        done = (u2 == self.terminal_u)

        ###
        s_info["true_props"] = true_props
        ###

        # Getting the reward
        rew = self._get_reward(u1,u2,s_info,add_rs, env_done)
        
        return u2, rew, done


    def get_states(self):
        return self.U

    def get_useful_transitions(self, u1):
        # This is an auxiliary method used by the HRL baseline to prune "useless" options
        return [self.delta_u[u1][u2].split("&") for u2 in self.delta_u[u1] if u1 != u2]

    def with_initial(self, v):
        """
        For reward machine A and state v, return A[v] (RM A with v as its initial state)
        """
        new_rm = copy.deepcopy(self)
        new_rm.u0 = v
        new_rm.known_transitions = {}
        return new_rm

    def move_output(self, u1, true_props, to):
        assert u1 != self.terminal_u, "the RM was set to a terminal state!"
        u2 = self.get_next_state(u1, true_props)
        done = (u2 == self.terminal_u)
        self._update_mean(u1, u2, to, true_props)
        
    # Private methods -----------------------------------
    def _update_mean(self, u1, u2, to, true_props):
        if u1 in self.delta_r and u2 in self.delta_r[u1]:
            reward_function = self.delta_r[u1][u2]
            if isinstance(reward_function, LabelRewardFunction):
                reward_function.change_for(true_props, to)
            reward_function.c = to

    def _get_mean(self,u1,u2,s_info,add_rs,env_done):
        # Getting reward from the RM
        reward = 0 # NOTE: if the agent falls from the reward machine it receives reward of zero
        if u1 in self.delta_r and u2 in self.delta_r[u1]:
            reward_function = self.delta_r[u1][u2]
            if not isinstance(reward_function, NoisyContRewardFunction): # or NoisyDisc...
                reward += reward_function.get_reward(s_info)
            else:
                reward += reward_function.get_mean(s_info)
        return reward

    def _get_reward(self,u1,u2,s_info,add_rs,env_done):
        """
        Returns the reward associated to this transition.
        """
        # Getting reward from the RM
        reward = 0 # NOTE: if the agent falls from the reward machine it receives reward of zero
        if u1 in self.delta_r and u2 in self.delta_r[u1]:
            reward += self.delta_r[u1][u2].get_reward(s_info)
        # Adding the reward shaping (if needed)
        rs = 0.0
        if add_rs:
            un = self.terminal_u if env_done else u2 # If the env reached a terminal state, we have to use the potential from the terminal RM state to keep RS optimality guarantees
            rs = self.gamma * self.potentials[un] - self.potentials[u1]
        # Returning final reward
        return reward + rs

    def _load_reward_machine(self, file):
        """
        Example:
            0      # initial state
            [2]    # terminal state
            (0,0,'!e&!n',ConstantRewardFunction(0))
            (0,1,'e&!g&!n',ConstantRewardFunction(0))
            (0,2,'e&g&!n',ConstantRewardFunction(1))
            (1,1,'!g&!n',ConstantRewardFunction(0))
            (1,2,'g&!n',ConstantRewardFunction(1))
        """
        # Reading the file
        f = open(file)
        lines = (line.rstrip() for line in f)
        lines = list(line for line in lines if line)
        f.close()
        # setting the DFA
        self.u0 = eval(lines[0])
        terminal_states = eval(lines[1])
        # adding transitions
        means = set()
        for e in lines[2:]:
            # Reading the transition
            u1, u2, dnf_formula, reward_function = eval(e)
            if isinstance(reward_function, NoisyContRewardFunction) or isinstance(reward_function, NoisyContLabelRewardFunction):
                if self.epsilon_cont is None:
                    self.epsilon_cont = reward_function.eps
                elif self.epsilon_cont < reward_function.eps:
                    self.epsilon_cont = reward_function.eps
            
            if isinstance(reward_function, ConstantRewardFunction):
                means.add(reward_function.c)
            else: # assume only constant, noisy, label, and noisy label are used
                means.update(reward_function.get_means())


            # terminal states
            if u1 in terminal_states:
                continue
            if u2 in terminal_states:
                u2  = self.terminal_u
            # Adding machine state
            self._add_state([u1,u2])
            # Adding state-transition to delta_u
            if u1 not in self.delta_u:
                self.delta_u[u1] = {}
                self.delta_u_compiled[u1] = {}
            self.delta_u[u1][u2] = dnf_formula
            self.delta_u_compiled[u1][u2] = compile_dnf(dnf_formula)
            # Adding reward-transition to delta_r
            if u1 not in self.delta_r:
                self.delta_r[u1] = {}
            self.delta_r[u1][u2] = reward_function
        # Sorting self.U... just because... 
        self.U = sorted(self.U)

        means = sorted(list(means))
        self.noise_delta = None
        for i in range(0, len(means)-1):
            delta = means[i+1] - means[i]
            if self.noise_delta is None:
                self.noise_delta = delta
            else:
                if delta < self.noise_delta:
                    self.noise_delta = delta

    def _add_state(self, u_list):
        for u in u_list:
            if u not in self.U and u != self.terminal_u:
                self.U.append(u)
