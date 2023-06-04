from baselines.lrm.reward_functions import EmpiricalRewardFunction
from baselines.lrm.tabu_search import run_tabu_search


class RewardMachine:
    def __init__(self, U_max, preprocess_trace, tabu_list_max, n_workers, rm_lr_steps, perfect_rm, use_perfect_rm):
        # General learning parameters
        self.tabu_list_max = tabu_list_max # maximum size for the tabu list
        self.preprocess_trace = preprocess_trace # when True, the trace is preprocess to remove replicated transitions
        self.U_max     = U_max     # maximum number of RM states
        self.n_workers = n_workers # number of threads to use when learning the RM
        self.rm_lr_steps = rm_lr_steps  # Number of learning steps for Tabu search
        self.perfect_rm = perfect_rm # This is the perfect rm for this domain (we use it for debugging purposes only)
        self.use_perfect_rm = use_perfect_rm # Flag indicating to use the perfect rm (we use it for debugging purposes only)
        self.N = {}  # auxiliary variable that indicates the possible consequences of an observation given a 1-state RM

        # <U,u0,delta_u,state_r,state_t,state_p>
        self.u0 = 0          # initial state
        self.U = None        # list of machine states
        self.delta_u = None  # state-transition function
        self.state_p = {}    # set of predictions given the RM state and observation
        self.state_r = []    # estimated reward given the RM state and observation
        self.traces  = []    # list of traces that we expect this RM to fit
        self.n_examples_long = 0  # total length of the full traces
        self.n_examples_short = 0 # total length of the compressed traces
        self.terminal_obs = set()   # set of terminal observations

    # Public methods -----------------------------------

    def add_trace(self, trace):
        self.n_examples_long += len(trace) - 1
        # preprocessing the trace
        if self.preprocess_trace:
            trace_clean = [trace[0]]
            for e,r in trace:
                if trace_clean[-1][0] != e:
                    trace_clean.append((e,r))
            trace = trace_clean
        # adding the trace
        self.traces.append(trace)
        self.n_examples_short += len(trace) - 1
        # Updating the 'N' model
        for t in range(1,len(trace)):
            o1 = trace[t-1][0]
            o2 = trace[t][0]
            if o1 not in self.N:
                self.N[o1] = set()
            self.N[o1].add(o2)

    def learn_the_reward_machine(self):
        """
        It uses the given traces to estimate the RM that makes the problem as Markovian as possible
        """
        if self.use_perfect_rm:
            # HACK[Perfect RM]: This sets delta_u to a handcrafted perfect RM---which is used for debugging purposes only
            delta_u = self.perfect_rm if self.delta_u is None else None
            score, score_perfect = 0, 0
        else:
            # Learning the RM from the traces
            # Format: delta[(ui,o)] = uj iff observation 'o' moves the RM from 'ui' to 'uj'
            current_rm = self.delta_u
            delta_u, score, score_perfect = run_tabu_search(self.traces, self.U_max, self.tabu_list_max, self.n_workers, self.rm_lr_steps, current_rm, self.perfect_rm)

        same_rm = delta_u is None
        if same_rm:
            print("Adapting the old RM to fit the new observations!")
            self.state_p = {}

            # Adding trivial predictions
            for o1 in self.N:
                for ui in range(self.U_max):
                    self.state_p[(ui,o1)] = set([o1])
                    if len(self.N[o1]) == 1:
                        for o2 in self.N[o1]:
                            self.state_p[(ui,o1)].add(o2)

            # Simulating the traces to estimate the empirical rewards and predictions
            for trace in self.traces:
                ui = self.get_initial_state()
                for t in range(1, len(trace)):
                    o1,_ = trace[t-1]
                    o2,r = trace[t]

                    # adding the prediction
                    self.state_p[(ui,o1)].add(o2)

                    # progressing ui
                    ui = self.get_next_state(ui, o2)

        else:
            print("A new RM must be learned!")

            # relabelling delta_u to use all the states from 0...U
            U = set()
            for u1,o in delta_u:
                U.add(u1)
                U.add(delta_u[u1,o])
            U = sorted(list(U))
            new_u = dict([(U[i], i) for i in range(len(U))])
            self.delta_u = {}
            for u1,o in delta_u:
                self.delta_u[new_u[u1],o] = new_u[delta_u[u1,o]]
            self.U = list(range(len(U)))

            # Resetting the reward machine
            self.state_p = {}
            self.state_r = [EmpiricalRewardFunction() for _ in range(self.U_max)]

            # Adding trivial predictions
            for o1 in self.N:
                for ui in range(self.U_max):
                    self.state_p[(ui,o1)] = set([o1])
                    if len(self.N[o1]) == 1:
                        for o2 in self.N[o1]:
                            self.state_p[(ui,o1)].add(o2)

            # Simulating the traces to estimate the empirical rewards and predictions
            for trace in self.traces:
                ui = self.get_initial_state()
                for t in range(1, len(trace)):
                    o1,_ = trace[t-1]
                    o2,r = trace[t]

                    # adding the prediction
                    self.state_p[(ui,o1)].add(o2)

                    # adding the reward
                    self.update_rewards(ui, o2, r)

                    # progressing ui
                    ui = self.get_next_state(ui, o2)

        print("\nThis is the current RM")
        print("U:", self.U)
        for k in self.delta_u:
            print("delta_u[%s]=%d"%(str(k),self.delta_u[k]))
        for ui in range(self.U_max):
            print("state_r[%d]"%ui)
            self.state_r[ui].show()
        print("terminal_obs:", self.terminal_obs)
        print("n_examples_long:", self.n_examples_long)
        print("n_examples_short:", self.n_examples_short)
        print("rm_score: %0.2f"%score)
        print("rm_perfect_score: %0.2f"%score_perfect)

        # Returns True if the new RM is identical to the new RM (i.e. there is no need to restart the q-functions)
        # it also returns info about the rm learning proces
        info = (self.n_examples_long, self.n_examples_short, len(self.traces), score_perfect, score)
        return same_rm, info

    def get_info(self):
        info = []
        info.append("U = " + str(self.U))
        info.append("u0 = " + str(self.u0))
        info.append("terminal_obs: " + str(self.terminal_obs))
        info.append("----------------")
        for k in self.delta_u:
            info.append("delta_u[%s]=%d"%(str(k),self.delta_u[k]))
        info.append("----------------")
        for k in self.state_p:
            info.append("state_p[%s]=%s"%(str(k),str(self.state_p[k])))
        info.append("----------------")

        for ui in range(self.U_max):
            info.append("state_r[%d]"%ui)
            info.extend(self.state_r[ui].get_info())
        return info

    def show(self):
        print("\nThis is the current RM")
        print("U:", self.U)
        print("u0:", self.u0)

        for k in self.delta_u:
            print("delta_u[%s]=%d"%(str(k),self.delta_u[k]))

        for ui in range(self.U_max):
            print("state_r[%d]"%ui)
            self.state_r[ui].show()

        print("terminal_obs:", self.terminal_obs)

        for k in self.state_p:
            print("state_p[%s]=%s"%(str(k),str(self.state_p[k])))
        
    def add_terminal_observations(self, o2):
        self.terminal_obs.add(o2)

    def update_rewards(self, u1, o2, r):
        self.state_r[u1].add_observed_reward(o2, r)

    def get_initial_state(self):
        return self.u0

    def get_next_state(self, u1, o2):
        if (u1,o2) in self.delta_u:
            return self.delta_u[(u1,o2)]
        return u1 # if the transition isn't defined, then the RM doesn't move from the current state

    def get_reward(self,u1,o1,a,o2):
        # Getting reward from the RM (reward is zero by default)
        return self.state_r[u1].get_reward(o1,a,o2)

    def is_terminal_observation(self, o2):
        return o2 in self.terminal_obs

    def is_observation_impossible(self, u1, o1, o2):
        #return o2 not in self.state_p[(u1,o1)]
        return ((u1,o1) not in self.state_p) or (o2 not in self.state_p[(u1,o1)])

    def get_states(self):
        return self.U
