from baselines.common.baseline import Baseline
from baselines.jirp.jirp_env_wrapper import JIRPEnvWrapper
from baselines.jirp.sat_hyp import sat_hyp
from baselines.jirp.util import *
from gym_hierarchical_subgoal_automata.automata.hierarchical_automaton import HierarchicalAutomaton
from gym_hierarchical_subgoal_automata.automata.logic import DNFFormula
import numpy as np
import time


class JIRP(Baseline):
    REPORT = False
    NOISE_UPDATE_X_EVERY_N = 20

    def __init__(self, args):
        super(JIRP, self).__init__(args)

        self.env = JIRPEnvWrapper(args)

        self.max_num_states = args.jirp_max_states

        self.infer_termination = True
        self.checking_epsilon = EXACT_EPSILON
        self.inference_epsilon = 0.0

        self.All = set()
        self.X = set()
        self.X_new = set()
        self.X_tl = set()

        # Initialization
        self.transitions, self.n_states_last = self.consistent_hyp(
            set(), set(), self.infer_termination, self.max_num_states, report=JIRP.REPORT
        )
        self.language = sample_language(self.X)
        self.empty_transition = dnf_for_empty(self.language)
        self.H = rm_from_transitions(self.transitions, self.empty_transition)

    def run_algorithm(self):
        timeout = self.time_left < 0
        while self.episode < self.num_episodes and not timeout:
            self.env.reset()
            labels = []
            rewards = []

            learned_machine = False

            step = 0
            while step < self.max_episode_length and not timeout:
                a = np.random.choice(self.env.get_actions())
                _, r, done, info = self.env.step(a)

                true_props = self.env.get_events()
                if len(labels) == 0 or labels[-1] != true_props:  # added compression of traces (needed for waterworld, and used in all other approaches)
                    labels.append(true_props)  # L(s, a, s')
                    rewards.append(r)
                step += 1

                if done:
                    self.All.add((tuple(labels), tuple(rewards)))
                    if not self.run_eqv_noise(self.checking_epsilon, rm_run(labels, self.H), rewards):
                        self.X_new.add((tuple(labels), tuple(rewards)))
                        if "TimeLimit.truncated" in info:  # could also see if RM is in a terminating state
                            tl = info["TimeLimit.truncated"]
                            if tl:
                                self.X_tl.add((tuple(labels), tuple(rewards)))

                    if self.X_new and self.episode % JIRP.NOISE_UPDATE_X_EVERY_N == 0:
                        if JIRP.REPORT:
                            print(f"len(X)={len(self.X)}")
                            print(f"len(X_new)={len(self.X_new)}")
                        self.X.update(self.X_new)
                        self.X_new = set()
                        self.language = sample_language(self.X)
                        self.empty_transition = dnf_for_empty(self.language)

                        result = self.consistent_hyp(
                            self.X, self.X_tl, self.infer_termination, self.max_num_states,
                            n_states_start=self.n_states_last, report=JIRP.REPORT
                        )

                        if self.time_left < 0:
                            timeout = True
                            break

                        if result is not None:
                            transitions_new, self.n_states_last = result
                        else:
                            raise ValueError(f"Couldn't find machine with at most {MAX_RM_STATES_N} states")
                        H_new = rm_from_transitions(transitions_new, self.empty_transition)
                        if not self.consistent_on_all(self.checking_epsilon, self.X, H_new):
                            print("NOT CONSISTENT IMMEDIATELY")
                        self.H = H_new
                        self.transitions = transitions_new
                        learned_machine = True

                        # num calls is not sequential (1..2...) since the code iterates through different number of states
                        # if the tasks are UNSAT
                        self.get_hierarchical_automaton().plot(self.output_dir, f"automaton_{self.num_calls}.png")
                    break

            self.episode += 1

            if self.episode % 100 == 0 or learned_machine or timeout:
                self.make_checkpoint(self.episode)

    def consistent_hyp(self, X, X_tl, infer_termination, max_num_states, n_states_start=1, report=True):
        """
        Finds a reward machine consistent with counterexample set X. Returns the RM
        and its number of states

        n_states_start makes the search start from machines with that number of states.
        Used to optimize succeeding search calls.
        """
        if len(X) == 0:
            transitions = dict()
            transitions[(0, tuple())] = [0, 0.0]
            return transitions, 1
        # TODO intercept empty X here
        for n_states in range(n_states_start, max_num_states + 1):
            if report:
                print(f"finding model with {n_states} states")

            start_time = time.time()
            try:
                new_transitions = sat_hyp(0.0, X, X_tl, n_states, infer_termination, report=report,
                                          timeout=self.time_left)  # epsilon does nothing in sat_hyp
            except TimeoutError:
                pass
            self.time_left -= (time.time() - start_time)
            self.num_calls += 1

            if self.time_left < 0:
                break

            if new_transitions is not None:
                return new_transitions, n_states
            continue

        return None

    def run_eqv_noise(self, epsilon, output1, output2):
        """
        Returns True if all outputs are within epsilon of each other (output1 is a noise-distorted output2, eg.)
        """
        if len(output1) != len(output2):
            return False
        for i in range(0, len(output1)):
            if abs(output1[i] - output2[i]) > epsilon:
                return False
        return True

    def consistent_on_all(self, epsilon, X, H):
        for (labelsx, rewardsx) in X:
            if not self.run_eqv_noise(epsilon, rm_run(labelsx, H), rewardsx):
                return False
        return True

    def get_num_states(self):
        return self.get_hierarchical_automaton().get_num_states()

    def get_num_edges(self):
        return self.get_hierarchical_automaton().get_num_edges()

    def get_num_traces(self):
        return len(self.X) + len(self.X_tl)

    def get_hierarchical_automaton(self):
        automaton = HierarchicalAutomaton("m0")
        for tx in self.transitions:
            from_state, label = tx
            to_state, rw = self.transitions[tx][0], self.transitions[tx][1]

            if from_state < 0:
                from_state = "t"
            if to_state < 0:
                to_state = "t"

            automaton.add_state(f"u{from_state}")
            automaton.add_state(f"u{to_state}")
            if from_state != to_state:
                labelp = "".join(sorted(label))
                automaton.add_formula_edge(f"u{from_state}", f"u{to_state}", DNFFormula([[labelp]]))
        return automaton
