from baselines.common.baseline import Baseline
from baselines.deepsynth.deepsynth_env_wrapper import DeepSynthEnvWrapper
from baselines.deepsynth.synth_wrapper import dfa_init, dfa_update, get_next_state
from gym_hierarchical_subgoal_automata.automata.hierarchical_automaton import HierarchicalAutomaton
from gym_hierarchical_subgoal_automata.automata.logic import DNFFormula
import numpy as np
import os
import stopit
import time


MIN_DFA_FRAMES = 50000            # The minimum number of exploration frames to generate an intial DFA
DFA_UPDATE_FREQ = 1000


class DeepSynth(Baseline):
    def __init__(self, args):
        super(DeepSynth, self).__init__(args)
        self.env = DeepSynthEnvWrapper(args)
        self.dfa_states = [0, 1]
        self.rewards = []
        self.set_of_episode_traces = []
        self.model_gen = []
        self.nfa_model = []
        self.dfa_model = []
        self.num_states, self.var, self.input_dict, self.hyperparams = dfa_init()
        self.total_num_steps = 0
        self.processed_dfa = None
        self.num_used_traces = 0  # not necessarily unique, just used across the learning for each call to the learner
        self.cbmc_path = args.deepsynth_cbmc_path

    def run(self, loaded_checkpoint):
        os.environ.putenv("PATH", self.cbmc_path + os.pathsep + os.environ.get("PATH", ""))
        super(DeepSynth, self).run(loaded_checkpoint)

    def run_algorithm(self):
        timed_out = self.time_left < 0
        while self.episode < self.num_episodes and not timed_out:
            start_time = time.time()
            self.env.reset()
            terminal = False
            episode_trace = ['start']
            episode_detected_objects = []
            episode_reward_sum = 0
            num_steps = 0

            while not terminal and num_steps < self.max_episode_length and not timed_out:
                # Get action
                action = np.random.choice(self.env.get_actions())

                # Take step
                obs, reward, terminal, _ = self.env.step(action)
                self.total_num_steps += 1
                num_steps += 1

                event = self.env.get_events()
                episode_detected_objects.append(event)
                if event != episode_trace[-1]:  # compression
                    episode_trace.append(event)

                new_obj_set = np.unique(episode_detected_objects).tolist()

                # ### SYNTH ### #
                old_dfa_states = self.dfa_states.copy()
                if self.total_num_steps >= MIN_DFA_FRAMES:
                    # # SYNTH updates the automaton here:
                    # Removed condition (self.total_num_steps % DFA_UPDATE_FREQ == 0) or  since it cause the same automaton
                    # to be learned over and over, causing very early timeouts. Thus, an automaton is only learned now when
                    # a counterexample is found.
                    if (get_next_state(episode_trace, self.input_dict['event_uniq'], self.processed_dfa) == -1) or \
                        (get_next_state(episode_trace, self.input_dict['event_uniq'], self.processed_dfa) == []):
                        trace = []
                        self.set_of_episode_traces.append(episode_trace)
                        for x in self.set_of_episode_traces:
                            trace = trace + x
                        trace = trace + ['start']

                        self.num_used_traces += len(self.set_of_episode_traces)  # note the set is restarted afterwards..

                        start_learning = time.time()

                        with stopit.ThreadingTimeout(self.time_left) as to_ctx_mgr:
                            self.num_states, self.processed_dfa, self.dfa_model, self.nfa_model, self.model_gen, self.var, self.input_dict = dfa_update(
                                trace, self.num_states,
                                self.dfa_model,
                                self.nfa_model,
                                self.model_gen, self.var,
                                self.input_dict,
                                self.hyperparams,
                                start_time,
                                self.num_calls)

                        self.time_left -= (time.time() - start_learning)
                        self.num_calls = self.num_calls + 1

                        if to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
                            timed_out = True
                            break

                        self.dfa_states = list(set([dfa_transitions[0] for dfa_transitions in self.processed_dfa] +
                                                   [dfa_transitions[2] for dfa_transitions in self.processed_dfa]))
                        self.set_of_episode_traces = [episode_trace]
                        self.get_hierarchical_automaton().plot(self.output_dir, f"automaton_{self.num_calls}.png")

                    # determine next dfa state
                    next_dfa_state = get_next_state(episode_trace, self.input_dict['event_uniq'], self.processed_dfa)
                    episode_detected_objects = new_obj_set

                    episode_reward_sum += reward

                if terminal:
                    episode_detected_objects = []
                    self.set_of_episode_traces.append(episode_trace)
                    episode_trace = ['start']

            self.rewards.append(episode_reward_sum)

            self.episode += 1

            if self.episode % 100 == 0 or timed_out:
                self.make_checkpoint(self.episode)

    def get_num_states(self):
        return self.num_states

    def get_num_edges(self):
        return len(self.processed_dfa) if self.processed_dfa is not None else 0

    def get_num_traces(self):
        return self.num_used_traces

    def get_hierarchical_automaton(self):
        automaton = HierarchicalAutomaton("m0")
        for state in self.dfa_states:
            automaton.add_state(f"u{state}")
        for from_state, obs_idx, to_state in self.processed_dfa:
            obs = self.var["o_event_uniq"][obs_idx - 1]
            automaton.add_formula_edge(f"u{from_state}", f"u{to_state}", DNFFormula([[obs]]))
        return automaton
