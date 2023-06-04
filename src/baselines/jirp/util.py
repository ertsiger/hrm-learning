import random
import itertools
import os, tempfile
import copy
from collections import defaultdict
from baselines.jirp.dnf_compile import compile_dnf, evaluate_dnf_compiled
from baselines.jirp.reward_machine import RewardMachine
from baselines.jirp.consts import *


def approximate_hyp(approximation_method, language, transitions, n_states):
    empty_transition = dnf_for_empty(language)
    if n_states >= 2:
        for i in range(2, n_states):
            minimized_rm = approximation_method(MINIMIZATION_EPSILON, language, i, n_states, transitions, empty_transition, report=True)
            if minimized_rm:
                print(f"FOUND MINIMIZED RM {i} < {n_states}")
                print(transitions)
                print(minimized_rm)
                return minimized_rm, n_states
    print("couldn't find minimized RM, returning exact")

    # display = False
    # global last_displayed_states
    # if n_states > last_displayed_states or n_states <= 2:
    #     display = True
    #     last_displayed_states = n_states
    return approximation_method(EXACT_EPSILON, language, n_states, n_states, transitions, empty_transition, report=True, display=True), n_states

def equivalent_on_X(epsilon, eqv_check, H1, v1, H2, v2, X):
    """
    Checks if state v1 from RM H1 is equivalent to v2 from H2
    """
    H1_old = H1
    H2_old = H2
    H1 = H1.with_initial(v1)
    H2 = H2.with_initial(v2)
    total = len(X)
    eqv = 0
    for (labels, _rewards) in X:
        output1 = rm_run(labels, H1)
        output2 = rm_run(labels, H2)
        if eqv_check(epsilon, output1, output2):
            eqv += 1
    if float(eqv)/total >= EQV_THRESHOLD:
        # print(f"H_new/{v1} ~ H_old/{v2} (p ~= {float(eqv)/total})")
        return True
    return False

def eqv_score(epsilon, eqv_check, H1, v1, H2, v2, X):
    if len(X) == 0:
        return 0

    H1 = H1.with_initial(v1)
    H2 = H2.with_initial(v2)
    total = len(X)
    eqv = 0
    for (labels, _rewards) in X:
        output1 = rm_run(labels, H1)
        output2 = rm_run(labels, H2)
        if eqv_check(epsilon, output1, output2):
            eqv += 1
    return float(eqv)/total

def transfer_Q(epsilon, check_eqv, H_new, H_old, Q_old, X):
    """
    Returns new set of q-functions, indexed by states of H_new, where
    some of the q-functions may have been transfered from H_old if the
    respective states were determined to be equivalent
    
    Although the thm. requires the outputs be the same on _all_ label sequences,
    choosing probably equivalent states may be good enough.
    """
    Q = dict()
    Q[-1] = dict() # (-1 is the index of the terminal state) (TODO check if necessary)
    scores = dict()
    for v in H_new.get_states():
        for u in H_old.get_states():
            score = eqv_score(epsilon, check_eqv, H_new, v, H_old, u, X)
            if score >= EQV_THRESHOLD:
                if v not in scores:
                    scores[v] = (u, score)
                else:
                    previous_u, previous_score = scores[v]
                    if score > previous_score:
                        scores[v] = (u, score)

    for v in H_new.get_states():
        if v in scores:
            Q[v] = copy.deepcopy(Q_old[scores[v][0]])
            # print(f"H_new/{v} ~ H_old/{scores[v][0]} (p ~= {scores[v][1]})")
        else:
            Q[v] = dict()
            # print(f"H_new/{v} /~ any")
    return Q

def rm_run(labels, H):
    """
    Returns the output of H when labels is provided as input
    """
    current_state = H.reset()
    rewards = []
    for props in labels:
        current_state, reward, done = H.step(current_state, props, {"true_props": props})
        rewards.append(reward)
        if done:
            break
    return rewards

def run_eqv(epsilon, output1, output2):
    """
    Returns True if the accumulating error of the outputs is less than epsilon
    """
    if len(output1) != len(output2):
        return False
    sum1 = 0
    sum2 = 0
    for i in range(0, len(output1)):
        sum1 += output1[i]
        sum2 += output2[i]
        if abs(sum1 - sum2) > epsilon:
            return False
    return True

def sample_language(X):
    """
    Returns the set of all values for true_props strings for a given counter-example set X

    E.g. the sample language for {(("b", "ab"), (0.0, 1.0)), (("ab", "a", "f"), (0.0, 0.0, 1.0))}
    is {"", "f", "b", "a", "ab"}.
    """
    language = set()
    language.add("") # always contains empty string
    for (labels, _rewards) in X:
        language.update(labels)
    return language

def sample_reward_alphabet(X):
    """
    Returns the set of all reward values that appear in X
    """
    alphabet = set()
    alphabet.add(0.0) # always includes 0
    for (_labels, rewards) in X:
        alphabet.update(rewards)
    return alphabet

def dnf_for_empty(language):
    """
    Returns the "neutral" CNF for a given sample language corresponding
    to no events being true

    Convenience method. Works on the result of sample_language(X).
    Semantically equivalent to \\epsilon, but needed when forming DNFs
    """
    L = set()
    for labels in language:
        if labels == "":
            continue
        for label in labels:
            L.add("!" + str(label))
    return "&".join(L)

# TODO build trie
def prefixes(X, without_terminal=False):
    yield ((), ()) # (\epsilon, \epsilon) \in Pref(X)
    for (labels, rewards) in X:
        ending = 1 if not without_terminal else 0
        for i in range(1, len(labels) + ending):
            yield (labels[0:i], rewards[0:i])

def prefixes_trace(trace, without_terminal=False):
    labels = trace[0]
    rewards = trace[1]
    yield ((), ())
    ending = 1 if not without_terminal else 0
    for i in range(1, len(labels) + ending):
        yield (labels[0:i], rewards[0:i])

def all_pairs(xs):
    xs = list(xs)
    for i in range(0, len(xs)):
        for j in range(0, len(xs)):
            yield (xs[i], xs[j])

def different_pairs(xs):
    xs = list(xs)
    for i in range(0, len(xs)):
        for j in range(0, len(xs)):
            if i == j:
                continue
            yield (xs[i], xs[j])

def different_pairs_ordered(xs):
    xs = list(xs)
    for i in range(0, len(xs)):
        for j in range(i+1, len(xs)):
            yield (xs[i], xs[j])

def all_states(n_states):
    return range(INITIAL_STATE, n_states)

def all_states_terminal(n_states):
    return itertools.chain(all_states(n_states), [TERMINAL_STATE])

def add_pvar(storage, storage_rev, used_pvars, subscript):
    """
    Records a propositional variable indexed with the subscript by assigning it a unique
    index used by the solver. Returns this index

    If the variable indexed with that subscript was already recorded, no mutation is done,
    while the index is still returned.
    """
    key = subscript
    pvar = storage_rev.get(key)
    if pvar is not None:
        return pvar
    used_pvars[0] += 1
    storage[used_pvars[0]] = subscript
    storage_rev[key] = used_pvars[0]
    return used_pvars[0]

def rm_from_transitions(transitions, empty_transition):
    delta_u = defaultdict(dict)
    delta_r = defaultdict(dict)
    chars = set()
    for char in empty_transition:
        if char.isalnum():
            chars.add(char)

    for (p, l) in transitions:
        (q, r) = transitions[(p, l)]
        conj = []
        for char in chars:
            if char in l:
                conj.append(char)
            else:
                conj.append(f"!{char}")
        conj = "&".join(conj)
        if q not in delta_u[p]:
            delta_u[p][q] = conj
        else:
            delta_u[p][q] = delta_u[p][q] + "|" + conj
        if q not in delta_r[p]:
            delta_r[p][q] = [(conj, r)]
        else:
            delta_r[p][q].append((conj, r))
    
    rm_strings = [f"{INITIAL_STATE}", f"[{TERMINAL_STATE}]"]

    for p in delta_u:
        for q in delta_u[p]:
            rs = "{"
            for (label, reward) in delta_r[p][q]:
                rs += f"'{label}': {reward},"
            rs += "}"
            rs = f"LabelRewardFunction({rs})"
            s = "({},{},'{}',{})".format(p, q, delta_u[p][q], rs)
            rm_strings.append(s)

    rm_string = "\n".join(rm_strings)
    new_file, filename = tempfile.mkstemp()
    os.write(new_file, rm_string.encode())
    os.close(new_file)

    return RewardMachine(filename)

def rm_to_transitions(rm):
    transitions = dict()
    for (u1, trs) in rm.delta_u.items():
        for (u2, dnf) in trs.items():
            (true_for, chars) = compile_dnf(dnf)
            not_chars = tuple(map(lambda char: f"!{char}",chars))
            labels = [tuple(label) for label in true_for]
            for label in labels:
                _, r, _ = rm.step(u1, label, {})
                if label == ():
                    transitions[(u1, not_chars)] = [u2, r]
                else:
                    transitions[(u1, label)] = [u2, r]
    return transitions


def initial_Q(H):
    """
    Returns a set of uninitialized q-functions indexed by states of RM H
    """
    Q = dict()
    Q[-1] = dict()
    for v in H.get_states():
        Q[v] = dict()
    return Q

def get_qmax(Q, s, actions, q_init):
    if s not in Q:
        Q[s] = dict([(a,q_init) for a in actions])
    return max(Q[s].values())

def get_best_action(Q, s, actions, q_init):
    qmax = get_qmax(Q,s,actions,q_init)
    best = [a for a in actions if Q[s][a] == qmax]
    return random.choice(best)

def display_transitions(transitions, name):
    from graphviz import Digraph
    dot = Digraph(format='png',comment=name, graph_attr={"fontsize":"6.0"}, edge_attr={"color": "#000000aa"})

    nodes = set()

    for (p, a) in transitions:
        if p not in nodes:
            nodes.add(p)
            dot.node(str(p))

    for (p, a) in transitions:
        [q, r] = transitions[(p, a)]
        r = 0.0 if r == 0.0 else r
        dot.edge(str(p), str(q), label=f"({''.join(a)}, {('%f' % r).rstrip('0').rstrip('.')})")
        
    dot = dot.unflatten()
    dot.render(f"graphviz/{name}.gv", view=True)

def display_rm(rm, name, view=True):
    from graphviz import Digraph
    dot = Digraph(format='pdf',comment=name, graph_attr={"fontsize":"6.0"}, edge_attr={"color": "#000000aa"})
    nodes = set()
    dot.node("-1")
    for p in rm.U:
        dot.node(str(p))
    for p in rm.U:
        for q in rm.delta_u[p]:
            dot.edge(str(p),str(q),label=f"({rm.delta_r[p][q]})")        
            # dot.edge(str(p),str(q),label=f"({rm.delta_u[p][q]}, {rm.delta_r[p][q]})")        
    dot = dot.unflatten()
    dot.render(f"graphviz/{name}.gv", view=view)

def serializeable_rm(rm):
    transitions = dict()
    for p in rm.U:
        for q in rm.delta_u[p]:
            transitions[f"{p}-{q}"] = f"({rm.delta_u[p][q]}, {rm.delta_r[p][q]})"
    return transitions

def isomorphic(t1, t2, n_states):
    for bij in itertools.permutations(range(1, n_states+1)):
        bij = zip(bij, range(1, n_states+1))
        bij = {i[0]: i[1] for i in bij}
        t2_bij = {(bij[key[0]], key[1]): [bij[value[0]], value[1]] for key, value in t2.items()}
        if t2_bij == t1:
            return True
    return False

def all_states_here(asdf, infer_termination):
    if infer_termination:
        return all_states_terminal(asdf)
    else:
        return all_states(asdf)

def write_to_asdf(s):
    with open("asdf.txt", "w") as asdf:
        asdf.write(s)

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

def new_artificial_cx(H, env_rm, X):
    language = sample_language(X)
    X_new = set()
    labels = []
    rewards = []
    while True:
        l = random.choice(list(language))
        labels.append(l)
        rewards = rm_run(labels, env_rm)
        if len(labels) > len(rewards):
            break
    labels.pop()
    assert len(labels) == len(rewards)
    assert rewards == rm_run(labels, env_rm)

    if not run_approx_eqv(rm_run(labels, H), rewards):
        X_new.add((tuple(labels), tuple(rewards)))

    return X_new

# TODO use in approx etc.
def n_states_transitions(transitions):
    keys = transitions.keys()
    states = set(map(lambda x: x[0], keys))
    return len(states) - 1 # discounting terminal state

def transitions_from(p, transitions):
    result = dict()
    for (p2, a) in transitions:
        if p != p2:
            continue
        result[a] = transitions[(p, a)]
    return result
    
def reachable(p, transitions):
    result = set()
    for (p2, a) in transitions:
        if p != p2:
            continue
        result.add(transitions[(p2,a)][0])
    return result

def load_t(n):
    return RewardMachine(f"./envs/grids/reward_machines/office/t{n}.txt")

def load_m(n):
    return RewardMachine(f"./envs/grids/reward_machines/office/min/t{n}.txt")

def load_c(n):
    return RewardMachine(f"./envs/grids/reward_machines/craft/t{n}.txt")

def load(path):
    return RewardMachine(path)

def sum_rm(language,rm1,rm2):
    n1 = len(rm1.U)
    n2 = len(rm2.U)
    def sum_state(p, first):
        nonlocal n1
        if first:
            return p
        if p == 0 or p == -1:
            return p
        return n1 - 1 + p
    transitions = dict()
    for i1 in range(0, n1):
        i = sum_state(i1,True)
        for labels in language:
            j1, r1, _ = rm1.step(i1, labels, {})
            j = sum_state(j1,True)
            if (i, labels) not in transitions or transitions[(i,labels)]==[i,0.0]:
                transitions[(i,labels)]=[j,r1]
    for i2 in range(0, n2):
        i = sum_state(i2,False)
        for labels in language:
            j2, r2, _ = rm2.step(i2, labels, {})
            j = sum_state(j2,False)
            if (i, labels) not in transitions or transitions[(i,labels)]==[i,0.0]:
                transitions[(i,labels)]=[j,r2]
    return transitions

def product_rm(language, rm1, rm2):
    state_dict = dict()
    state_dict[(0, 0)] = 0
    state_dict[(-1,-1)] = -1

    def product_state(p1, p2):
        nonlocal rm1
        nonlocal rm2
        if (p1,p2) in state_dict:
            return state_dict[(p1,p2)]
        else:
            state_dict[(p1,p2)] = len(state_dict) - 1
            return state_dict[(p1,p2)]

    transitions = dict()
    for i1 in itertools.chain(rm1.U, [TERMINAL_STATE]):
        for i2 in itertools.chain(rm2.U,[TERMINAL_STATE]):
            i = product_state(i1,i2)
            for labels in language: # TODO doesn't work for labels with multiple events
                j1, r1, _ = rm1.step(i1, labels, {}) if i1 != TERMINAL_STATE else (TERMINAL_STATE,0,None)
                j2, r2, _ = rm2.step(i2, labels, {}) if i2 != TERMINAL_STATE else (TERMINAL_STATE,0,None)
                j = product_state(j1,j2)
                r = max(r1,r2) if j == TERMINAL_STATE else 0
                transitions[(i,labels)] = [j,r]

    # return transitions

    visited = set()
    to_visit= [-1,0]

    while to_visit:
        c = to_visit.pop()
        visited.add(c)
        for n in reachable(c, transitions):
            if n not in visited:
                to_visit.append(n)

    result = dict()

    state_dict = dict()
    state_dict[0]=0
    state_dict[-1]=-1
    def add_state(p):
        nonlocal state_dict
        if p not in state_dict:
            state_dict[p] = len(state_dict) - 1
        return state_dict[p]

    for (p, a) in transitions:
        if p in visited:
            [q, r] = transitions[(p, a)]
            p2 = add_state(p)
            q2 = add_state(q)
            result[(p2, a)] = [q2,r]
    
    return result
