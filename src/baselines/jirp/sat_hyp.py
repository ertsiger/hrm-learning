import time

from baselines.jirp.util import *
from baselines.jirp.consts import *
from pysat.solvers import Glucose4
from threading import Timer


def sat_hyp(epsilon, X, X_tl, n_states, infer_termination, report=True, inspect=False, display=False, timeout=None):
    language = sample_language(X)
    empty_transition = dnf_for_empty(language)
    reward_alphabet = sample_reward_alphabet(X)

    prop_d = dict() # maps SAT's propvar (int) to (p: state, l: labels, q: state)
    prop_d_rev = dict()
    prop_o = dict() # maps SAT's propvar (int) to (p: state, l: labels, r: reward)
    prop_o_rev = dict()
    prop_x = dict() # maps SAT's propvar (int) to (l: labels, q: state)
    prop_x_rev = dict()
    used_pvars = [0] # p. var. counter
    g = Glucose4() # solver

    # convenience methods
    def add_pvar_d(d):
        nonlocal prop_d
        nonlocal prop_d_rev
        return add_pvar(prop_d, prop_d_rev, used_pvars, d)

    def add_pvar_o(o):
        nonlocal prop_o
        nonlocal prop_o_rev
        return add_pvar(prop_o, prop_o_rev, used_pvars, o)

    def add_pvar_x(x):
        nonlocal prop_x
        nonlocal prop_x_rev
        return add_pvar(prop_x, prop_x_rev, used_pvars, x)

    # Encoding reward machines
    # (1)
    for p in all_states_here(n_states, infer_termination):
        for l in language:
            g.add_clause([add_pvar_d((p, l, q)) for q in all_states_here(n_states, infer_termination)])
            for q1 in all_states_here(n_states, infer_termination):
                for q2 in all_states_here(n_states, infer_termination):
                    if q1==q2:
                        continue
                    p_l_q1 = add_pvar_d((p, l, q1))
                    p_l_q2 = add_pvar_d((p, l, q2))
                    g.add_clause([-p_l_q1, -p_l_q2])

    # (2)
    for p in all_states_here(n_states, infer_termination):
        for l in language:
            g.add_clause([add_pvar_o((p, l, r)) for r in reward_alphabet])
            for r1 in reward_alphabet:
                for r2 in reward_alphabet:
                    if r1 == r2:
                        continue
                    p_l_r1 = add_pvar_o((p, l, r1))
                    p_l_r2 = add_pvar_o((p, l, r2))
                    g.add_clause([-p_l_r1, -p_l_r2])

    # Consistency with sample
    # (3)
    g.add_clause([add_pvar_x((tuple(), INITIAL_STATE))]) # starts in the initial state
    for p in all_states_here(n_states, infer_termination):
        if p == INITIAL_STATE:
            continue
        g.add_clause([-add_pvar_x((tuple(), p))])

    # (4)
    for (labels, _rewards) in prefixes(X, without_terminal=False):
        if labels == ():
            continue
        lm = labels[0:-1]
        l = labels[-1]
        for p in all_states_here(n_states, infer_termination):
            for q in all_states_here(n_states, infer_termination):
                x_1 = add_pvar_x((lm, p))
                d = add_pvar_d((p, l, q))
                x_2 = add_pvar_x((labels, q))
                g.add_clause([-x_1, -d, x_2])

    # (5)
    for (labels, rewards) in prefixes(X, without_terminal=False):
        if labels == ():
            continue
        lm = labels[0:-1]
        l = labels[-1]
        r = rewards[-1]
        for p in all_states_here(n_states, infer_termination):
            x = add_pvar_x((lm, p))
            o = add_pvar_o((p, l, r))
            g.add_clause([-x, o])
    
    # (Termination)
    if infer_termination:
        for (labels, _rewards) in prefixes(X, without_terminal=True):
            if labels == ():
                continue
            lm = labels[0:-1]
            l = labels[-1]
            x_2 = add_pvar_x((labels, TERMINAL_STATE)) # TODO REMOVE unneeded
            for p in all_states_here(n_states, infer_termination):
                if p == TERMINAL_STATE:
                    continue
                x_1 = add_pvar_x((lm, p))
                d = add_pvar_d((p, l, TERMINAL_STATE))
                g.add_clause([-x_1, -d])

        for (labels, rewards) in X:
            if labels == ():
                continue
            lm = labels[0:-1]
            l = labels[-1]
            x_2 = add_pvar_x((labels, TERMINAL_STATE)) # TODO REMOVE unneeded
            for p in all_states_here(n_states, infer_termination):
                if p == TERMINAL_STATE:
                    continue
                x_1 = add_pvar_x((lm, p))
                d = add_pvar_d((p, l, TERMINAL_STATE))
                d_t = -d if (labels, rewards) in X_tl else d
                g.add_clause([-x_1, d_t])

        for p in all_states_here(n_states, infer_termination):
            if p == TERMINAL_STATE:
                continue
            for l in language:
                d = add_pvar_d((TERMINAL_STATE, l, p))
                g.add_clause([-d])

        for p in all_states_here(n_states, infer_termination):
            for l in language:
                o = add_pvar_o((TERMINAL_STATE, l, 0.0))
                g.add_clause([o])

    # found = False
    # # (Relevant events)
    # for relevant in powerset(language):
    #     assumptions = []
    #     for p in all_states_here(n_states):
    #         if p == TERMINAL_STATE:
    #             continue
    #         for l in language:
    #             if l in relevant:
    #                 continue
    #             d = add_pvar_d((p, l, p))
    #             o = add_pvar_o((p, l, 0.0))
    #             assumptions.extend([d, o])
    #     g.solve(assumptions=assumptions)
    #     # if len(relevant) == len(language):
    #     #     IPython.embed()
    #     if g.get_model() is None:
    #         continue
    #     else:
    #         found = True
    #         if report:
    #             print(f"found with assumptions {relevant}")
    #         break

    # if not found:
    #     return None

    def interrupt(s):
        s.interrupt()

    if timeout is not None:
        timer = Timer(timeout, interrupt, [g])
        timer.start()

    start = time.time()
    g.solve_limited(expect_interrupt=True)

    if timeout is not None:
        total_time = time.time() - start
        if total_time > timeout:
            raise TimeoutError

    if g.get_model() is None:
        return None

    if report:
        print("found")

    transitions = dict() #defaultdict(lambda: [None, None]) # maps (state, true_props) to (state, reward)

    for pvar in g.get_model():
        if abs(pvar) in prop_d:
            if pvar > 0:
                (p, l, q) = prop_d[abs(pvar)]
                # assert transitions[(p, tuple(l))][0] is None
                if (p, tuple(l)) not in transitions:
                    transitions[(p, tuple(l))] = [None, None]
                transitions[(p, tuple(l))][0] = q
                # assert q is not None
        elif abs(pvar) in prop_o:
            if pvar > 0:
                (p, l, r) = prop_o[abs(pvar)]
                if (p, tuple(l)) not in transitions:
                    transitions[(p, tuple(l))] = [None, None]
                # assert transitions[(p, tuple(l))][1] is None
                transitions[(p, tuple(l))][1] = r
        elif abs(pvar) in prop_x:
            pass
        else:
            raise ValueError("Uknown p-var dict")
    
    g.delete()
    return transitions
