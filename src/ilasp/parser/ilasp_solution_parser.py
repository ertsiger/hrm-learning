from gym_hierarchical_subgoal_automata.automata.hierarchical_automaton import HierarchicalAutomaton
from gym_hierarchical_subgoal_automata.automata.logic import DNFFormula, TRUE
from ilasp.ilasp_common import CALL_STR, LEAF_AUTOMATON, N_TRANSITION_STR
from ilasp.parser.ilasp_parser_utils import parse_call_rule, parse_negative_transition_rule
from typing import List, Dict, Tuple


def parse_ilasp_solutions(automaton_name, ilasp_learnt_filename):
    """
    We assume that all states with outgoing edges are reachable!
    """
    # (from, to) -> [(machine) -> list(edges)]
    call_edges: Dict[Tuple[str, str], Dict[str, List[str]]] = {}

    # (from, to) -> [(edge) -> formula]
    formula_edges: Dict[Tuple[str, str], Dict[int, List[str]]] = {}

    _parse_calls_and_formulas(ilasp_learnt_filename, call_edges, formula_edges)
    return _build_automaton(automaton_name, call_edges, formula_edges)


def _parse_calls_and_formulas(ilasp_learnt_filename, call_edges: Dict[Tuple[str, str], Dict[str, List[str]]],
                              formula_edges: Dict[Tuple[str, str], Dict[int, List[str]]]):
    with open(ilasp_learnt_filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith(CALL_STR):
                parsed_call = parse_call_rule(line)
                edge_src_dst = (parsed_call.src, parsed_call.dst)
                if edge_src_dst in call_edges:
                    if parsed_call.edge in call_edges[edge_src_dst]:
                        raise RuntimeError(f"Error: A call is defined twice for the edge {parsed_call.edge} from "
                                           f"{parsed_call.src} to {parsed_call.dst}.")
                else:
                    call_edges[edge_src_dst] = {}
                if parsed_call.dst_automaton not in call_edges[edge_src_dst]:
                    call_edges[edge_src_dst][parsed_call.dst_automaton] = []
                call_edges[edge_src_dst][parsed_call.dst_automaton].append(parsed_call.edge)
            elif line.startswith(N_TRANSITION_STR):
                parsed_transition = parse_negative_transition_rule(line)
                edge_src_dst = (parsed_transition.src, parsed_transition.dst) # , parsed_transition.edge)
                if edge_src_dst not in formula_edges:
                    formula_edges[edge_src_dst] = {}
                if parsed_transition.edge not in formula_edges[edge_src_dst]:
                    formula_edges[edge_src_dst][parsed_transition.edge] = []
                formula = formula_edges[edge_src_dst][parsed_transition.edge]
                for pos_fluent in parsed_transition.pos:
                    formula.append(f"~{pos_fluent}")
                for neg_fluent in parsed_transition.neg:
                    formula.append(neg_fluent)


def _build_automaton(automaton_name, call_edges: Dict[Tuple[str, str], Dict[str, List[str]]],
                     formula_edges: Dict[Tuple[str, str], Dict[int, List[str]]]):
    automaton = HierarchicalAutomaton(automaton_name)
    for edge_src_dst in call_edges:
        for called_automaton in call_edges[edge_src_dst]:
            # form the DNF formula from the edges that are actually associated to formulas (if now formula is associated
            # then the array will remain empty and, thus, equivalent to TRUE)
            dnf_formula_l = []
            for edge_id in call_edges[edge_src_dst][called_automaton]:
                if edge_src_dst in formula_edges and edge_id in formula_edges[edge_src_dst]:
                    dnf_formula_l.append(formula_edges[edge_src_dst][edge_id])
            dnf_formula = DNFFormula(dnf_formula_l)

            src, dst = edge_src_dst
            if called_automaton == LEAF_AUTOMATON:
                automaton.add_formula_edge(src, dst, dnf_formula)
            else:
                automaton.add_call_edge(src, dst, called_automaton, dnf_formula)
    return automaton
