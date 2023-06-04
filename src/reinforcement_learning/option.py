from gym_hierarchical_subgoal_automata.automata.condition import EdgeCondition, FormulaCondition
from gym_hierarchical_subgoal_automata.automata.logic import DNFFormula


class Option:
    """
    Model of an option.
    Source: "Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning",
            Sutton, Precup, Singh (1999)
    """
    def __init__(self, base_condition: EdgeCondition, context: DNFFormula, start_state, automaton_name: str,
                 automaton_state: str, target_automaton_state: str, parent, num_steps=0):
        """
        Creates an option.

        :param base_condition: What the option aims to accomplish, either a conjunction formula or a call.
        :param context: The context under which the option runs (either TRUE or a DNF with a single disjunct).
        :param start_state: The environment state at which the option starts.
        :param automaton_name: The automaton in which the option starts.
        :param automaton_state: The automaton state in which the option starts.
        :param target_automaton_state: The automaton state in which the agent will be if the option is successfully
        run (i.e., the associated call condition has been completed or the associated formula condition has been
        satisfied).
        :param parent: The option being executed at a higher level of (temporal) abstraction.
        :param num_steps: The number of steps for which the option has run so far.
        """
        assert context.is_true() or len(context) == 1

        self._base_condition = base_condition
        self._context = context
        self._condition = None
        self._start_state = start_state
        self._automaton_name = automaton_name
        self._automaton_state = automaton_state
        self._target_automaton_state = target_automaton_state
        self._parent: Option = parent
        self._num_steps = num_steps

    def get_base_condition(self):
        return self._base_condition

    def get_context(self):
        return self._context

    def get_condition(self) -> EdgeCondition:
        """
        Returns the condition that results from unifying the base condition and the context. The difference w.r.t.
        get_base_condition() only occurs for formula conditions, call conditions remain unchanged.
        """
        if self._condition is None:
            if self._base_condition.is_call():
                self._condition = self._base_condition
            else:
                self._condition = FormulaCondition(self._base_condition.get_formula().logic_and(self._context))
        return self._condition

    def get_start_state(self):
        return self._start_state

    def get_automaton_name(self):
        return self._automaton_name

    def get_automaton_state(self):
        return self._automaton_state

    def get_target_automaton_state(self):
        return self._target_automaton_state

    def get_parent(self):
        return self._parent

    def get_num_steps(self):
        return self._num_steps

    def update(self):
        self._num_steps += 1
        if self._parent is not None:
            self._parent.update()
