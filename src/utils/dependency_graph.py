from typing import Dict, Set, Tuple


class DependencyGraph:
    """
    Class modelling a dependency graph, which is used to show the dependencies between automata in the main code.
    Nice inspiration for an API: https://www.npmjs.com/package/dependency-graph.
    """
    def __init__(self):
        self._node_to_dependants: Dict[str, Set[str]] = {}
        self._banned_dependencies: Set[Tuple[str, str]] = set()

    def add_node(self, node: str):
        if node not in self._node_to_dependants:
            self._node_to_dependants[node] = set()

    def add_dependency(self, dependant_node: str, dependency_node: str):  # depends on
        assert dependency_node in self._node_to_dependants
        self._node_to_dependants[dependency_node].add(dependant_node)

    def add_banned_dependency(self, dependant_node: str, dependency_node: str):  # cannot depend on
        self._banned_dependencies.add((dependant_node, dependency_node))

    def is_dependant_on(self, dependant_node: str, dependency_node: str):
        """Returns true if dependant_node has dependency_node as a dependency."""
        assert dependency_node in self._node_to_dependants
        dependants_list = self._node_to_dependants[dependency_node]
        if dependant_node in dependants_list:
            return True
        for node in dependants_list:
            if self.is_dependant_on(dependant_node, node):
                return True
        return False

    def is_directly_dependant_on(self, dependant_node: str, dependency_node: str):
        """
        Returns true if dependant_node has a direct dependency on dependency_node (i.e., there is a single directed
        edge between them.
        """
        assert dependency_node in self._node_to_dependants
        return dependant_node in self._node_to_dependants[dependency_node]

    def is_banned_dependency(self, dependant_node: str, dependency_node: str):
        return (dependant_node, dependency_node) in self._banned_dependencies

    def clean_dependencies(self, dependant_node: str):
        for node in self._node_to_dependants:
            dependants = self._node_to_dependants[node]
            if dependant_node in dependants:
                dependants.remove(dependant_node)

    def get_dependencies(self, dependant_node: str):
        dependencies = set()
        self._get_dependencies_helper(dependant_node, dependencies)
        return dependencies

    def _get_dependencies_helper(self, dependant_node: str, dependencies: Set[str]):
        for node in self._node_to_dependants:
            dependants = self._node_to_dependants[node]
            if dependant_node in dependants:
                dependencies.add(node)
                self._get_dependencies_helper(node, dependencies)


if __name__ == "__main__":
    g = DependencyGraph()
    g.add_node("m0")
    g.add_node("m1")
    g.add_node("m2")
    g.add_node("m3")

    g.add_dependency("m0", "m1")
    g.add_dependency("m1", "m2")
    g.add_dependency("m0", "m3")

    from itertools import permutations

    for n1, n2 in permutations(["m0", "m1", "m2", "m3"], r=2):
        print(f"{n1} depends on {n2}:", g.is_dependant_on(n1, n2))

    for n in ["m0", "m1", "m2", "m3"]:
        print(f"Dependencies of {n}: {g.get_dependencies(n)}")
