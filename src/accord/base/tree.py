from dataclasses import dataclass
from typing import List, Set

import networkx as nx

from .template import GenericTemplate, RelationalTemplate, Template
from .variable import VarId


@dataclass
class GenericTree:
    """
    An unordered sequence of GenericTemplates with cross-linked variables. A GenericTree
    is valid if and only if the configuration of cross-linked GenericVariables forms a
    valid poly-tree.

    templates: Unordered GenericTemplates with cross-linked GenericVariables.
    """

    templates: List[GenericTemplate]


@dataclass
class RelationalTree:
    """
    An unordered sequence of RelationalTemplates with cross-linked Variables. A
    RelationalTree is valid if and only if the configuration of cross-linked Variables
    forms a valid poly-tree.

    templates: Unordered RelationalTemplates with cross-linked Variable identifiers.
    """

    templates: List[RelationalTemplate]

    def as_graph(self) -> nx.MultiDiGraph:
        """Returns a NetworkX MultiDiGraph representation of this RelationalTree."""
        g = nx.MultiDiGraph()
        for t in self.templates:
            g.add_edge(t.source_id, t.target_id, type_=t.relation_type)
        return g

    def unique_variable_ids(self) -> Set[VarId]:
        """Returns all the unique Variables identifiers in this RelationalTree."""
        variable_ids = set()
        for template in self.templates:
            variable_ids.add(template.source_id)
            variable_ids.add(template.target_id)
        return variable_ids


@dataclass
class Tree:
    """
    An unordered sequence of Templates with cross-linked Variables. A Tree is valid if
    and only if the configuration of cross-linked Variables forms a valid poly-tree.

    templates: Unordered Templates with cross-linked Variables.
    pairing_template: The Template in templates that is paired to a  specific Template
        amongst a specific QAData's pairing_templates.
    """

    templates: List[Template]
    pairing_template: Template
