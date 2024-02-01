from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

import networkx as nx

from .template import GenericTemplate, RelationalTemplate, Template
from .variable import Term, VarId


InstantiationMap = Dict[VarId, Term]


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
class InstantiationData:
    """
    All necessary information for instantiating a RelationalTree into a Tree.

    pairing_template: The RelationalTemplate in the RelationalTree's templates to pair
        to a specific Template amongst a specific QAData's pairing_templates.
    pairing: A tuple containing the identifier of the Variable in pairing_template
        whose Term value to fix, as well as said fixed Term value.
    answer_id: The identifier of the Variable in the RelationalTree's templates to
        instantiate using a specific QAData's answer_choices.
    anti_factual_ids: The identifiers of all Variables in the RelationalTree's
        templates to instantiate anti-factually, rather than factually. Can be empty.
    mapping: A mapping between all Variable identifiers and their respective
        instantiation Terms.
    """

    pairing_template: Optional[RelationalTemplate] = None
    pairing: Optional[Tuple[VarId, Term]] = None
    answer_id: Optional[VarId] = None
    anti_factual_ids: Optional[List[VarId]] = None
    mapping: Optional[InstantiationMap] = None


@dataclass
class InstantiationFamily:
    tree: RelationalTree
    instantiations: List[InstantiationData] = field(default_factory=list)

    def add(self, instantiation: InstantiationData):
        self.instantiations.append(instantiation)


@dataclass
class InstantiationForest:
    families: List[InstantiationFamily] = field(default_factory=list)

    def add_family(self, tree: RelationalTree):
        family = InstantiationFamily(tree)
        self.families.append(family)
        return family


@dataclass
class Tree:
    """
    An unordered sequence of Templates with cross-linked Variables. A Tree is valid if
    and only if the configuration of cross-linked Variables forms a valid poly-tree.

    templates: Unordered Templates with cross-linked Variables.
    data: InstantiationData necessary for instantiating the templates of this Tree.
    """

    templates: List[Template]
    data: InstantiationData

    def instantiate(self) -> "Tree":
        for template in self.templates:
            template.source.term = self.data.mapping[template.source.identifier]
            template.target.term = self.data.mapping[template.target.identifier]
        return self
