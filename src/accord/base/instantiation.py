from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .relation import Relation, RelationType
from .template import RelationalTemplate, Template
from .tree import RelationalTree, Tree
from .variable import Term, Variable, VarId

InstantiationId = str
InstantiationMap = Dict[VarId, Term]


@dataclass
class InstantiationData:
    """
    All necessary information for instantiating a RelationalTree into a Tree.

    identifier: A unique identifier (like 'I1' or 'I2') to differentiate it from others.
    pairing_template: The RelationalTemplate in a RelationalTree's templates to pair
        to a specific Template amongst a specific QAData's pairing_templates.
    pairing: A tuple containing the identifier of the Variable in pairing_template
        whose Term value to fix, as well as said fixed Term value.
    qa_template: The specific Template amongst a specific QAData's pairing_templates to
        pair with pairing_template.
    answer_id: The identifier of the Variable in the RelationalTree's templates to
        instantiate using a specific QAData's answer_choices.
    reasoning_hops: The number of reasoning hops between the pairing variable and the
        answer variable; or -1 if unknown
    anti_factual_ids: The identifiers of all Variables in the RelationalTree's
        templates to instantiate anti-factually, rather than factually. Can be empty.
    mapping: A mapping between all Variable identifiers and their respective
        instantiation Terms.
    """

    identifier: Optional[InstantiationId] = None
    pairing_template: Optional[RelationalTemplate] = None
    pairing: Optional[Tuple[VarId, Term]] = None
    qa_template: Optional[Template] = None
    answer_id: Optional[VarId] = None
    reasoning_hops: int = -1
    anti_factual_ids: Optional[List[VarId]] = None
    mapping: Optional[InstantiationMap] = None

    def partial_equals(
        self,
        other: Optional["InstantiationData"],
        ids: bool = False,
        template_source_ids: bool = False,
        template_relation_types: bool = False,
        template_target_ids: bool = False,
        pairing_ids: bool = False,
        pairing_terms: bool = False,
        answer_ids: bool = False,
        anti_factual_ids: bool = False,
        mappings: bool = False,
    ) -> bool:
        if other is None:
            return False
        if ids and self.identifier != other.identifier:
            return False
        if template_source_ids or template_relation_types or template_target_ids:
            if self.pairing_template is None and other.pairing_template is not None:
                return False
            if not self.pairing_template.partial_equals(
                other.pairing_template,
                source_ids=template_source_ids,
                relation_types=template_relation_types,
                target_ids=template_target_ids,
            ):
                return False
        if pairing_ids and self.pairing[0] != other.pairing[0]:
            return False
        if pairing_terms and self.pairing[1] != other.pairing[1]:
            return False
        if answer_ids and self.answer_id != other.answer_id:
            return False
        if (
            anti_factual_ids
            and sorted(self.anti_factual_ids) != sorted(other.anti_factual_ids)
        ):
            return False
        if mappings and self.mapping != other.mapping:
            return False
        return True

    def mapping_distance(
        self,
        other: "InstantiationData",
        count_answer_ids: bool = False,
        count_pairing_ids: bool = False,
    ) -> int:
        count = 0
        blacklist = []
        if not count_answer_ids:
            blacklist.extend([self.answer_id, other.answer_id])
        if not count_pairing_ids:
            blacklist.extend([self.pairing[0], other.pairing[0]])
        for var_id, term in self.mapping.items():
            if other.mapping[var_id] != term and var_id not in blacklist:
                count += 1
        return count

    def instantiate(
        self,
        tree: RelationalTree,
        relation_map: Dict[RelationType, Relation],
    ) -> Tree:
        templates, pairing_template = [], None
        for template in tree.templates:
            # Instantiate the template and add it to the list.
            new_template = self._instantiate_template(template, relation_map)
            templates.append(new_template)

            # If it's the pairing template, replace its relation with the specific
            # variant from the QAData.
            if template == self.pairing_template:
                new_template.relation = self.qa_template.relation
                pairing_template = new_template
        if pairing_template is None:
            raise ValueError("Pairing template not found.")
        return Tree(templates, pairing_template)

    def _instantiate_template(
        self,
        template: RelationalTemplate,
        relation_map: Dict[RelationType, Relation],
    ) -> Template:
        return Template(
            source=Variable(
                identifier=template.source_id,
                term=self.mapping[template.source_id],
            ),
            relation=relation_map[template.relation_type],
            target=Variable(
                identifier=template.target_id,
                term=self.mapping[template.target_id],
            ),
        )


@dataclass
class InstantiationFamily:
    tree: RelationalTree
    data_ids: List[InstantiationId] = field(default_factory=list)
    data_map: Optional[Dict[InstantiationId, InstantiationData]] = None

    def add(self, instantiation_id: InstantiationId):
        self.data_ids.append(instantiation_id)


@dataclass
class InstantiationForest:
    families: List[InstantiationFamily] = field(default_factory=list)
    data_map: Dict[InstantiationId, InstantiationData] = field(default_factory=dict)

    def add_family(self, tree: RelationalTree) -> InstantiationFamily:
        family = InstantiationFamily(tree)
        self.families.append(family)
        return family

    def add_data(self, data: InstantiationData):
        self.data_map[data.identifier] = data

    def map_family_data(self):
        for family in self.families:
            family.data_map = {id_: self.data_map[id_] for id_ in family.data_ids}
