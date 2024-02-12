from typing import Any, Dict, Hashable, List, Optional
from dataclasses import dataclass
from copy import deepcopy

from .template import Template
from .tree import Tree
from .variable import Term
from .instantiation import (
    InstantiationData,
    InstantiationForest,
    InstantiationId,
    InstantiationMap,
)


Label = str
QAGroupId = str


@dataclass
class QAData:
    """
    Contains all relevant data for a single QA sample/datum from a larger QA dataset.

    identifier: A unique identifier differentiating this QA data sample from others.
    question: The question text of the QA data sample.
    correct_answer_label: The key (label) into 'answer_choices' of the correct answer.
    answer_choices: The answer choices (label, term) of the QA data sample.
    pairing_templates: The Templates can that be used for pairing the QA data sample to
        a RelationalTree. Each such pairing template should have exactly one Variable
        that is free and one that is instantiated with a Term. At least one of the
        pairing_templates must fit at least one of the templates in a given tree for
        the two to be paired. Fitting means that the two templates' RelationTypes are
        the same (though not necessarily the whole Relation).
    kwargs: Other relevant parameters, if any.
    """

    identifier: str
    question: str
    correct_answer_label: Label
    answer_choices: Dict[Label, Term]
    pairing_templates: List[Template]
    kwargs: Dict[Hashable, Any]

    def __post_init__(self):
        for template in self.pairing_templates:
            s, t = template.source.term, template.target.term
            if (s is None and t is None) or (s is not None and t is not None):
                raise ValueError("Pairing template must have one free variable.")


@dataclass
class QAGroup:
    identifier: QAGroupId
    data_ids: Dict[Label, InstantiationId]
    mapping_map: Dict[Label, Optional[InstantiationMap]]
    data_map: Optional[Dict[Label, InstantiationData]] = None

    def instantiate(self, forest: InstantiationForest):
        self.data_map = {}
        for label, mapping in self.mapping_map.items():
            if mapping is None:
                self.data_map[label] = forest.data_map[self.data_ids[label]]
            else:
                self.data_map[label] = deepcopy(forest.data_map[self.data_ids[label]])
                self.data_map[label].mapping = deepcopy(mapping)


@dataclass
class QAPrompt:
    qa_data: QAData
    qa_group: QAGroup
    tree_map: Dict[Label, Tree]
