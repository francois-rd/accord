from typing import Iterable, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from ..base import (
    Case,
    Relation,
    RelationalCaseLink,
    RelationalTemplate,
    RelationalTree,
    RelationType,
    VarId,
)


class ReductionOrder(Enum):
    MAINTAIN = "MAINTAIN"
    REVERSE = "REVERSE"

    def __invert__(self) -> "ReductionOrder":
        if self == ReductionOrder.MAINTAIN:
            return ReductionOrder.REVERSE
        elif self == ReductionOrder.REVERSE:
            return ReductionOrder.MAINTAIN
        else:
            raise TypeError(f"Unsupported reduction order: {self}")


@dataclass(frozen=True)
class Reduction:
    relation_type: RelationType
    order: ReductionOrder

    def __invert__(self) -> "Reduction":
        return Reduction(self.relation_type, ~self.order)


class Reducer:
    def __init__(self, relations: Iterable[Relation]):
        self.relation_types = [r.type_ for r in relations]
        self.permutations = {}

    def register(
        self,
        case_link: RelationalCaseLink,
        reduction: Reduction,
        raise_: bool = False,
    ):
        """
        Registers a specific RelationalCaseLink and its Reduction.

        Permutation Case equivalencies are detected automatically. Attempting to
        register an identical or equivalent RelationalCaseLink to one that has already
        been registered is silently ignored, or optionally, an error can be raised.
        """
        equiv = case_link.equivalent()
        if case_link in self.permutations or equiv in self.permutations:
            if raise_:
                raise ValueError(f"{case_link} has already been added")
        else:
            self.permutations[case_link] = reduction

    def reduce_case_link(self, case_link: RelationalCaseLink) -> Optional[Reduction]:
        """
        Returns the Reduction for a specific RelationalCaseLink (or its equivalent).
        """
        equiv = case_link.equivalent()
        if case_link in self.permutations:
            return self.permutations[case_link]
        elif equiv in self.permutations:
            return ~self.permutations[equiv]
        else:
            return None

    def reduce_templates(
        self,
        t1: RelationalTemplate,
        t2: RelationalTemplate,
    ) -> Optional[RelationalTemplate]:
        # Figure out which Case we are dealing with.
        case_link = RelationalCaseLink.from_templates(t1, t2)
        if case_link.case == Case.ZERO:
            first, second = None, None
        elif case_link.case == Case.ONE:
            first, second = t1.source_id, t2.target_id
        elif case_link.case == Case.TWO:
            first, second = t1.source_id, t2.source_id
        elif case_link.case == Case.THREE:
            first, second = t1.target_id, t2.target_id
        elif case_link.case == Case.FOUR:
            first, second = t1.target_id, t2.source_id
        else:
            raise ValueError(f"Unsupported value for case: {case_link.case}")

        # Return a new ReasoningTemplate based on the Reduction results.
        reduction = self.reduce_case_link(case_link)
        if reduction is None or reduction.relation_type not in self.relation_types:
            return None
        if reduction.order == ReductionOrder.REVERSE:
            first, second = second, first  # NOTE: No need to temp variable in Python.
        return RelationalTemplate(first, reduction.relation_type, second)

    def valid_answer_ids(
        self,
        tree: RelationalTree,
        pairing_template: RelationalTemplate,
        pairing_id: VarId,
        return_counts: bool = False,
    ) -> List[Union[VarId, Tuple[VarId, int]]]:
        r = self._answer_ids_and_counts(tree.templates, pairing_template, pairing_id, 0)
        return [id_count if return_counts else id_count[0] for id_count in r]

    def _answer_ids_and_counts(
        self,
        templates: List[RelationalTemplate],
        pairing_template: RelationalTemplate,
        pairing_id: VarId,
        count: int,
    ) -> Iterable[Tuple[VarId, int]]:
        if pairing_template in templates:
            if pairing_id == pairing_template.source_id:
                yield pairing_template.target_id, count
            elif pairing_id == pairing_template.target_id:
                yield pairing_template.source_id, count
            else:
                raise ValueError("Pairing variable not in pairing template.")
        for template in templates:
            if template != pairing_template:
                new_template = self.reduce_templates(template, pairing_template)
                if self._is_valid_match(new_template, pairing_template, pairing_id):
                    to_remove = [template, pairing_template]
                    to_keep = [t for t in templates if t not in to_remove]
                    to_keep.append(new_template)
                    yield from self._answer_ids_and_counts(
                        to_keep, new_template, pairing_id, count + 1
                    )

    @staticmethod
    def _is_valid_match(
        test_template: RelationalTemplate,
        pairing_template: RelationalTemplate,
        pairing_id: VarId,
    ) -> bool:
        if test_template is None:
            return False
        if not test_template.partial_equals(pairing_template, relation_types=True):
            return False
        return pairing_id in [test_template.source_id, test_template.target_id]
