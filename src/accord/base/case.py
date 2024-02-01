from dataclasses import dataclass
from enum import Enum

from .relation import RelationId, RelationType
from .template import RelationalTemplate


class Case(Enum):
    """
    A permutation Case linking two Relations.

    The five permutation Cases are:
        Case 0: A relation1 B; C relation2 D (no linking relation)
        Case 1: A relation1 B; B relation2 C (relation1.target == relation2.source)
        Case 2: A relation1 B; C relation2 B (relation1.target == relation2.target)
        Case 3: B relation1 A; B relation2 C (relation1.source == relation2.source)
        Case 4: B relation1 A; C relation2 B (relation1.source == relation2.target)

    NOTE: The Cases are equivalent with respect to permutation of the Relations (under
    the assumption of order invariance of the Relations). Specifically:
        Case 0: A relation1 B; C relation2 D (original)
             => A relation2 B; C relation1 D (permute relations)
             => C relation1 D; A relation2 B (assumption: order invariance)
             => A relation1 B; C relation2 D (change of variable)
             => Case 0
        Case 1: A relation1 B; B relation2 C (original)
             => A relation2 B; B relation1 C (permute relations)
             => B relation1 C; A relation2 B (assumption: order invariance)
             => B relation1 A; C relation2 B (change of variable)
             => Case 4
        Case 2: A relation1 B; C relation2 B (original)
             => A relation2 B; C relation1 B (permute relations)
             => C relation1 B; A relation2 B (assumption: order invariance)
             => A relation1 B; C relation2 B (change of variable)
             => Case 2
        Case 3: B relation1 A; B relation2 C (original)
             => B relation2 A; B relation1 C (permute relations)
             => B relation1 C; B relation2 A (assumption: order invariance)
             => B relation1 A; B relation2 C (change of variable)
             => Case 3
        Case 4: B relation1 A; C relation2 B (original)
             => B relation2 A; C relation1 B (permute relations)
             => C relation1 B; B relation2 A (assumption: order invariance)
             => A relation1 B; B relation2 C (change of variable)
             => Case 1
    """

    """Case 0: A relation1 B; C relation2 D (no linking relation)"""
    ZERO = 0

    """Case 1: A relation1 B; B relation2 C (relation1.target == relation2.source)"""
    ONE = 1

    """Case 2: A relation1 B; C relation2 B (relation1.target == relation2.target)"""
    TWO = 2

    """Case 3: B relation1 A; B relation2 C (relation1.source == relation2.source)"""
    THREE = 3

    """Case 4: B relation1 A; C relation2 B (relation1.source == relation2.target)"""
    FOUR = 4

    def equivalent(self) -> 'Case':
        """
        Returns the equivalent permutation Case under the assumption of
        order-invariant Relations.
        """
        if self == Case.ONE:
            return Case.FOUR
        elif self == Case.ZERO or self == Case.TWO or self == Case.THREE:
            return self
        elif self == Case.FOUR:
            return Case.ONE
        else:
            raise TypeError(f"Unsupported permutation case: {self}")


@dataclass
class GenericCaseLink:
    """
    A GenericCaseLink is a 2-hop permutation Case linking two generic relations.

    In other words, it is a permutation Case in addition to the two specific
    relations (regardless of specific type) to which the Case is being applied.

    r1_id: An identifier for the first specific relation (regardless of RelationType)
    r2_id: An identifier for the second specific relation (regardless of RelationType)
    case: The permutation Case linking the two relations.
    """
    r1_id: RelationId
    r2_id: RelationId
    case: Case


@dataclass(frozen=True)
class RelationalCaseLink:
    """
    A RelationalCaseLink is a 2-hop permutation Case linking two RelationTypes.

    In other words, it is a permutation Case in addition to the two specific
    RelationTypes (regardless of specific instance) to which the Case is being applied.

    r1_type: The type of the first Relation (regardless of specific Relation instance).
    r2_type: The type of the second Relation (regardless of specific Relation instance).
    case: The permutation Case linking the two RelationTypes.
    """
    r1_type: RelationType
    r2_type: RelationType
    case: Case

    def equivalent(self) -> 'RelationalCaseLink':
        """Returns the equivalent CaseLink under assumed Relation permutation."""
        return RelationalCaseLink(self.r2_type, self.r1_type, self.case.equivalent())

    @staticmethod
    def from_templates(
            t1: RelationalTemplate,
            t2: RelationalTemplate,
    ) -> 'RelationalCaseLink':
        # Figure out which Case we are dealing with. Robustly check for circular
        # template links as otherwise we can't be sure case checks are unique.
        t1_source, t1_target = t1.source_id, t1.target_id
        t2_source, t2_target = t2.source_id, t2.target_id
        if t1_target == t2_source:
            if t1_source in [t1_target, t2_target] or t2_target == t2_source:
                raise ValueError(f"Circular template links: t1={t1}; t2={t2}")
            return RelationalCaseLink(t1.relation_type, t2.relation_type, Case.ONE)
        elif t1_target == t2_target:
            if t1_source in [t1_target, t2_source] or t2_source == t2_target:
                raise ValueError(f"Circular template links: t1={t1}; t2={t2}")
            return RelationalCaseLink(t1.relation_type, t2.relation_type, Case.TWO)
        elif t1_source == t2_source:
            if t1_target in [t1_source, t2_target] or t2_target == t2_source:
                raise ValueError(f"Circular template links: t1={t1}; t2={t2}")
            return RelationalCaseLink(t1.relation_type, t2.relation_type, Case.THREE)
        elif t1_source == t2_target:
            if t1_target in [t1_source, t2_source] or t2_source == t2_target:
                raise ValueError(f"Circular template links: t1={t1}; t2={t2}")
            return RelationalCaseLink(t1.relation_type, t2.relation_type, Case.FOUR)
        else:
            if t1_source == t2_target or t2_source == t2_target:
                raise ValueError(f"Circular template links: t1={t1}; t2={t2}")
            return RelationalCaseLink(t1.relation_type, t2.relation_type, Case.ZERO)
