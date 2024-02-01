from dataclasses import dataclass
from typing import Optional


Term = str
VarId = str


@dataclass
class GenericVariable:
    """
    A term variable for a GenericReasoningTemplate.

    identifier: A unique identifier (like 'V1' or 'V2') to differentiate it from others.
    parent: A variable can have exactly one parent (another variable from which this
        variable will copy its term instantiation value)
    has_children: A variable can have any number of children (vice versa to parent),
        the existence of which is used while building up the GenericReasoningTree.
    """
    identifier: Optional[VarId] = None
    parent: Optional['GenericVariable'] = None
    has_children: bool = False


@dataclass
class Variable:
    """
    A term variable for a ReasoningTemplate.

    identifier: A unique identifier (like 'V1' or 'V2') to differentiate it from others.
    term: Instantiation of this variable using a Term (i.e., a concept taken from a
        specific database of such concepts).
    """
    identifier: VarId
    term: Optional[Term] = None
