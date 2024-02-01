from dataclasses import dataclass

RelationId = str
RelationType = str


@dataclass
class Relation:
    """
    Represents a relation linking two Variables.

    type_: Represents the type of relation in the abstract (e.g., 'causal').
    description: A free-form text description of the relation (for humans).
    surface_form: A free-form text instantiation of the relation (for LLMs).
    """

    type_: RelationType
    description: str
    surface_form: str
