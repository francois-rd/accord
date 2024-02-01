from dataclasses import dataclass

from .relation import Relation, RelationId, RelationType
from .variable import GenericVariable, Variable, VarId


@dataclass
class GenericTemplate:
    """
    Represents a reasoning skill triple in the abstract: (source, x, target), where x
    is a specific relation in a GenericTree whose RelationType is undetermined, and
    source and target are GenericVariables.

    source: A source GenericVariable.
    relation_id: An identifier for a specific relation (regardless of RelationType).
    target: A target GenericVariable.
    """
    source: GenericVariable
    relation_id: RelationId
    target: GenericVariable


@dataclass
class RelationalTemplate:
    """
    Represents a reasoning skill triple with a specific RelationType. For example,
    (source, 'spatial', target) represents commonsense spatial reasoning) between the
    source and target Variable identifiers.

    source: A source Variable identifier in a RelationalTree.
    relation: A specific Relation's type.
    target: A target Variable identifier in a RelationalTree.
    """
    source_id: VarId
    relation_type: RelationType
    target_id: VarId

    def partial_equals(
            self,
            other: 'RelationalTemplate',
            source_ids: bool = False,
            relation_types: bool = False,
            target_ids: bool = False,
    ) -> bool:
        if source_ids and self.source_id != other.source_id:
            return False
        if relation_types and self.relation_type != other.relation_type:
            return False
        if target_ids and self.target_id != other.target_id:
            return False
        return True


@dataclass
class Template:
    """
    Formalizes a reasoning skill (e.g., commonsense spatial reasoning) into a relational
    triple: (source, relation, target), where all components can be fully instantiated.

    source: A source Variable.
    relation: A Relation between the Variables.
    target: A target Variable.
    """
    source: Variable
    relation: Relation
    target: Variable

    def partial_equals(
            self,
            other: 'Template',
            source_ids: bool = False,
            source_terms: bool = False,
            relation_types: bool = False,
            relation_descriptions: bool = False,
            relation_surface_forms: bool = False,
            target_ids: bool = False,
            target_terms: bool = False,
    ) -> bool:
        if source_ids and self.source.identifier != other.source.identifier:
            return False
        if source_terms and self.source.term != other.source.term:
            return False
        if relation_types and self.relation.type_ != other.relation.type_:
            return False
        if relation_descriptions and \
                self.relation.description != other.relation.description:
            return False
        if relation_surface_forms and \
                self.relation.surface_form != other.relation.surface_form:
            return False
        if target_ids and self.target.identifier != other.target.identifier:
            return False
        if target_terms and self.target.term != other.target.term:
            return False
        return True
