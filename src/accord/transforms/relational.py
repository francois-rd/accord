from typing import Optional, Tuple

from ..base import GenericTree, Relation, RelationalTemplate, RelationalTree


class RelationalTransform:
    def __init__(self, relations: Tuple[Relation]):
        self.relations = relations

    def __call__(self, tree: GenericTree) -> Optional[RelationalTree]:
        # Sanity check on input.
        if len(tree.templates) != len(self.relations):
            raise ValueError("Number of relations must match number of tree templates.")

        # Transform generic tree into relational tree.
        new_templates = []
        for template, relation in zip(tree.templates, self.relations):
            new_templates.append(RelationalTemplate(
                source_id=template.source.identifier,
                relation_type=relation.type_,
                target_id=template.target.identifier,
            ))
        return RelationalTree(new_templates)
