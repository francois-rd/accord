from typing import Dict, Optional, Tuple

from ..base import GenericTree, Relation, RelationalTemplate, RelationalTree
from ..components import GeneratorFilter, Reducer


class RelationalTransform:
    def __init__(
        self,
        relations: Tuple[Relation],
        reducer: Optional[Reducer],
        filters: Dict[str, GeneratorFilter],
    ):
        self.relations = relations
        self.reducer = reducer
        self.filters = filters
        self.default_filter = GeneratorFilter(0.0)

    def __call__(self, tree: GenericTree) -> Optional[RelationalTree]:
        # Sanity check on input.
        if len(tree.templates) != len(self.relations):
            raise ValueError("Number of relations must match number of tree templates.")

        # Transform generic tree into relational tree.
        new_templates = []
        for template, relation in zip(tree.templates, self.relations):
            new_templates.append(
                RelationalTemplate(
                    source_id=template.source.identifier,
                    relation_type=relation.type_,
                    target_id=template.target.identifier,
                )
            )

        # Probabilistically filter tree based on maximum reasoning hops it can achieve.
        new_tree = RelationalTree(new_templates)
        if self.reducer is None:
            return new_tree
        max_hops = []
        for template in new_tree.templates:
            for id_ in [template.source_id, template.target_id]:
                ids_and_hops = self.reducer.valid_answer_ids(
                    new_tree, template, id_, return_reasoning_hops=True,
                )
                max_hops.append(max([hops for _, hops in ids_and_hops]))
        if self.filters.get(str(max(max_hops)), self.default_filter).passes():
            return new_tree
        return None
