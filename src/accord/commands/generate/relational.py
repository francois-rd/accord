from typing import List, Iterable
from itertools import product
import networkx as nx

from ..configs import ResourcesConfig
from ...base import GenericTree, RelationalTree
from ...transforms import RelationalTransform
from ...io import save_dataclass_jsonl, load_dataclass_jsonl, load_relations_csv


class Generate:
    def __init__(self, resources: ResourcesConfig):
        self.trees = load_dataclass_jsonl(resources.generic_trees_file, GenericTree)
        self.relations = load_relations_csv(resources.relations_file)
        self.resources = resources

    def run(self):
        save_dataclass_jsonl(self.resources.relational_trees_file, *self._transform())

    def _transform(self) -> List[RelationalTree]:
        """Transform all trees."""
        tree_groups = {}
        for n_relations in product(self.relations, repeat=self.resources.tree_size):
            transform = RelationalTransform(n_relations)
            group_key = tuple(sorted(set([r.type_ for r in n_relations])))
            for tree in self.trees:
                new_tree = transform(tree)
                if new_tree is not None:
                    tree_groups.setdefault(group_key, []).append(new_tree)
        return list(self._remove_isomorphic_trees(tree_groups.values()))

    def _remove_isomorphic_trees(
            self,
            grouped_trees: Iterable[Iterable[RelationalTree]],
    ) -> List[RelationalTree]:
        """Remove isomorphic (i.e., duplicate) trees based on matched relation types."""
        for tree_group in grouped_trees:
            unique_trees = []
            for tree in tree_group:
                isomorphic = False
                for other_tree in unique_trees:
                    tree_graph, other_graph = tree.as_graph(), other_tree.as_graph()
                    if nx.is_isomorphic(tree_graph, other_graph, edge_match=self._em):
                        isomorphic = True
                        break
                if not isomorphic:
                    unique_trees.append(tree)
            yield from unique_trees

    @staticmethod
    def _em(e1, e2):
        v1 = {v['type_'] for v in e1.values()}
        v2 = {v['type_'] for v in e2.values()}
        return v1 == v2
