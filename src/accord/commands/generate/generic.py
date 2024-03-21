from itertools import combinations

from ..configs import FilterConfig, GeneralConfig, ResourcesConfig
from ...base import Case, GenericCaseLink
from ...components import GeneratorFilter
from ...transforms import GenericTreeTransform
from ...io import save_dataclass_jsonl


def generate(resources: ResourcesConfig, general: GeneralConfig, f_cfg: FilterConfig):
    """
    Generates all valid GenericTrees with n_hop GenericTemplates.

    Specifically, generates all possible Case combinations for all possible
    <n_hop choose 2> combinations of generic relations, builds GenericTrees
    from these, then filters out those that are not poly-trees.

    Saves up to <number of Cases> ^ <n_hop choose 2> GenericTrees to file,
    though typically much fewer in practice after filtering.
    """

    def _helper(all_pairs, list_index: int):
        if list_index >= len(all_pairs):
            yield []
        else:
            r1, r2 = all_pairs[list_index]
            for case in Case:
                for res in _helper(all_pairs, list_index + 1):
                    val = [GenericCaseLink(r1, r2, case)]
                    val.extend(res)
                    yield val

    g_filter = GeneratorFilter(f_cfg.generic_prob, seed=general.random_seed)
    trees, transform = [], GenericTreeTransform()
    relations = [f"R{i}" for i in range(resources.tree_size)]
    for case_link_list in g_filter(_helper(list(combinations(relations, 2)), 0)):
        tree = transform(case_link_list)
        if tree is not None:
            trees.append(tree)
    if general.verbose:
        print(f"Total number of generic trees generated: {len(trees)}")
    save_dataclass_jsonl(resources.generic_trees_file, *trees)
