from accord.base import RelationalTree


def max_hops(tree: RelationalTree) -> int:
    return len(tree.templates)


def n_hop_factory(tree: RelationalTree, title: str = "Reasoning hops") -> dict:
    stats = {"title": title}
    stats.update({i: 0 for i in range(max_hops(tree) + 1)})
    return stats


def max_af_vars(tree: RelationalTree) -> int:
    return len(tree.unique_variable_ids()) - 2


def af_vars_factory(tree: RelationalTree, title: str = "Number AF variables") -> dict:
    stats = {"title": title}
    stats.update({i: n_hop_factory(tree) for i in range(max_af_vars(tree) + 1)})
    return stats
