from accord.base import RelationalTree


def n_hop_factory(tree: RelationalTree, title: str = "Reasoning hops") -> dict:
    stats = {"title": title}
    stats.update({i: 0 for i in range(len(tree.templates))})
    return stats


def af_vars_factory(tree: RelationalTree, title: str = "Number AF variables") -> dict:
    max_af_vars = len(tree.unique_variable_ids()) - 2
    stats = {"title": title}
    stats.update({i: n_hop_factory(tree) for i in range(max_af_vars + 1)})
    return stats
