from typing import Dict, Iterable, List, Set
from itertools import product
from enum import Enum

from .sorter import QueryResultSorter
from .instantiator import Instantiator, InstantiatorVariant, Query
from ..base import InstantiationMap, RelationalTree, Term, VarId


InstantiationOrder = Dict[VarId, int]


class BeamSearchProtocol(Enum):
    AF_IN_LINE = "AF_IN_LINE"
    AF_POST_HOC = "AF_POST_HOC"


class BeamSearch:
    def __init__(
            self,
            factual_instantiator: Instantiator,
            anti_factual_instantiator: Instantiator,
            protocol: BeamSearchProtocol,
            sorter: QueryResultSorter,
            top_k: int = 0,
    ):
        self.factual_instantiator = factual_instantiator
        self.anti_factual_instantiator = anti_factual_instantiator
        self.protocol = protocol
        self.sorter = sorter
        self.top_k = top_k

    def __call__(
            self,
            tree: RelationalTree,
            anti_factual_ids: List[VarId],
            seed_mapping: InstantiationMap,
            *args,
            **kwargs,
    ) -> Iterable[InstantiationMap]:
        if self.protocol == BeamSearchProtocol.AF_IN_LINE:
            fn = self._do_inline
        elif self.protocol == BeamSearchProtocol.AF_POST_HOC:
            fn = self._do_post_hoc
        else:
            raise ValueError(
                f"Unsupported value for beam search protocol: {self.protocol}"
            )
        order = {k: 0 for k in seed_mapping}
        yield from fn(tree, anti_factual_ids, seed_mapping, order, 1, *args, **kwargs)

    def _do_inline(
            self,
            tree: RelationalTree,
            anti_factual_ids: List[VarId],
            fixed_mapping: InstantiationMap,
            instantiation_order: InstantiationOrder,
            count: int,
            *args,
            **kwargs,
    ) -> Iterable[InstantiationMap]:
        candidate_mapping = self._query_frontier_variables(
            tree, anti_factual_ids, fixed_mapping, *args, **kwargs,
        )
        if candidate_mapping:
            # NOTE: If any list in values (which is a dict_values(List[str]) object)
            # if empty, then product() returns an empty iterator, which is the same as
            # skipping the yield for this mapping (which is what we want).
            keys, values = zip(*candidate_mapping.items())
            for instantiation in product(*values):
                new_mapping = {**fixed_mapping, **dict(zip(keys, instantiation))}
                # Skip mapping with non-unique terms for each variable.
                if len(set(new_mapping.values())) == len(new_mapping.values()):
                    new_order = {**instantiation_order, **{k: count for k in keys}}
                    yield from self._do_inline(
                        tree, anti_factual_ids, new_mapping, new_order, count + 1,
                        *args, **kwargs,
                    )
        else:  # If every variable is mapped. NOT "if some variable has empty mapping".
            # Skip mapping with non-unique terms for each variable.
            if len(set(fixed_mapping.values())) == len(fixed_mapping.values()):
                is_valid = self._is_valid_mapping(
                    tree, anti_factual_ids, fixed_mapping, instantiation_order,
                    *args, **kwargs,
                )
                if is_valid:
                    yield fixed_mapping

    def _do_post_hoc(
            self,
            tree: RelationalTree,
            anti_factual_ids: List[VarId],
            fixed_mapping: InstantiationMap,
            instantiation_order: InstantiationOrder,
            count: int,
            *args,
            **kwargs,
    ) -> Iterable[InstantiationMap]:
        # For each factual mapping, find all combinations of re-mappings for AF vars.
        all_factual_mappings = self._do_inline(
            tree, anti_factual_ids, fixed_mapping, instantiation_order, count,
            *args, **kwargs,
        )
        for mapping in all_factual_mappings:
            # For each Af variable, find all possible valid AF instantiations.
            af_mapping = {}
            for af_var in anti_factual_ids:
                # For a given AF variable, find which templates it is in and, for each
                # one, find its partner and query the AF instantiator based on the
                # partner's current mapping.
                self.sorter.new_collection(
                    InstantiatorVariant.ANTI_FACTUAL, *args, **kwargs,
                )
                af_term = mapping[af_var]
                for template in tree.templates:
                    if af_var == template.source_id:
                        partner_var = template.target_id
                    elif af_var == template.target_id:
                        partner_var = template.source_id
                    else:
                        continue
                    q = Query(template, af_var, mapping[partner_var])
                    result = self.anti_factual_instantiator.query(q, *args, **kwargs)
                    self.sorter.add_query_result(
                        result, q, *args, **kwargs, query_existing_term=af_term,
                    )
                af_mapping[af_var] = self._clean_up()

            # Given a specific factual mapping, af_mapping now contains all possible
            # valid AF instantiations of each AF variable. Yield a new mapping for each
            # possible product of all these values.
            if af_mapping:
                # NOTE: If any list in values (which is a dict_values(List[str]) object)
                # if empty, then product() returns an empty iterator, which is the same
                # as skipping the yield for this mapping (which is what we want).
                keys, values = zip(*af_mapping.items())
                for instantiation in product(*values):
                    new_mapping = mapping.copy()
                    new_mapping.update(dict(zip(keys, instantiation)))
                    # Skip mapping with non-unique terms for each variable.
                    if len(set(new_mapping.values())) == len(new_mapping.values()):
                        yield new_mapping
            else:  # If there are no AF vars. NOT "if some AF var has empty mapping".
                # NOTE: Technically, no need to check for non-unique terms, since
                # _do_inline() did it already. Keeping this here for robustness.
                # Skip mapping with non-unique terms for each variable.
                if len(set(mapping.values())) == len(mapping.values()):
                    yield mapping

    def _query_frontier_variables(
            self,
            tree: RelationalTree,
            anti_factual_ids: List[VarId],
            fixed_mapping: InstantiationMap,
            *args,
            **kwargs,
    ) -> Dict[VarId, Set[Term]]:
        # Find frontier variables (variables whose relational partner is fixed).
        frontier_variables = {}
        for template in tree.templates:
            source, target = template.source_id, template.target_id
            for partner, test in [(source, target), (target, source)]:
                if partner in fixed_mapping:
                    if test not in fixed_mapping:
                        templates_ = frontier_variables.setdefault(test, {})
                        if partner in templates_:
                            raise ValueError("ReasoningTree has a cycle")
                        templates_[partner] = template

        # For each frontier variable and each fixed relational partner, query the
        # appropriate instantiator to find all candidate instantiations for the frontier
        # variable. Then, only keep those that are common across relational partners.
        candidate_mapping = {}
        do_inline = self.protocol == BeamSearchProtocol.AF_IN_LINE
        for frontier_var, templates in frontier_variables.items():
            if frontier_var in anti_factual_ids and do_inline:
                instantiator = self.anti_factual_instantiator
                self.sorter.new_collection(
                    InstantiatorVariant.ANTI_FACTUAL, *args, **kwargs,
                )
            else:
                instantiator = self.factual_instantiator
                self.sorter.new_collection(InstantiatorVariant.FACTUAL, *args, **kwargs)
            for partner_var, template in templates.items():
                q = Query(template, frontier_var, fixed_mapping[partner_var])
                result = instantiator.query(q, *args, **kwargs)
                self.sorter.add_query_result(result, q, *args, **kwargs)
            candidate_mapping[frontier_var] = self._clean_up()
        return candidate_mapping

    def _clean_up(self) -> Set[Term]:
        results = self.sorter.sort_collection()
        return set(results[:self.top_k] if 0 < self.top_k < len(results) else results)

    def _is_valid_mapping(
            self,
            tree: RelationalTree,
            anti_factual_ids: List[VarId],
            fixed_mapping: InstantiationMap,
            instantiation_order: InstantiationOrder,
            *args,
            **kwargs,
    ) -> bool:
        factual_query = self.factual_instantiator.query
        do_inline = self.protocol == BeamSearchProtocol.AF_IN_LINE
        for template in tree.templates:
            q = Query(template, template.source_id, fixed_mapping[template.target_id])
            query_term = fixed_mapping[template.source_id]
            source, target = template.source_id, template.target_id
            for query_id, partner in [(source, target), (target, source)]:
                if instantiation_order[query_id] > instantiation_order[partner]:
                    if query_id in anti_factual_ids and do_inline:
                        if query_term in factual_query(q, *args, **kwargs):
                            return False
                    else:
                        if query_term not in factual_query(q, *args, **kwargs):
                            return False
            if instantiation_order[source] == instantiation_order[target]:
                if do_inline:
                    if source in anti_factual_ids and target in anti_factual_ids:
                        if query_term in factual_query(q, *args, **kwargs):
                            return False
                    elif source not in anti_factual_ids \
                            and target not in anti_factual_ids:
                        if query_term not in factual_query(q, *args, **kwargs):
                            return False
                    else:
                        return False
                else:
                    if query_term not in factual_query(q, *args, **kwargs):
                        return False
        return True
