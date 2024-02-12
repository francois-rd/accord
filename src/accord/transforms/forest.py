from typing import Dict, List, Iterable, Optional, Tuple
from itertools import combinations
from dataclasses import replace
from copy import deepcopy

from ..components import BeamSearch, BeamSearchProtocol, Reducer, TermFormatter
from ..base import (
    InstantiationData,
    InstantiationForest,
    RelationalTemplate,
    RelationalTree,
    QAData,
    Template,
    Term,
    VarId,
)


class ForestTransform:
    def __init__(
        self,
        reducer: Optional[Reducer],
        beam_search: BeamSearch,
        protocol: BeamSearchProtocol,
        formatter: TermFormatter,
        language: str,
    ):
        self.reducer = reducer
        self.beam_search = beam_search
        self.protocol = protocol
        self.formatter = formatter
        self.language = language
        self.data_id_counter = 0

    def __call__(
        self,
        trees: Iterable[RelationalTree],
        qa_data: QAData,
    ) -> InstantiationForest:
        """
        Given a sequence of (relationally-transformed) ReasoningTrees and some QAData,
        apply a sequence of Transforms to each tree, returning a ReasoningForest.
        """
        forest = InstantiationForest()
        for tree in trees:
            family = None  # This avoids adding families with no valid instantiations.
            for pairing_data in self._all_pairings(tree, qa_data):
                for anti_factual_ids in self._all_anti_factual_ids(tree, pairing_data):
                    for full_data in self._instantiate_variables(
                        tree, pairing_data, anti_factual_ids, qa_data
                    ):
                        if family is None:
                            # Delay creating a new family until at least one valid hit.
                            family = forest.add_family(tree)
                        family.add(full_data.identifier)
                        forest.add_data(full_data)
        return forest

    def _all_pairings(
        self,
        tree: RelationalTree,
        qa_data: QAData,
    ) -> Iterable[InstantiationData]:
        for template in tree.templates:
            for pairing, qa_template in self._find_pairings(template, qa_data):
                if self.reducer is None:
                    answers = tree.unique_variable_ids() - {pairing[0]}
                else:
                    answers = self.reducer.valid_answer_ids(tree, template, pairing[0])
                for answer in answers:
                    yield InstantiationData(
                        pairing_template=deepcopy(template),
                        pairing=pairing,
                        qa_template=qa_template,
                        answer_id=answer,
                    )

    @staticmethod
    def _find_pairings(
        template: RelationalTemplate,
        qa_data: QAData,
    ) -> Iterable[Tuple[Tuple[VarId, Term], Template]]:
        for qa_template in qa_data.pairing_templates:
            # Fitting a pairing fails if the RelationTypes don't match.
            if qa_template.relation.type_ != template.relation_type:
                continue

            # Whichever variable in the pairing template is instantiated, the equivalent
            # variable in the test template becomes the pairing variable in the tree.
            if qa_template.source.term is not None:
                yield (template.source_id, qa_template.source.term), qa_template
            if qa_template.target.term is not None:
                yield (template.target_id, qa_template.target.term), qa_template

    @staticmethod
    def _all_anti_factual_ids(
        tree: RelationalTree,
        data: InstantiationData,
    ) -> Iterable[List[VarId]]:
        """
        Exhaustively chooses all possible combinations of anti-factual Variables. For
        each such combination, returns a new ReasoningTree marking this combination.

        Specifically, for each Variable that isn't the answer choice or pairing term,
        mark all combinations of other Variables (in any quantity from none to all of
        them) as anti-factual Variables. In other words, there *can* be no anti-factual
        Variables at all, and there *must* be at least two untouched Variables (to treat
        as the answer choice and pairing Variables).
        """
        af_variables = tree.unique_variable_ids() - {data.pairing[0], data.answer_id}
        for k in range(len(af_variables) + 1):
            for combination in combinations(af_variables, k):
                yield list(combination)

    def _instantiate_variables(
        self,
        tree: RelationalTree,
        data: InstantiationData,
        anti_factual_ids: List[VarId],
        qa_data: QAData,
    ) -> Iterable[InstantiationData]:
        if self.protocol == BeamSearchProtocol.AF_IN_LINE:
            for label_term in qa_data.answer_choices.values():
                seed_map = {
                    data.answer_id: self._format(label_term),
                    data.pairing[0]: self._format(data.pairing[1]),
                }
                yield from self._do_instantiate(tree, data, anti_factual_ids, seed_map)
        elif self.protocol == BeamSearchProtocol.AF_POST_HOC:
            label_term = qa_data.answer_choices[qa_data.correct_answer_label]
            seed_map = {
                data.answer_id: self._format(label_term),
                data.pairing[0]: self._format(data.pairing[1]),
            }
            yield from self._do_instantiate(tree, data, anti_factual_ids, seed_map)
        else:
            raise ValueError(
                f"Unsupported value for beam search protocol: {self.protocol}"
            )

    def _format(self, term: Term) -> Term:
        return self.formatter.format(term, self.language)

    def _do_instantiate(
        self,
        tree: RelationalTree,
        data: InstantiationData,
        anti_factual_ids: List[VarId],
        seed_mapping: Dict[VarId, Term],
    ) -> Iterable[InstantiationData]:
        for mapping in self.beam_search(tree, anti_factual_ids, seed_mapping):
            new_data = replace(
                deepcopy(data),
                identifier=f"I{self.data_id_counter}",
                anti_factual_ids=anti_factual_ids,
                mapping=deepcopy(mapping),
            )
            self.data_id_counter += 1
            yield new_data
