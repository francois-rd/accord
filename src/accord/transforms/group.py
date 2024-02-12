from typing import Callable, Dict, Iterable, List, Optional, Tuple
from copy import deepcopy
import random

from ..components import BeamSearchProtocol, TermFormatter
from ..base import (
    InstantiationData,
    InstantiationFamily,
    Label,
    QAData,
    QAGroup,
    Relation,
    Term,
)


MappingDistanceFunc = Callable[[InstantiationData, InstantiationData], bool]


def mapping_distance_factory(
    target_distances: List[int],
    count_answer_ids: bool = False,
    count_pairing_ids: bool = False,
) -> MappingDistanceFunc:
    if len(target_distances) == 1 and target_distances[0] < 0:
        targets = None
    elif any(x < 0 for x in target_distances) and any(x >= 0 for x in target_distances):
        raise ValueError(
            "Target distances cannot contain both negative and non-negative values."
        )
    else:
        targets = target_distances

    def fn(d1: InstantiationData, d2: InstantiationData) -> bool:
        return True if targets is None else d1.mapping_distance(
            d2,
            count_answer_ids=count_answer_ids,
            count_pairing_ids=count_pairing_ids,
        ) in targets
    return fn


class BasicQAGroupTransform:
    def __init__(
        self,
        protocol: BeamSearchProtocol,
        formatter: TermFormatter,
        language: str,
        relations: Iterable[Relation],
        mapping_distance_fn: Optional[MappingDistanceFunc],

    ):
        self.protocol = protocol
        self.formatter = formatter
        self.language = language
        self.relation_map = {r.type_: r for r in relations}
        self.mapping_distance_fn = mapping_distance_fn
        self.group_id_counter = 0

    def __call__(
        self,
        qa_data: QAData,
        family: InstantiationFamily,
        *args,
        **kwargs,
    ) -> Iterable[QAGroup]:
        if self.protocol == BeamSearchProtocol.AF_IN_LINE:
            for group in self._group_by_all_but_mapping(family).values():
                if self.mapping_distance_fn is None:
                    yield from self._do_simple_in_line(qa_data, group)
                else:
                    yield from self._do_distance_in_line(qa_data, group)
        elif self.protocol == BeamSearchProtocol.AF_POST_HOC:
            # Each group here is a list of all combinations of full instantiations
            # for a particular all_but_mapping partial InstantiationData.
            for group in self._group_by_all_but_mapping(family).values():
                if self.mapping_distance_fn is None:
                    yield from self._do_simple_post_hoc(qa_data, group)
                else:
                    yield from self._do_distance_post_hoc(qa_data, group)
        else:
            raise ValueError(
                f"Unsupported value for beam search protocol: {self.protocol}"
            )

    @staticmethod
    def _group_by_all_but_mapping(
        family: InstantiationFamily,
    ) -> Dict[Tuple, List[InstantiationData]]:
        groups = {}
        for data in family.data_map.values():
            t = data.pairing_template
            t = t.source_id, t.relation_type, t.target_id
            qa_t = data.qa_template
            qa_t = qa_t.source.term, qa_t.relation.surface_form, qa_t.target.term
            anti_factual_ids = tuple(sorted(data.anti_factual_ids))
            data_key = (t, qa_t, data.pairing, data.answer_id, anti_factual_ids)
            groups.setdefault(data_key, []).append(data)
        return groups

    def _group_by_label(
        self,
        qa_data: QAData,
        group: List[InstantiationData],
    ) -> Optional[Dict[Label, List[InstantiationData]]]:
        # Make subgroups based on answer choices in the data mapping.
        label_groups = {}
        for data in group:
            for label, term in qa_data.answer_choices.items():
                if data.mapping[data.answer_id] == self._format(term):
                    label_groups.setdefault(label, []).append(data)

        # If at least one label has no hits, return None.
        if set(label_groups.keys()) == set(qa_data.answer_choices.keys()):
            return label_groups
        return None

    def _do_distance_in_line(
        self,
        qa_data: QAData,
        group: List[InstantiationData],
    ) -> Iterable[QAGroup]:
        # For each instantiation data representing a correct answer, find relevant
        # other data to pair it with and return all of these.
        label_groups = self._group_by_label(qa_data, group)
        if label_groups is None:
            return
        for correct in label_groups[qa_data.correct_answer_label]:
            relevant_others = {}
            for label, others in label_groups.items():
                if label == qa_data.correct_answer_label:
                    continue
                for other in others:
                    if self.mapping_distance_fn(correct, other):
                        relevant_others.setdefault(label, []).append(other)
            yield from self._do_one_in_line(qa_data, correct, relevant_others)

    def _do_one_in_line(
        self,
        qa_data: QAData,
        correct: InstantiationData,
        relevant_others: Dict[Label, List[InstantiationData]]
    ) -> Iterable[QAGroup]:
        # Shuffle to remove any bias from the otherwise systematic combinatorics.
        for others in relevant_others.values():
            random.shuffle(others)

        # Grab the ith item from each subgroup until one subgroup is exhausted.
        i, done = 0, False
        while not done:
            data_ids = {qa_data.correct_answer_label: correct.identifier}
            data_map = {k: None for k in qa_data.answer_choices}
            for label, others in relevant_others.items():
                if i >= len(others):
                    done = True
                    break
                data_ids[label] = others[i].identifier
            if not done:
                group = QAGroup(f"G{self.group_id_counter}", data_ids, data_map)
                self.group_id_counter += 1
                yield group
            i += 1

    def _do_simple_in_line(
        self,
        qa_data: QAData,
        group: List[InstantiationData],
    ) -> Iterable[QAGroup]:
        label_groups = self._group_by_label(qa_data, group)
        if label_groups is None:
            return

        # Shuffle to remove any bias from the otherwise systematic combinatorics.
        for data_list in label_groups.values():
            random.shuffle(data_list)

        # Grab the ith item from each subgroup until one subgroup is exhausted.
        i, done = 0, False
        while not done:
            data_ids, data_map = {}, {k: None for k in qa_data.answer_choices}
            for label, data_list in label_groups.items():
                if i >= len(data_list):
                    done = True
                    break
                data_ids[label] = data_list[i].identifier
            if not done:
                group = QAGroup(f"G{self.group_id_counter}", data_ids, data_map)
                self.group_id_counter += 1
                yield group
            i += 1

    def _do_distance_post_hoc(
        self,
        qa_data: QAData,
        group: List[InstantiationData],
    ) -> Iterable[QAGroup]:
        # Shuffle to remove any bias from the otherwise systematic combinatorics.
        random.shuffle(group)

        for i, correct in enumerate(group):
            relevant_others = []
            for j, other in enumerate(group):
                if i != j and self.mapping_distance_fn(correct, other):
                    relevant_others.append(other)
            yield from self._do_one_post_hoc(qa_data, correct, relevant_others)

    def _do_one_post_hoc(
        self,
        qa_data: QAData,
        correct: InstantiationData,
        relevant_others: List[InstantiationData],
    ) -> Iterable[QAGroup]:
        # Grab all non-correct choice labels.
        other_choices = {
            label: term for label, term in qa_data.answer_choices.items()
            if label != qa_data.correct_answer_label
        }

        # Arbitrarily batch the others. For each full-sized batch, arbitrarily pair the
        # answer choices with one of the batch items.
        batch_size = len(qa_data.answer_choices) - 1
        for i in range(0, len(relevant_others), batch_size):
            batch = relevant_others[i:i + batch_size]
            if len(batch) != batch_size:
                continue

            data_ids = {qa_data.correct_answer_label: correct.identifier}
            data_map = {qa_data.correct_answer_label: None}
            for (label, term), data in zip(other_choices.items(), batch):
                data_ids[label] = data.identifier
                data_map[label] = deepcopy(data.mapping)
                data_map[label][data.answer_id] = self._format(term)
            group = QAGroup(f"G{self.group_id_counter}", data_ids, data_map)
            self.group_id_counter += 1
            yield group

    def _do_simple_post_hoc(
        self,
        qa_data: QAData,
        group: List[InstantiationData],
    ) -> Iterable[QAGroup]:
        # Replace only the answer term mapping. These trees are otherwise identical.
        for data in group:
            data_ids, data_map = {}, {}
            for label, term in qa_data.answer_choices.items():
                data_ids[label] = data.identifier
                data_map[label] = deepcopy(data.mapping)
                data_map[label][data.answer_id] = self._format(term)
            group = QAGroup(f"G{self.group_id_counter}", data_ids, data_map)
            self.group_id_counter += 1
            yield group

    def _format(self, term: Term) -> Term:
        return self.formatter.format(term, self.language)
