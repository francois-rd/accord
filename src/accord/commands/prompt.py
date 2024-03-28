from typing import Callable, Dict, Iterable, List, Optional, Tuple
from copy import deepcopy
import random

from tqdm import tqdm

from .configs import (
    FilterConfig,
    GeneralConfig,
    QAPromptSurfacerConfig,
    ResourcesConfig,
    update,
)
from ..base import (
    InstantiationData,
    InstantiationForest,
    Label,
    QAData,
    QAGroup,
    QAPrompt,
    Relation,
    RelationType,
    Tree,
)
from ..components import (
    DuplicateTemplateFunc,
    GeneratorFilter,
    LLM,
    LLMResult,
    QADataSurfacer,
    QAPromptSurfacer,
    TemplateSequencer,
    TemplateSequenceSurfacer,
    TemplateSurfacer,
    TermUnFormatter,
    TermSurfacer,
    TextSurfacer,
)
from ..io import (
    load_dataclass_jsonl,
    load_forest_jsonl,
    load_relations_csv,
    save_dataclass_jsonl,
)


def placeholder(
    _: ResourcesConfig,
    __: GeneralConfig,
    ___: FilterConfig,
    ____: QAPromptSurfacerConfig,
):
    pass


def factory(
    qa_dataset_loader: Callable[[], List[QAData]],
    un_formatter_loader: Callable[[], Optional[TermUnFormatter]],
    llm_loader: Callable[[], LLM],
    duplicate_template_fn: DuplicateTemplateFunc,
) -> Callable:
    class Prompt:
        def __init__(
            self,
            resources: ResourcesConfig,
            general: GeneralConfig,
            filter_cfg: FilterConfig,
            prompt_cfg: QAPromptSurfacerConfig,
        ):
            self.resources = resources
            self.general = general
            self.filter_cfg = filter_cfg
            self.prompt_cfg = prompt_cfg
            self.surfacer = self._init_surfacer()
            self.llm = llm_loader()
            self.group_id_counter = 0

        def _init_surfacer(self) -> QAPromptSurfacer:
            # Aliases to shorten line lengths.
            qa_surfacer = self.prompt_cfg.qa_data_surfacer

            # Create TemplateSequenceSurfacer only if trees are given.
            template_sequence_surfacer = None
            if self.resources.tree_size > 0:
                template_sequence_surfacer = self._init_template_sequence_surfacer()

            # Create Surfacer.
            return QAPromptSurfacer(
                prefix=self.prompt_cfg.prefix,
                surfacer_separator=self.prompt_cfg.surfacer_separator,
                prefix_surfacer=TextSurfacer(
                    prefix=self.prompt_cfg.prefix_surfacer.prefix,
                    text=self.prompt_cfg.prefix_surfacer.text,
                ),
                template_sequence_surfacer=template_sequence_surfacer,
                qa_data_surfacer=QADataSurfacer(
                    prefix=qa_surfacer.prefix,
                    question_answer_separator=qa_surfacer.question_answer_separator,
                    answer_choice_separator=qa_surfacer.answer_choice_separator,
                    answer_choice_formatter=qa_surfacer.answer_choice_formatter,
                ),
                suffix_surfacer=TextSurfacer(
                    prefix=self.prompt_cfg.suffix_surfacer.prefix,
                    text=self.prompt_cfg.suffix_surfacer.text,
                ),
            )

        def _init_template_sequence_surfacer(self):
            # Aliases to shorten line lengths.
            seq_surfacer = self.prompt_cfg.template_sequence_surfacer
            seq = seq_surfacer.sequencer
            within = seq.shuffle_within_tree_template_order
            between = seq.shuffle_between_tree_template_order

            # Create Surfacer.
            return TemplateSequenceSurfacer(
                prefix=seq_surfacer.prefix,
                template_separator=seq_surfacer.template_separator,
                sequencer=TemplateSequencer(
                    chosen_answer_position=seq.chosen_answer_position,
                    shuffle_tree_order=seq.shuffle_tree_order,
                    shuffle_within_tree_template_order=within,
                    shuffle_between_tree_template_order=between,
                    remove_duplicate_templates=seq.remove_duplicate_templates,
                    duplicate_template_fn=duplicate_template_fn,
                ),
                template_surfacer=TemplateSurfacer(
                    prefix=seq_surfacer.template_surfacer.prefix,
                    term_surfacer=TermSurfacer(
                        prefix=seq_surfacer.template_surfacer.term_surfacer.prefix,
                        suffix=seq_surfacer.template_surfacer.term_surfacer.suffix,
                        un_formatter=un_formatter_loader(),
                    ),
                ),
            )

        def _run_tree_size_0(self):
            disable = not self.general.verbose
            for qa_data in tqdm(qa_dataset_loader(), desc="Progress", disable=disable):
                with update(self.resources, qa_data) as resources:
                    label = qa_data.correct_answer_label
                    result = self._prompt(label, QAPrompt(qa_data, None), qa_data)
                    save_dataclass_jsonl(resources.llm_results_file, *[result])

        def _run_tree_size_1(self):
            disable = not self.general.verbose
            for qa_data in tqdm(qa_dataset_loader(), desc="Progress", disable=disable):
                with update(self.resources, qa_data) as resources:
                    groups, results = self._do_run_tree_size_1(qa_data)
                    save_dataclass_jsonl(resources.group_file, *groups)
                    save_dataclass_jsonl(resources.llm_results_file, *results)

        def _do_run_tree_size_1(
            self,
            qa_data: QAData,
        ) -> Tuple[List[QAGroup], List[LLMResult]]:
            groups, results = [], []
            for template in qa_data.pairing_templates:
                # Build a QAGroup and QAPrompt from just the pairing templates.
                data_map, tree_map = {}, {}
                for label, term in qa_data.answer_choices.items():
                    new_template = deepcopy(template)
                    if new_template.source.term is None:
                        new_template.source.term = term
                    else:
                        new_template.target.term = term
                    tree_map[label] = Tree([new_template], new_template)
                    data_map[label] = InstantiationData(
                        qa_template=new_template,
                    )
                prompt = QAPrompt(qa_data, tree_map)
                group = QAGroup(f"G{self.group_id_counter}", {}, {}, data_map)
                groups.append(group)
                self.group_id_counter += 1

                # For each kept answer choice, query the LLM with the QAPrompt.
                for label in self._choose_labels(qa_data):
                    results.append(self._prompt(label, prompt, qa_data, group))
            return groups, results

        def _run_other_tree_size(self):
            relations = load_relations_csv(self.resources.relations_file)
            relation_map = {r.type_: r for r in relations}
            disable = not self.general.verbose
            for qa_data in tqdm(qa_dataset_loader(), desc="Progress", disable=disable):
                with update(self.resources, qa_data) as resources:
                    forest = load_forest_jsonl(
                        family_file_path=resources.forest_families_file,
                        data_file_path=resources.forest_data_file,
                    )
                    groups = load_dataclass_jsonl(resources.group_file, t=QAGroup)
                    results = self._do_run_other(qa_data, forest, groups, relation_map)
                    save_dataclass_jsonl(resources.llm_results_file, *results)

        def _do_run_other(
            self,
            qa_data: QAData,
            forest: InstantiationForest,
            groups: List[QAGroup],
            relation_map: Dict[RelationType, Relation],
        ) -> Iterable[LLMResult]:
            # For each QAGroup, create an associated QAPrompt.
            for group in groups:
                fam, af_vars, hops, tree_map = None, None, None, {}
                group.instantiate(forest)
                for label, data in group.data_map.items():
                    if fam is None:
                        for family in forest.families:
                            if data.identifier in family.data_ids:
                                fam = family
                                af_vars = str(len(data.anti_factual_ids))
                                hops = str(data.reasoning_hops)
                                break
                    tree_map[label] = data.instantiate(fam.tree, relation_map)
                prob = self.filter_cfg.prompt_probs.get(af_vars, {}).get(hops, 0.0)
                if not GeneratorFilter(prob).passes():
                    continue
                prompt = QAPrompt(qa_data, tree_map)

                # For each kept answer choice, query the LLM with the QAPrompt.
                for chosen_answer_label in self._choose_labels(qa_data):
                    yield self._prompt(chosen_answer_label, prompt, qa_data, group)

        def _choose_labels(self, qa_data: QAData) -> Iterable[Label]:
            keep, others = [], []
            for label in qa_data.answer_choices:
                if label == qa_data.correct_answer_label:
                    keep.append(label)
                else:
                    others.append(label)
            keep.extend(random.sample(others, self.filter_cfg.num_anti_factual_answers))
            for label in keep:
                yield label

        def _prompt(
            self,
            chosen_answer_label: Label,
            prompt: QAPrompt,
            qa_data: QAData,
            group: Optional[QAGroup] = None,
        ) -> LLMResult:
            text = self.surfacer(prompt, chosen_answer_label)
            result = self.llm(text, qa_data)
            # TODO: Too much room on disk to store as text. Revisit later.
            # result.prompt_text = text
            result.chosen_answer_label = chosen_answer_label
            if group is not None:
                result.qa_group_id = group.identifier
            return result

        def run(self):
            if self.resources.tree_size == 0:
                self._run_tree_size_0()
            elif self.resources.tree_size == 1:
                self._run_tree_size_1()
            else:
                self._run_other_tree_size()

    return Prompt
