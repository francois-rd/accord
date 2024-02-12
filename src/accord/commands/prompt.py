from typing import Callable, Dict, Iterable, List, Optional
import os

from tqdm import tqdm

from .configs import GeneralConfig, QAPromptSurfacerConfig, ResourcesConfig
from ..base import (
    InstantiationForest,
    QAData,
    QAGroup,
    QAPrompt,
    Relation,
    RelationType,
)
from ..components import (
    DuplicateTemplateFunc,
    LLM,
    LLMResult,
    QADataSurfacer,
    QAPromptSurfacer,
    TemplateSequencer,
    TemplateSequenceSurfacer,
    TemplateSurfacer,
    TermUnFormatter,
    TextSurfacer,
)
from ..io import (
    ForestIO,
    load_dataclass_jsonl,
    load_relations_csv,
    save_dataclass_jsonl,
)


def placeholder(_: ResourcesConfig, __: GeneralConfig, ___: QAPromptSurfacerConfig):
    pass


def factory(
    qa_dataset_loader: Callable[[], List[QAData]],
    un_formatter_loader: Callable[[], Optional[TermUnFormatter]],
    llm_loader: Callable[[], LLM],
    duplicate_template_fn: DuplicateTemplateFunc,
) -> Callable:
    class Generator:
        def __init__(
            self,
            resources: ResourcesConfig,
            general: GeneralConfig,
            prompt_cfg: QAPromptSurfacerConfig,
        ):
            self.resources = resources
            self.general = general
            self.prompt_cfg = prompt_cfg
            self.surfacer = self._init_surfacer()
            self.llm = llm_loader()

        def _init_surfacer(self) -> QAPromptSurfacer:
            # Aliases to shorten line lengths.
            qa_surfacer = self.prompt_cfg.qa_data_surfacer
            seq_surfacer = self.prompt_cfg.template_sequence_surfacer
            seq = seq_surfacer.sequencer
            within = seq.shuffle_within_tree_template_order
            between = seq.shuffle_between_tree_template_order

            # Create Surfacer.
            surfacer = QAPromptSurfacer(
                prefix=self.prompt_cfg.prefix,
                surfacer_separator=self.prompt_cfg.surfacer_separator,
                prefix_surfacer=TextSurfacer(
                    prefix=self.prompt_cfg.prefix_surfacer.prefix,
                    text=self.prompt_cfg.prefix_surfacer.text,
                ),
                template_sequence_surfacer=TemplateSequenceSurfacer(
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
                    surfacer=TemplateSurfacer(
                        prefix=seq_surfacer.surfacer.prefix,
                        un_formatter=un_formatter_loader(),
                    ),
                ),
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
            return surfacer

        def _do_run(
            self,
            qa_data: QAData,
            forest: InstantiationForest,
            groups: List[QAGroup],
            relation_map: Dict[RelationType, Relation],
        ) -> Iterable[LLMResult]:
            # For each QAGroup, create an associated QAPrompt.
            for group in groups:
                tree_map = {}
                group.instantiate(forest)
                for label, data in group.data_map.items():
                    for f in forest.families:
                        if data.identifier in f.data_ids:
                            tree_map[label] = data.instantiate(f.tree, relation_map)
                            break
                prompt = QAPrompt(qa_data, group, tree_map)

                # For each possible answer choice, query the LLM with the QAPrompt.
                for chosen_answer_label in qa_data.answer_choices:
                    result = self.llm(self.surfacer(prompt, chosen_answer_label))
                    result.chosen_answer_label = chosen_answer_label
                    result.qa_group_id = group.identifier
                    yield result

        def run(self):
            relations = load_relations_csv(self.resources.relations_file)
            relation_map = {r.type_: r for r in relations}
            forest_io = ForestIO(
                dir_path=self.resources.forest_and_group_dir,
                family_file_name=self.resources.forest_families_file,
                data_file_name=self.resources.forest_data_file,
            )

            disable = not self.general.verbose
            for qa_data in tqdm(qa_dataset_loader(), desc="Progress", disable=disable):
                forest = forest_io.load_jsonl(qa_data)
                file_path = os.path.join(
                    self.resources.forest_and_group_dir,
                    qa_data.identifier,
                    self.resources.group_file,
                )
                groups = load_dataclass_jsonl(file_path, t=QAGroup)

                file_path = os.path.join(
                    self.resources.forest_and_group_dir,
                    qa_data.identifier,
                    self.resources.llm_results_file,
                )
                results = self._do_run(qa_data, forest, groups, relation_map)
                save_dataclass_jsonl(file_path, *results)

    return Generator
