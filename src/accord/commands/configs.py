from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import List, Optional

from ..base import QAData
from ..components import BeamSearchProtocol


@dataclass
class GeneralConfig:
    verbose: bool = False
    random_seed: int = 314159


@dataclass
class ResourcesConfig:
    # Top level directory for resources and data.
    root_dir: str = "data"

    # Input dataset/database directories.
    qa_dataset_dir: str = "${root_dir}/${qa_dataset}"
    term_database_dir: str = "${root_dir}/${term_database}"
    qa_dataset: str = ""
    term_database: str = ""

    # GenericTree directory and data.
    generic_trees_dir: str = "${root_dir}/generic_trees"
    generic_trees_file: str = "${generic_trees_dir}/${tree_size}.jsonl"

    # Directories for instantiating generic trees with particular QA/term data
    # and transforming the results.
    all_results_dir: str = "${root_dir}/results"
    result: str = "${qa_dataset}_and_${term_database}"
    result_dir: str = "${all_results_dir}/${result}"

    # Relation and reduction data needed for a Reducer.
    relations_file: str = "${result_dir}/relations.csv"
    reductions_file: str = "${result_dir}/reductions.csv"

    # RelationalTree directory and data.
    relational_trees_dir: str = "${result_dir}/relational_trees"
    relational_trees_file: str = "${relational_trees_dir}/${tree_size}.jsonl"

    # QAData directory and data.
    qa_temp_dir: Optional[str] = None
    qa_data_dir: str = "${result_dir}/qa_data/${tree_size}/${qa_temp_dir}"

    # InstantiationForest directory and data.
    forest_dir: str = "${qa_data_dir}/forests"
    forest_families_file: str = "${forest_dir}/families.jsonl"
    forest_data_file: str = "${forest_dir}/data.jsonl"

    # InstantiationForest directory and data.
    group_dir: str = "${qa_data_dir}/groups"
    group_file: str = "${group_dir}/groups.jsonl"

    # InstantiationForest directory and data.
    llm_results_dir: str = "${qa_data_dir}/llm_results/${llm}"
    llm_results_file: str = "${llm_results_dir}/llm_results.jsonl"

    # Size of tree (number of ReasoningTemplates).
    tree_size: int = 2

    # Language model to use.
    llm: str = "dummy"


@contextmanager
def update(resources: ResourcesConfig, qa_data: QAData):
    resources.qa_temp_dir = qa_data.identifier
    try:
        yield resources
    finally:
        resources.qa_temp_dir = None


@dataclass
class FilterConfig:
    generic_prob: float = 0.0
    relational_prob: float = 0.0
    pairing_prob: float = 0.0
    anti_factual_prob: float = 0.0


@dataclass
class ReducerConfig:
    ignore: bool = False
    raise_on_dup: bool = True


@dataclass
class BeamSearchConfig:
    protocol: BeamSearchProtocol = BeamSearchProtocol.AF_POST_HOC
    top_k: int = 0


@dataclass
class SorterConfig:
    sorter: str = "semantic_distance"
    semantic_distance_target: float = 0.0
    semantic_distance_calculator: str = ""
    semantic_distance_aggregator: str = "sum"


@dataclass
class MappingDistanceConfig:
    ignore: bool = True
    target_distances: List[int] = field(default_factory=lambda: [-1])
    count_answer_ids: bool = False
    count_pairing_ids: bool = False


@dataclass
class TemplateSequencerConfig:
    chosen_answer_position: int = -1
    shuffle_tree_order: bool = False
    shuffle_within_tree_template_order: bool = False
    shuffle_between_tree_template_order: bool = False
    remove_duplicate_templates: bool = False


@dataclass
class SurfacerConfig:
    prefix: str = ""


@dataclass
class TextSurfacerConfig(SurfacerConfig):
    text: str = ""


@dataclass
class TermSurfacerConfig(SurfacerConfig):
    suffix: str = ""


@dataclass
class TemplateSurfacerConfig(SurfacerConfig):
    term_surfacer: TermSurfacerConfig = field(
        default_factory=lambda: TermSurfacerConfig("[", "]")
    )


@dataclass
class TemplateSequenceSurfacerConfig(SurfacerConfig):
    template_separator: str = "\n"
    template_surfacer: TemplateSurfacerConfig = field(
        default_factory=lambda: TemplateSurfacerConfig("- ")
    )
    sequencer: TemplateSequencerConfig = field(
        default_factory=lambda: TemplateSequencerConfig()
    )


@dataclass
class QADataSurfacerConfig(SurfacerConfig):
    question_answer_separator: str = "\n"
    answer_choice_separator: str = "    "
    answer_choice_formatter: str = "{}: {}"


@dataclass
class QAPromptSurfacerConfig(SurfacerConfig):
    surfacer_separator: str = "\n"
    prefix_surfacer: TextSurfacerConfig = field(
        default_factory=lambda: TextSurfacerConfig("Instructions:\n")
    )
    template_sequence_surfacer: TemplateSequenceSurfacerConfig = field(
        default_factory=lambda: TemplateSequenceSurfacerConfig("Statements:\n")
    )
    qa_data_surfacer: QADataSurfacerConfig = field(
        default_factory=lambda: QADataSurfacerConfig("Question:\n")
    )
    suffix_surfacer: TextSurfacerConfig = field(
        default_factory=lambda: TextSurfacerConfig("Answer:\n")
    )
