from dataclasses import dataclass, field
from typing import List

from ..components import BeamSearchProtocol


@dataclass
class GeneralConfig:
    verbose: bool = False


@dataclass
class ResourcesConfig:
    # Top level directory for resources and data.
    root_dir: str = "data"

    # Next level directories.
    trees_dir: str = "${root_dir}/trees"
    qa_dataset_dir: str = "${root_dir}/${qa_dataset}"
    term_database_dir: str = "${root_dir}/${term_database}"
    qa_dataset: str = ""
    term_database: str = ""

    # GenericTree directory and data.
    generic_trees_dir: str = "${trees_dir}/generic_trees"
    generic_trees_file: str = "${generic_trees_dir}/${tree_size}.jsonl"

    # Directories for instantiating generic trees with particular QA/term data.
    instantiation: str = "${qa_dataset}_and_${term_database}"
    instantiation_dir: str = "${trees_dir}/${instantiation}"

    # Relation and reduction data needed for a Reducer.
    relations_file: str = "${instantiation_dir}/relations.csv"
    reductions_file: str = "${instantiation_dir}/reductions.csv"

    # RelationalTree directory and data.
    relational_trees_dir: str = "${instantiation_dir}/relational_trees"
    relational_trees_file: str = "${relational_trees_dir}/${tree_size}.jsonl"

    # InstantiationForest and QAGroup directory and data.
    forest_and_group_dir: str = "${instantiation_dir}/forest_and_group_${tree_size}"
    forest_families_file: str = "families.jsonl"
    forest_data_file: str = "data.jsonl"
    group_file: str = "groups.jsonl"
    llm_results_file: str = "llm_results.json"

    # Size of tree (number of ReasoningTemplates) to handle.
    tree_size: int = 2


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
    random_seed: int = 314159


@dataclass
class MappingDistanceConfig:
    use_mapping_distance: bool = False
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
class TemplateSequenceSurfacerConfig(SurfacerConfig):
    template_separator: str = "\n"
    surfacer: SurfacerConfig = field(
        default_factory=lambda: SurfacerConfig("- ")
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
