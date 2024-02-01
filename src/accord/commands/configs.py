from dataclasses import dataclass

from ..components import BeamSearchProtocol


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

    # InstantiationForest directory.
    forest_dir: str = "${instantiation_dir}/forest_${tree_size}"

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
