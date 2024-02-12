from typing import Callable, List

from tqdm import tqdm

from ..configs import BeamSearchConfig, GeneralConfig, ReducerConfig, ResourcesConfig
from ...base import RelationalTree, QAData
from ...components import BeamSearch, Instantiator, QueryResultSorter, TermFormatter
from ...transforms import ForestTransform
from ...io import ForestIO, load_dataclass_jsonl, load_reducer_csv, load_relations_csv


def placeholder(
    _: ResourcesConfig,
    __: GeneralConfig,
    ___: BeamSearchConfig,
    ____: ReducerConfig,
):
    pass


def factory(
    qa_dataset_loader: Callable[[], List[QAData]],
    factual_instantiator_loader: Callable[[], Instantiator],
    anti_factual_instantiator_loader: Callable[[], Instantiator],
    formatter_loader: Callable[[], TermFormatter],
    sorter_loader: Callable[[], QueryResultSorter],
    language: str,
) -> Callable:
    class Generator:
        def __init__(
            self,
            resources: ResourcesConfig,
            general: GeneralConfig,
            beam_search_cfg: BeamSearchConfig,
            reducer_cfg: ReducerConfig,
        ):
            self.resources = resources
            self.general = general
            self.beam_search_cfg = beam_search_cfg
            self.reducer_cfg = reducer_cfg

        def run(self):
            # Load the relationally-transformed trees as well as all the dataset.
            trees = load_dataclass_jsonl(
                self.resources.relational_trees_file,
                RelationalTree,
            )
            qa_dataset = qa_dataset_loader()

            # Load the reducer.
            if self.reducer_cfg.ignore:
                reducer = None
            else:
                reducer = load_reducer_csv(
                    file_path=self.resources.reductions_file,
                    relations=load_relations_csv(self.resources.relations_file),
                    raise_=self.reducer_cfg.raise_on_dup,
                )

            beam_search = BeamSearch(
                factual_instantiator=factual_instantiator_loader(),
                anti_factual_instantiator=anti_factual_instantiator_loader(),
                protocol=self.beam_search_cfg.protocol,
                sorter=sorter_loader(),
                top_k=self.beam_search_cfg.top_k,
            )

            # Load the forest transform.
            transform = ForestTransform(
                reducer=reducer,
                beam_search=beam_search,
                protocol=self.beam_search_cfg.protocol,
                formatter=formatter_loader(),
                language=language,
            )

            # For each QAData, transform all trees, then save.
            forest_io = ForestIO(
                dir_path=self.resources.forest_and_group_dir,
                family_file_name=self.resources.forest_families_file,
                data_file_name=self.resources.forest_data_file,
            )
            disable = not self.general.verbose
            for qa_data in tqdm(qa_dataset, desc="Progress", disable=disable):
                forest_io.save_jsonl(qa_data, transform(trees, qa_data))

    return Generator
