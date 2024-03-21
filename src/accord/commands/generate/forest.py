from typing import Callable, List

from tqdm import tqdm

from ...base import RelationalTree, QAData
from ...transforms import ForestTransform
from ...components import (
    BeamSearch,
    GeneratorFilter,
    Instantiator,
    QueryResultSorter,
    TermFormatter,
)
from ...io import (
    load_dataclass_jsonl,
    load_reducer_csv,
    load_relations_csv,
    save_forest_jsonl,
)
from ..configs import (
    BeamSearchConfig,
    FilterConfig,
    GeneralConfig,
    ReducerConfig,
    ResourcesConfig,
    update,
)


def placeholder(
    _: ResourcesConfig,
    __: GeneralConfig,
    ___: BeamSearchConfig,
    ____: ReducerConfig,
    _____: FilterConfig,
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
            filter_cfg: FilterConfig,
        ):
            self.resources = resources
            self.general = general
            self.beam_search_cfg = beam_search_cfg
            self.reducer_cfg = reducer_cfg
            self.filter_cfg = filter_cfg

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
            seed = self.general.random_seed
            transform = ForestTransform(
                reducer=reducer,
                beam_search=beam_search,
                protocol=self.beam_search_cfg.protocol,
                formatter=formatter_loader(),
                language=language,
                pairing_filter=GeneratorFilter(self.filter_cfg.pairing_prob, seed),
                anti_factual_filter=GeneratorFilter(self.filter_cfg.anti_factual_prob),
                verbose=self.general.verbose
            )

            # For each QAData, transform all trees, then save.
            disable = not self.general.verbose
            for qa_data in tqdm(qa_dataset, desc="Progress", disable=disable):
                with update(self.resources, qa_data) as resources:
                    save_forest_jsonl(
                        family_file_path=resources.forest_families_file,
                        data_file_path=resources.forest_data_file,
                        forest=transform(trees, qa_data),
                    )

    return Generator
