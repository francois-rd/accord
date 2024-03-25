from typing import Callable, List, Optional
import json

from tqdm import tqdm

from ...base import QAData
from ...components import TermFormatter
from ...transforms import BasicQAGroupTransform, MappingDistanceFunc
from ...io import (
    load_forest_jsonl,
    load_reducer_csv,
    load_relations_csv,
    save_dataclass_jsonl,
)
from ..configs import (
    BeamSearchConfig,
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
):
    pass


def factory(
    qa_dataset_loader: Callable[[], List[QAData]],
    formatter_loader: Callable[[], TermFormatter],
    language: str,
    mapping_distance_fn: Optional[MappingDistanceFunc],
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
            # Load the reducer.
            if self.reducer_cfg.ignore:
                reducer = None
            else:
                reducer = load_reducer_csv(
                    file_path=self.resources.reductions_file,
                    relations=load_relations_csv(self.resources.relations_file),
                    raise_=self.reducer_cfg.raise_on_dup,
                )

            transform = BasicQAGroupTransform(
                protocol=self.beam_search_cfg.protocol,
                formatter=formatter_loader(),
                language=language,
                relations=load_relations_csv(self.resources.relations_file),
                mapping_distance_fn=mapping_distance_fn,
                reducer=reducer,
                verbose=self.general.verbose,
            )

            disable = not self.general.verbose
            for qa_data in tqdm(qa_dataset_loader(), desc="Progress", disable=disable):
                with update(self.resources, qa_data) as resources:
                    forest = load_forest_jsonl(
                        family_file_path=resources.forest_families_file,
                        data_file_path=resources.forest_data_file,
                    )
                    groups = [g for f in forest.families for g in transform(qa_data, f)]
                    save_dataclass_jsonl(resources.group_file, *groups)
            if self.general.verbose:
                print(f"Summary stats:\n{json.dumps(transform.get_stats(), indent=4)}")

    return Generator
