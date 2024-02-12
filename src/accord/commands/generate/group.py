from typing import Callable, List, Optional
import os

from tqdm import tqdm

from ..configs import BeamSearchConfig, GeneralConfig, ResourcesConfig
from ...base import QAData
from ...components import TermFormatter
from ...io import ForestIO, load_relations_csv, save_dataclass_jsonl
from ...transforms import BasicQAGroupTransform, MappingDistanceFunc


def placeholder(_: ResourcesConfig, __: GeneralConfig, ___: BeamSearchConfig):
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
        ):
            self.resources = resources
            self.general = general
            self.beam_search_cfg = beam_search_cfg

        def run(self):
            transform = BasicQAGroupTransform(
                protocol=self.beam_search_cfg.protocol,
                formatter=formatter_loader(),
                language=language,
                relations=load_relations_csv(self.resources.relations_file),
                mapping_distance_fn=mapping_distance_fn,
            )

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
                groups = [g for f in forest.families for g in transform(qa_data, f)]
                save_dataclass_jsonl(file_path, *groups)

    return Generator
