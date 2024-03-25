from typing import Callable, List, Optional
import json

from tqdm import tqdm

from ..configs import BeamSearchConfig, GeneralConfig, ResourcesConfig, update
from ...base import QAData
from ...components import TermFormatter
from ...io import load_forest_jsonl, load_relations_csv, save_dataclass_jsonl
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
