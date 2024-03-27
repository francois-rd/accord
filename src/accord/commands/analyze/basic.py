from dataclasses import dataclass, field
from typing import Callable, Dict, List
import json

from tqdm import tqdm

from ..configs import GeneralConfig, ResourcesConfig, update
from ...base import QAData, QAGroup
from ...io import load_dataclass_jsonl, load_forest_jsonl, save_dataclass_jsonl
from ...components import (
    Analysis,
    AnalysisData,
    AnalysisTable,
    BinType,
    LLMResult,
    TableType,
    analyze,
    to_table,
)


@dataclass
class BasicAnalysisConfig:
    tree_sizes: List[int] = field(default_factory=list)
    llms: List[str] = field(default_factory=list)
    bin_types: List[BinType] = field(default_factory=list)
    table_types: List[TableType] = field(default_factory=list)


def placeholder(_: ResourcesConfig, __: GeneralConfig, ___: BasicAnalysisConfig):
    pass


def factory(
    qa_dataset_loader: Callable[[], List[QAData]],
) -> Callable:
    class Analyze:
        def __init__(
            self,
            resources: ResourcesConfig,
            general: GeneralConfig,
            analysis_cfg: BasicAnalysisConfig,
        ):
            self.resources = resources
            self.general = general
            self.analysis_cfg = analysis_cfg
            self.qa_dataset = None

        def run(self):
            # Analyze data.
            results = {}
            self.qa_dataset = qa_dataset_loader()
            tree_sizes, disable = self.analysis_cfg.tree_sizes, not self.general.verbose
            for tree_size in tqdm(tree_sizes, desc="Analysis", disable=disable):
                forest_and_groups = self._load_forest_and_groups(tree_size)
                for llm in self.analysis_cfg.llms:
                    llm_results = self._load_llm_results(tree_size, llm)
                    data = self._collate_analysis_data(forest_and_groups, llm_results)
                    for bin_type in self.analysis_cfg.bin_types:
                        a = analyze(Analysis(
                            tree_size=tree_size, llm=llm, bin_type=bin_type, data=data,
                        ))
                        results.setdefault(llm, {}).setdefault(bin_type, []).append(a)

            # Convert analysis results to tables.
            tables = []
            table_types = self.analysis_cfg.table_types
            for table_type in tqdm(table_types, desc="Analysis", disable=disable):
                for llm in self.analysis_cfg.llms:
                    for bin_type in self.analysis_cfg.bin_types:
                        table = AnalysisTable(
                            llm=llm, bin_type=bin_type, table_type=table_type,
                        )
                        tables.append(to_table(table, results[llm][bin_type]))
            save_dataclass_jsonl(self.resources.analysis_file, *tables)

            # Crudely print results if desired.
            if self.general.verbose:
                for table in tables:
                    print(f"LLM.{table.llm}    {table.bin_type}    {table.table_type}")
                    print(json.dumps(table.data, indent=4))

        def _load_forest_and_groups(self, tree_size: int) -> Dict[str, dict]:
            all_data = {}
            for qa_data in self.qa_dataset:
                with update(self.resources, qa_data, tree_size=tree_size) as resources:
                    forest = None
                    if tree_size > 1:
                        forest = load_forest_jsonl(
                            family_file_path=resources.forest_families_file,
                            data_file_path=resources.forest_data_file,
                        )
                    groups = None
                    if tree_size > 0:
                        groups = load_dataclass_jsonl(resources.group_file, t=QAGroup)
                all_data[qa_data.identifier] = dict(
                    qa_data=qa_data, forest=forest, groups=groups,
                )
            return all_data

        def _load_llm_results(self, tree_size: int, llm: str) -> Dict[str, list]:
            all_llm_results = {}
            if self.qa_dataset is None:
                self.qa_dataset = qa_dataset_loader()
            for qa_data in self.qa_dataset:
                with update(self.resources, qa_data, tree_size=tree_size, llm=llm) as r:
                    llm_results = load_dataclass_jsonl(r.llm_results_file, t=LLMResult)
                all_llm_results[qa_data.identifier] = llm_results
            return all_llm_results

        @staticmethod
        def _collate_analysis_data(
            all_forest_and_groups: Dict[str, dict],
            all_llm_results: Dict[str, list],
        ) -> List[AnalysisData]:
            analysis_data = []
            for qa_data_id, forest_and_groups in all_forest_and_groups.items():
                analysis_data.append(AnalysisData(
                    qa_data=forest_and_groups["qa_data"],
                    forest=forest_and_groups["forest"],
                    groups=forest_and_groups["groups"],
                    llm_results=all_llm_results[qa_data_id],
                ))
            return analysis_data

    return Analyze
