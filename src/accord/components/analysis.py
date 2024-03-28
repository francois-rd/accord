from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

from ..base import InstantiationForest, Label, QAData, QAGroup
from .llm import LLMResult


class BinType(Enum):
    TREE_SIZE = "TREE_SIZE"
    REASONING_HOPS = "REASONING_HOPS"
    DISTRACTORS = "DISTRACTORS"


class TableType(Enum):
    ACCURACY = "ACCURACY"
    AF_CONFUSION = "AF_CONFUSION"  # NOTE: This is NOT exactly a confusion matrix.
    FAILED = "FAILED"


@dataclass
class AnalysisData:
    qa_data: QAData
    forest: Optional[InstantiationForest]
    groups: Optional[List[QAGroup]]
    llm_results: List[LLMResult]


@dataclass
class AnswerLabels:
    generated_answer_label: Optional[Label]
    chosen_answer_label: Label
    correct_answer_label: Label


# Key is a bin value for a specific BinType.
AnalysisResults = Dict[int, List[AnswerLabels]]


@dataclass
class Analysis:
    tree_size: int
    llm: str
    bin_type: BinType
    data: Optional[List[AnalysisData]]
    results: AnalysisResults = field(default_factory=dict)


# Key is a series name. Value is {bin value: metric value}.
TableData = Dict[str, Dict[int, float]]


@dataclass
class AnalysisTable:
    llm: str
    bin_type: BinType
    table_type: TableType
    baseline: Optional[float] = None
    data: TableData = field(default_factory=dict)


def find_group(llm_result: LLMResult, groups: List[QAGroup]) -> QAGroup:
    for group in groups:
        if group.identifier == llm_result.qa_group_id:
            return group


def tree_size(analysis: Analysis, _: AnalysisData, __: LLMResult) -> int:
    return analysis.tree_size


def reasoning_hops(
    analysis: Analysis,
    analysis_data: AnalysisData,
    llm_result: LLMResult,
) -> int:
    if analysis.tree_size < 2:
        return analysis.tree_size
    group = find_group(llm_result, analysis_data.groups)
    arbitrary_data_id = next(group.data_ids.values().__iter__())
    return analysis_data.forest.data_map[arbitrary_data_id].reasoning_hops


def distractors(
    analysis: Analysis,
    analysis_data: AnalysisData,
    llm_result: LLMResult,
) -> int:
    return analysis.tree_size - reasoning_hops(analysis, analysis_data, llm_result)


def analyze(analysis: Analysis) -> Analysis:
    if analysis.bin_type == BinType.TREE_SIZE:
        fn = tree_size
    elif analysis.bin_type == BinType.REASONING_HOPS:
        fn = reasoning_hops
    elif analysis.bin_type == BinType.DISTRACTORS:
        fn = distractors
    else:
        raise ValueError(f"Unsupported BinType: {analysis.bin_type}")
    for analysis_data in analysis.data:
        for llm_result in analysis_data.llm_results:
            data_bin = fn(analysis, analysis_data, llm_result)
            analysis.results.setdefault(data_bin, []).append(AnswerLabels(
                generated_answer_label=llm_result.generated_answer_label,
                chosen_answer_label=llm_result.chosen_answer_label,
                correct_answer_label=analysis_data.qa_data.correct_answer_label,
            ))
    return analysis


def collate_results(analyses: List[Analysis]) -> AnalysisResults:
    results = {}
    for analysis in analyses:
        if analysis.tree_size == 0:
            results.setdefault(-1, []).extend(analysis.results[0])
        else:
            for k, v in analysis.results.items():
                results.setdefault(k, []).extend(v)
    return results


def safe_div(score, count, zero_div=-1.0):
    return zero_div if count == 0 else score / count


def accuracy(table: AnalysisTable, results: AnalysisResults) -> AnalysisTable:
    def acc(labels, condition):
        score, count = 0, 0
        for r in labels:
            if condition(r.chosen_answer_label, r.correct_answer_label):
                if r.generated_answer_label == r.chosen_answer_label:
                    score += 1
                count += 1
        return safe_div(score, count)

    def f_acc(labels):
        return acc(labels, lambda x, y: x == y)

    def af_acc(labels):
        return acc(labels, lambda x, y: x != y)

    for data_bin, label_results in results.items():
        if data_bin == -1:
            table.baseline = f_acc(label_results)
        else:
            table.data.setdefault("f_acc", {})[data_bin] = f_acc(label_results)
            table.data.setdefault("af_acc", {})[data_bin] = af_acc(label_results)
    return table


def af_confusion(table: AnalysisTable, results: AnalysisResults) -> AnalysisTable:
    def count(labels):
        f_llm_f_gt, f_llm_af_gt, af_llm_f_gt, af_llm_af_gt, f_gt, af_gt = \
            0, 0, 0, 0, 0, 0
        for r in labels:
            if r.chosen_answer_label == r.correct_answer_label:
                if r.generated_answer_label == r.chosen_answer_label:
                    f_llm_f_gt += 1
                else:
                    af_llm_f_gt += 1
                f_gt += 1
            if r.chosen_answer_label != r.correct_answer_label:
                if r.generated_answer_label == r.chosen_answer_label:
                    f_llm_af_gt += 1
                else:
                    af_llm_af_gt += 1
                af_gt += 1
        return (
            safe_div(f_llm_f_gt, f_gt), safe_div(af_llm_f_gt, f_gt),
            safe_div(f_llm_af_gt, af_gt), safe_div(af_llm_af_gt, af_gt),
        )

    for data_bin, result in results.items():
        if data_bin == -1:
            table.baseline = count(result)[0]
        else:
            counts = count(result)
            table.data.setdefault("f_llm_f_gt", {})[data_bin] = counts[0]
            table.data.setdefault("af_llm_f_gt", {})[data_bin] = counts[1]
            table.data.setdefault("f_llm_af_gt", {})[data_bin] = counts[2]
            table.data.setdefault("af_llm_af_gt", {})[data_bin] = counts[3]
    return table


def failed(table: AnalysisTable, results: AnalysisResults) -> AnalysisTable:
    def fails(labels, condition):
        score, count = 0, 0
        for r in labels:
            if condition(r.chosen_answer_label, r.correct_answer_label):
                if r.generated_answer_label is None:
                    score += 1
                count += 1
        return score / count

    def f_fails(labels):
        return fails(labels, lambda x, y: x == y)

    def af_fails(labels):
        return fails(labels, lambda x, y: x != y)

    for data_bin, label_results in results.items():
        if data_bin == -1:
            table.baseline = f_fails(label_results)
        else:
            table.data.setdefault("f_fails", {})[data_bin] = f_fails(label_results)
            table.data.setdefault("af_fails", {})[data_bin] = af_fails(label_results)
    return table


def to_table(table: AnalysisTable, analyses: List[Analysis]) -> AnalysisTable:
    if table.table_type == TableType.ACCURACY:
        return accuracy(table, collate_results(analyses))
    elif table.table_type == TableType.AF_CONFUSION:
        return af_confusion(table, collate_results(analyses))
    elif table.table_type == TableType.FAILED:
        return failed(table, collate_results(analyses))
    else:
        raise ValueError(f"Unsupported TableType: {table.table_type}")
