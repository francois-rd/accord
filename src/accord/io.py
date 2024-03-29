from typing import Any, Callable, Dict, Iterable, List, Type, TypeVar
from dataclasses import asdict
from pathlib import Path
from enum import Enum
import json
import os

from dacite import Config, from_dict
import pandas as pd

from .components import Reducer, Reduction
from .base import (
    InstantiationData,
    InstantiationFamily,
    InstantiationForest,
    Relation,
    RelationalCaseLink,
)


T = TypeVar("T")

DEFAULT_CONFIG = Config(cast=[Enum, tuple])


def enum_dict_factory(data):
    return dict((k, v.value if isinstance(v, Enum) else v) for k, v in data)


def ensure_path(file_path: str):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    return file_path


def save_json(file_path: str, obj, **kwargs):
    with open(ensure_path(file_path), "w", encoding='utf-8') as f:
        json.dump(obj, f, **kwargs)


def save_jsonl(file_path: str, *objs: Any, **kwargs):
    str_objs = []
    for o in objs:
        str_objs.append(json.dumps(o, **kwargs) + os.linesep)
    with open(ensure_path(file_path), "w", encoding='utf-8') as f:
        f.writelines(str_objs)


def save_dataclass_json(
    file_path: str,
    obj: Any,
    dict_factory: Callable = enum_dict_factory,
    **kwargs,
):
    with open(ensure_path(file_path), "w", encoding='utf-8') as f:
        json.dump(asdict(obj, dict_factory=dict_factory), f, **kwargs)


def save_dataclass_jsonl(
    file_path: str,
    *objs: Any,
    dict_factory: Callable = enum_dict_factory,
    **kwargs,
):
    str_objs = []
    for obj in objs:
        str_obj = json.dumps(asdict(obj, dict_factory=dict_factory), **kwargs)
        str_objs.append(str_obj + os.linesep)
    with open(ensure_path(file_path), "w", encoding='utf-8') as f:
        f.writelines(str_objs)


def load_json(file_path: str, **kwargs) -> Any:
    with open(file_path, "r", encoding='utf-8') as f:
        return json.load(f, **kwargs)


def load_jsonl(file_path: str, **kwargs) -> Any:
    with open(file_path, "r", encoding='utf-8') as f:
        return [json.loads(line.strip(), **kwargs) for line in f.readlines()]


def load_dataclass_json(
    file_path: str,
    t: Type[T],
    dacite_config: Config = DEFAULT_CONFIG,
    **kwargs,
) -> T:
    with open(file_path, "r", encoding='utf-8') as f:
        return from_dict(t, json.load(f, **kwargs), config=dacite_config)


def load_dataclass_jsonl(
    file_path: str,
    t: Type[T],
    dacite_config: Config = DEFAULT_CONFIG,
    **kwargs,
) -> List[T]:
    def helper(s):
        return from_dict(t, json.loads(s, **kwargs), config=dacite_config)

    with open(file_path, "r", encoding='utf-8') as f:
        return [helper(line.strip()) for line in f.readlines()]


def load_records_csv(file_path: str, **kwargs) -> Dict:
    return pd.read_csv(file_path, **kwargs).to_dict(orient="records")


def load_relations_csv(file_path: str, **kwargs) -> List[Relation]:
    return [Relation(**r) for r in load_records_csv(file_path, **kwargs)]


def load_reducer_csv(
    file_path: str,
    relations: Iterable[Relation],
    raise_: bool,
    **kwargs,
) -> Reducer:
    reducer = Reducer(relations)
    for data in load_records_csv(file_path, **kwargs):
        case_link_data = {
            "r1_type": data["relation1"],
            "r2_type": data["relation2"],
            "case": data["case"],
        }
        case_link = from_dict(RelationalCaseLink, case_link_data, config=DEFAULT_CONFIG)
        reduction_data = {
            "relation_type": data["reduction_type"],
            "order": data["reduction_order"],
        }
        reduction = from_dict(Reduction, reduction_data, config=DEFAULT_CONFIG)
        reducer.register(case_link, reduction, raise_=raise_)
    return reducer


def save_forest_jsonl(
    family_file_path: str,
    data_file_path: str,
    forest: InstantiationForest,
    **kwargs,
):
    save_dataclass_jsonl(family_file_path, *forest.families, **kwargs)
    save_dataclass_jsonl(data_file_path, *forest.data_map.values(), **kwargs)


def load_forest_jsonl(
    family_file_path: str,
    data_file_path: str,
    **kwargs,
) -> InstantiationForest:
    fams = load_dataclass_jsonl(family_file_path, t=InstantiationFamily, **kwargs)
    data = load_dataclass_jsonl(data_file_path, t=InstantiationData, **kwargs)
    forest = InstantiationForest(fams, {d.identifier: d for d in data})
    forest.map_family_data()
    return forest
