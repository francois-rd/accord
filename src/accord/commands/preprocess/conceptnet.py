from dataclasses import dataclass, field
from typing import List, Dict
import os

import pandas as pd

from ..configs import GeneralConfig, ResourcesConfig
from ...base import RelationType
from ...databases.conceptnet import AntiFactualMethod


@dataclass
class ConceptNetConfig:
    # File path to CSV file containing the raw ConceptNet 5.7.0 assertions (edge data).
    raw_data_file: str = "raw/conceptnet-assertions-5.7.0.csv"

    # Directory in which to store the preprocessed data (one file per relation).
    preprocessed_dir: str = "preprocessed"

    # The name of ConceptNet relations to keep as part of preprocessing. For example,
    # to keep the relation /r/Antonym, add the relation name "Antonym" to this field.
    relations: List[str] = field(default_factory=list)

    # The name of ConceptNet languages to keep as part of preprocessing. For example,
    # to keep the language /c/en (English), add the language name "en" to this field.
    languages: List[str] = field(default_factory=list)

    # A mapping between ConceptNet relation names and Relation type names. For example,
    # {"UsedFor": "purpose"} maps the ConceptNet relation /r/UsedFor to the relation
    # type named "purpose".
    relation_map: Dict[str, RelationType] = field(default_factory=dict)

    # The method of ConceptNet-based anti-factual instantiation to use.
    anti_factual_method: AntiFactualMethod = AntiFactualMethod.SAME_RELATION


def preprocess(
    resources: ResourcesConfig, general: GeneralConfig, cfg: ConceptNetConfig
):
    # Merge resource paths.
    raw_data_file = os.path.join(resources.term_database_dir, cfg.raw_data_file)
    preprocessed_dir = os.path.join(resources.term_database_dir, cfg.preprocessed_dir)

    # Load the data.
    if general.verbose:
        print(f"Loading raw data...", flush=True)
    df = pd.read_csv(raw_data_file, sep="\t", header=None)
    df.columns = ["uri", "relation", "source", "target", "info"]
    if general.verbose:
        print(f"Done.", flush=True)

    # Drop irrelevant columns.
    if general.verbose:
        print(f"Processing...", flush=True)
    df.drop(columns=["uri", "info"], inplace=True)

    # Keep only relevant relations.
    relations = [f"/r/{relation}" for relation in cfg.relations]
    df = df.loc[df["relation"].isin(relations)]

    # Keep only relevant languages.
    for language in cfg.languages:
        df = df.loc[df["source"].str.startswith(f"/c/{language}")]
        df = df.loc[df["target"].str.startswith(f"/c/{language}")]
    if general.verbose:
        print(f"Done.", flush=True)

    # Split data by relation type and save to file.
    if general.verbose:
        print(f"Saving...", flush=True)
    os.makedirs(preprocessed_dir, exist_ok=True)
    for relation, new_df in df.groupby(df["relation"]):
        file_name = f"{relation.replace('/r/', '')}.csv"
        file_path = os.path.join(preprocessed_dir, file_name)
        new_df.drop(columns=["relation"], inplace=True)
        new_df.to_csv(file_path, index=False, header=False)
    if general.verbose:
        print(f"Done.", flush=True)
