from typing import Dict, Iterable
import os

import pandas as pd

from ...base import RelationType, Term
from ...components import TermFormatter


class ConceptNet(TermFormatter):
    def __init__(self, input_dir: str, relation_map: Dict[str, RelationType]):
        self.df_map = {}
        for path, _, files in os.walk(input_dir):
            for file in files:
                df = pd.read_csv(os.path.join(path, file), header=None)
                df.columns = [self.source, self.target]
                conceptnet_relation = os.path.splitext(file)[0]
                relation_type = relation_map[conceptnet_relation]
                self.df_map[relation_type] = df
        self.reverse_relation_map = {v: k for k, v in relation_map.items()}
        self.format_map = {}

    def get_assertions(self, relation_type: RelationType) -> pd.DataFrame:
        return self.df_map[relation_type]

    def get_all_assertions(self) -> Dict[RelationType, pd.DataFrame]:
        return self.df_map

    def format(self, term: Term, language: str, *args, **kwargs) -> Term:
        if term.startswith(f"/c/{language}/"):
            # Term is already probably formatted correctly. Still, have it be
            # in lowercase and without spaces just in case.
            return term.lower().replace(' ', '_')
        elif term.startswith("/c/"):
            raise ValueError(f"Mismatch for term={term} and language={language}.")
        else:
            return f"/c/{language}/{term.lower().replace(' ', '_')}"

    def get_relations(self, node: str, other: str) -> Iterable[RelationType]:
        for relation_type, df in self.df_map.items():
            s_node = self._find_matches(df, self.source, node)
            t_node = self._find_matches(df, self.target, node)
            s_other = self._find_matches(df, self.source, other)
            t_other = self._find_matches(df, self.target, other)
            if not df.loc[(s_node & t_other) | (s_other & t_node)].empty:
                yield self.reverse_relation_map[relation_type]

    @staticmethod
    def _find_matches(df, column, word):
        return df[column].apply(lambda x: x == word or x.startswith(word + "/"))

    @property
    def source(self):
        return "source"

    @property
    def target(self):
        return "target"
