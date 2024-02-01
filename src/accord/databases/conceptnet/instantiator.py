from dataclasses import replace
from typing import Optional
from enum import Enum

import pandas as pd

from .interface import ConceptNet
from ...components import Instantiator, InstantiatorVariant, Query, QueryResult
from ...base import Term


class AntiFactualMethod(Enum):
    """
    The method of ConceptNet-based anti-factual instantiation to use. Each method
    limits the choice of anti-factual variable instantiation in a different way.

    ALL_RELATIONS:
        For a given relation type, r, and partner variable instantiation, p, the
       anti-factual variable can take on any value in ConceptNet, v, that is not an
       existing assertion in ConceptNet. That is, any v from any assertion
       (v, s, q) xor (q, s, v) where (v, r, p) xor (p, r, v) (xor: depending on
       whether p is the source or target) is not an existing assertion.

    SAME_RELATION:
        For a given relation type, r, and partner variable instantiation, p, the
       anti-factual variable can take on any value in ConceptNet, v, that is not an
       existing assertion in ConceptNet, but does share the same relation type. That
       is, any v from any assertion (v, r, q) xor (q, r, v) where (v, r, p) xor
       (p, r, v) (xor: depending on whether p is the source or target) is not an
       existing assertion.

    OTHER_RELATIONS:
       For a given relation type, r, and partner variable instantiation, p, the
       anti-factual variable can take on any value in ConceptNet, v, that is not an
       existing assertion in ConceptNet, and does not share the same relation type.
       That is, any v from any assertion (v, s, q) xor (q, s, v) where (v, r, p) xor
       (p, r, v) (xor: depending on whether p is the source or target) is not an
       existing assertion and where s != r.

    SAME_PARTNER:
       For a given relation type, r, and partner variable instantiation, p, the
       anti-factual variable can take on any value in ConceptNet, v, that is not an
       existing assertion in ConceptNet, but does share the same partner
       instantiation. That is, any v from any assertion (v, s, p) xor (q, s, p)
       where (v, r, p) xor (p, r, v) (xor: depending on whether p is the source or
       target) is not an existing assertion.
    """
    ALL_RELATIONS = "ALL_RELATIONS"
    SAME_RELATION = "SAME_RELATION"
    OTHER_RELATIONS = "OTHER_RELATIONS"
    SAME_PARTNER = "SAME_PARTNER"


class ConceptNetInstantiator(Instantiator):
    def __init__(
            self,
            concept_net: ConceptNet,
            language: str,
            variant: InstantiatorVariant,
            method: Optional[AntiFactualMethod] = None,
    ):
        self.concept_net = concept_net
        self.language = language
        self.variant = variant
        self.method = method

    def query(self, query: Query, *args, **kwargs) -> QueryResult:
        partner_term = self.concept_net.format(query.partner_term, self.language)
        query = replace(query, partner_term=partner_term)
        if self.variant == InstantiatorVariant.FACTUAL:
            return self._factual_query(query)
        elif self.variant == InstantiatorVariant.ANTI_FACTUAL:
            return self._anti_factual_query(query)
        else:
            raise ValueError(f"Unsupported instantiator variant: {self.variant}")

    def _factual_query(self, query: Query) -> QueryResult:
        partner_col = self._partner_column(query)
        df = self.concept_net.get_assertions(query.template.relation_type)
        df = self._keep_matching(df, partner_col, query.partner_term)
        return set(df[self._query_column(query)].tolist())

    def _anti_factual_query(self, query: Query) -> QueryResult:
        query_col = self._query_column(query)
        partner_col = self._partner_column(query)
        factual_blacklist = self._factual_query(query)
        if self.method == AntiFactualMethod.ALL_RELATIONS:
            hits = set()
            for relation_type, df in self.concept_net.get_all_assertions().items():
                hits.update(df[query_col].tolist())
        elif self.method == AntiFactualMethod.SAME_RELATION:
            df = self.concept_net.get_assertions(query.template.relation_type)
            hits = set(df[query_col].tolist())
        elif self.method == AntiFactualMethod.OTHER_RELATIONS:
            hits = set()
            for relation_type, df in self.concept_net.get_all_assertions().items():
                if relation_type != query.template.relation_type:
                    hits.update(df[query_col].tolist())
        elif self.method == AntiFactualMethod.SAME_PARTNER:
            hits = set()
            for df in self.concept_net.get_all_assertions().values():
                df = self._keep_matching(df, partner_col, query.partner_term)
                hits.update(df[query_col].tolist())
        else:
            raise ValueError(f"Unsupported anti-factual method: {self.method}")
        return hits - factual_blacklist

    def _query_column(self, query: Query) -> str:
        source, target = self.concept_net.source, self.concept_net.target
        return source if query.template.source_id == query.query_id else target

    def _partner_column(self, query: Query) -> str:
        source, target = self.concept_net.source, self.concept_net.target
        return target if query.template.source_id == query.query_id else source

    @staticmethod
    def _keep_matching(
            df: pd.DataFrame,
            partner_col: str,
            partner_term: Term,
    ) -> pd.DataFrame:
        return df.loc[df[partner_col] == partner_term]
