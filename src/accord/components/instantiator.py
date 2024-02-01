from dataclasses import dataclass
from typing import Set
from enum import Enum

from ..base import RelationalTemplate, Term, VarId


class InstantiatorVariant(Enum):
    FACTUAL = "FACTUAL"
    ANTI_FACTUAL = "ANTI_FACTUAL"


@dataclass
class Query:
    template: RelationalTemplate
    query_id: VarId
    partner_term: Term


QueryResult = Set[Term]


class Instantiator:
    def query(self, query: Query, *args, **kwargs) -> QueryResult:
        raise NotImplementedError
