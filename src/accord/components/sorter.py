from typing import Callable, Iterable, List, Optional
import random

from .instantiator import InstantiatorVariant, Query, QueryResult
from ..base import Term


class QueryResultSorter:
    def new_collection(self, variant: InstantiatorVariant, *args, **kwargs):
        raise NotImplementedError

    def add_query_result(
            self,
            result: QueryResult,
            query: Query,
            *args,
            query_existing_term: Optional[Term] = None,
            **kwargs,
    ):
        raise NotImplementedError

    def sort_collection(self) -> List[Term]:
        raise NotImplementedError


class SemanticDistanceCalculator:
    def __call__(
            self,
            query_term: Term,
            query: Query,
            *args,
            query_existing_term: Optional[Term] = None,
            **kwargs,
    ) -> float:
        # TODO: NOTE: All three Terms here are formatted. Depending on the underlying
        #  implementation of the distance metric, we may need to "unformat" them. But,
        #  this operation may be dependent on the specifics of both this implementation
        #  AND the original formatting (e.g., ConceptNet). For example, ConceptNet
        #  formatting includes both a language component as well as (occasionally)
        #  word-sense information. This may be useful to some distance calculator
        #  implementations but not for others. Details are left open.
        raise NotImplementedError


class SemanticDistanceSorter(QueryResultSorter):
    def __init__(
            self,
            target_distance: float,
            semantic_distance_calculator: SemanticDistanceCalculator,
            distance_aggregator: Callable[[Iterable[float]], float] = sum,
    ):
        self.target = target_distance
        self.semantic_distance = semantic_distance_calculator
        self.aggregator = distance_aggregator
        self.collection = None
        self.count = 0

    def new_collection(self, _: InstantiatorVariant, *args, **kwargs):
        self.collection = {}
        self.count = 0

    def add_query_result(
            self,
            result: QueryResult,
            query: Query,
            *args,
            query_existing_term: Optional[Term] = None,
            **kwargs,
    ):
        self.count += 1
        for term in result:
            distance = self.semantic_distance(
                term, query, *args, query_existing_term=query_existing_term, **kwargs,
            )
            self.collection.setdefault(term, []).append(abs(self.target - distance))

    def sort_collection(self) -> List[Term]:
        items = self.collection.items()
        scores = {k: self.aggregator(v) for k, v in items if len(v) == self.count}
        self.collection = None
        self.count = 0
        return [k for k, _ in sorted(scores.items(), key=lambda item: item[1])]


class RandomUnSorter(QueryResultSorter):
    def __init__(self):
        self.collection = None

    def new_collection(self, _: InstantiatorVariant, *args, **kwargs):
        self.collection = []

    def add_query_result(
            self,
            result: QueryResult,
            _: Query,
            *args,
            __: Optional[Term] = None,
            **kwargs,
    ):
        self.collection.append(result)

    def sort_collection(self) -> List[Term]:
        results = sorted(set.intersection(*self.collection))
        random.shuffle(results)
        self.collection = None
        return results
