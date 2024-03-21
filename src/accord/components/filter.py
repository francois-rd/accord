from typing import Any, Iterable, Optional, TypeVar
import random

T = TypeVar("T")


class GeneratorFilter:
    def __init__(self, filter_prob: float, seed: Optional[Any] = None):
        if seed is not None:
            random.seed(seed)
        if filter_prob < 0 or filter_prob > 1:
            raise ValueError("Filter probability must be in range [0, 1].")
        self.prob = filter_prob

    def __call__(self, generator: Iterable[T]) -> Iterable[T]:
        for x in generator:
            if random.random() >= self.prob:
                yield x
