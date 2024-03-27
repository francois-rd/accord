from dataclasses import dataclass
from typing import Optional

from ..base import QAData
from ..components import LLM, LLMResult, LLMOutputParser


@dataclass
class DummyConfig:
    response: str = "dummy"


class DummyLLM(LLM):
    def __init__(
        self,
        model_name: str,
        cfg: DummyConfig,
        parser: Optional[LLMOutputParser] = None,
    ):
        super().__init__(model_name, parser)
        self.response = cfg.response

    def __call__(self, _: str, qa_data: QAData, *args, **kwargs) -> LLMResult:
        return LLMResult(self.response, self.parser(self.response, qa_data))
