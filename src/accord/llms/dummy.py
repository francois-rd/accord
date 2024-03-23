from dataclasses import dataclass

from ..components import LLM, LLMResult


@dataclass
class DummyConfig:
    response: str = "dummy"


class DummyLLM(LLM):
    def __init__(self, model_name: str, cfg: DummyConfig):
        super().__init__(model_name)
        self.response = cfg.response

    def __call__(self, _: str, *args, **kwargs) -> LLMResult:
        return LLMResult(self.response)
