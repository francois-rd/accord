from dataclasses import dataclass
from typing import Optional

from ..base import Label, QAGroupId


@dataclass
class LLMResult:
    generated_text: str
    chosen_answer_label: Optional[Label] = None
    qa_group_id: Optional[QAGroupId] = None


class LLM:
    def __call__(self, text: str, *args, **kwargs) -> LLMResult:
        raise NotImplementedError


class DummyLLM(LLM):
    def __init__(self, response: str):
        self.response = response

    def __call__(self, _: str, *args, **kwargs) -> LLMResult:
        return LLMResult(self.response)
