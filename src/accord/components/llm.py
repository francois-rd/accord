from dataclasses import dataclass
from typing import Optional

from ..base import Label, QAGroupId


@dataclass
class LLMResult:
    generated_text: str
    prompt_text: Optional[str] = None
    chosen_answer_label: Optional[Label] = None
    qa_group_id: Optional[QAGroupId] = None


class LLM:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def __call__(self, text: str, *args, **kwargs) -> LLMResult:
        raise NotImplementedError
