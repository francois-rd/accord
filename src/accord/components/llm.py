from dataclasses import dataclass
from typing import Optional
import re

from ..base import Label, QAData, QAGroupId


@dataclass
class LLMResult:
    generated_text: str
    generated_answer_label: Optional[Label]
    prompt_text: Optional[str] = None
    chosen_answer_label: Optional[Label] = None
    qa_group_id: Optional[QAGroupId] = None


class LLMOutputParser:
    def __call__(
        self,
        generated_text: str,
        qa_data: QAData,
        *args,
        **kwargs,
    ) -> Optional[Label]:
        raise NotImplementedError


class PatternMatchLLMOutputParser(LLMOutputParser):
    def __init__(self, pattern: str, flags=None):
        if flags is None:
            self.pattern = re.compile(pattern)
        else:
            self.pattern = re.compile(pattern, flags=flags)

    def __call__(
        self,
        generated_text: str,
        qa_data: QAData,
        *args,
        **kwargs,
    ) -> Optional[Label]:
        for label in qa_data.answer_choices:
            match = self.pattern.search(generated_text)
            if match is not None and match.group(1) == label:
                return label
        return None


class ExactMatchLLMOutputParser(LLMOutputParser):
    def __call__(
        self,
        generated_text: str,
        qa_data: QAData,
        *args,
        **kwargs,
    ) -> Optional[Label]:
        for label in qa_data.answer_choices:
            if generated_text.strip() == label:
                return label
        return None


class SimpleLLMOutputParser(LLMOutputParser):
    def __init__(self, simple_pattern: str = r"Answer:\s*(\w+)", flags=None):
        self.pattern_parser = PatternMatchLLMOutputParser(simple_pattern, flags=flags)
        self.exact_parser = ExactMatchLLMOutputParser()

    def __call__(
        self,
        generated_text: str,
        qa_data: QAData,
        *args,
        **kwargs,
    ) -> Optional[Label]:
        label = self.exact_parser(generated_text, qa_data, *args, **kwargs)
        if label is None:
            return self.pattern_parser(generated_text, qa_data, *args, **kwargs)
        return label


class LLM:
    def __init__(self, model_name: str, parser: Optional[LLMOutputParser] = None):
        self.model_name = model_name
        self.parser = SimpleLLMOutputParser() if parser is None else parser

    def __call__(self, text: str, qa_data: QAData, *args, **kwargs) -> LLMResult:
        raise NotImplementedError
