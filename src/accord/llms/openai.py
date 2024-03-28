from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import os

from retry import retry

from ..base import QAData
from ..components import LLM, LLMResult, LLMOutputParser


@dataclass
class OpenAIConfig:
    # The prompt for system instructions.
    system_prompt: str = ""

    # See openai.OpenAI.chat.completions.create for details.
    # NOTE: Skip 'model' and 'messages', which is handled specially.
    query_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "max_tokens": 5,
            "seed": 314159,
        },
    )


class OpenAILLM(LLM):
    def __init__(
        self,
        model_name: str,
        cfg: OpenAIConfig,
        parser: Optional[LLMOutputParser] = None,
    ):
        from openai import OpenAI  # Delayed import.

        super().__init__(model_name, parser)
        self.cfg = cfg
        self.llm = OpenAI(api_key=os.environ["OPENAI_API_KEY"]).chat.completions.create

    @retry(delay=1)
    def __call__(self, text: str, qa_data: QAData, *args, **kwargs) -> LLMResult:
        response = self.llm(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.cfg.system_prompt},
                {"role": "user", "content": text}
            ],
            **self.cfg.query_params,
        )
        generated_text = response.choices[0].message.content
        return LLMResult(generated_text, self.parser(generated_text, qa_data))
