from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..base import QAData
from ..components import LLM, LLMResult, LLMOutputParser


@dataclass
class TransformersConfig:
    # RNG seed for replication of results.
    seed: int = 314159

    # The prompt for system instructions, or None for model that don't support the
    # transformers.Pipeline chat templating functionality.
    system_prompt: Optional[str] = None

    # Model quantization options for bitsandbytes.
    quantization: Optional[str] = None

    # See transformers.AutoModelForCausalLM.from_pretrained for details.
    # NOTE: Skip 'quantization_config', which is handled specially.
    model_params: Dict[str, Any] = field(
        default_factory=lambda: {"trust_remote_code": True},
    )

    # See transformers.Pipeline for details.
    # NOTE: Skip 'task', 'model', and 'torch_dtype', which are handled specially.
    pipeline_params: Dict[str, Any] = field(default_factory=dict)

    # See transformers.GenerationConfig for details.
    generation_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "return_full_text": False,
            "max_new_tokens": 5,
            "num_return_sequences": 1,
        },
    )


class TransformersLLM(LLM):
    def __init__(
        self,
        model_name: str,
        cfg: TransformersConfig,
        parser: Optional[LLMOutputParser] = None,
    ):
        # Delayed imports.
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            pipeline,
            set_seed,
        )

        # Basic initialization.
        set_seed(cfg.seed)
        super().__init__(model_name, parser)
        self.cfg = cfg

        # Quantization.
        model_params = self.cfg.model_params
        if self.cfg.quantization is not None:
            if self.cfg.quantization == "8bit":
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            elif self.cfg.quantization == "4bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                raise ValueError(f"Unsupported quantization: {self.cfg.quantization}")
            model_params.update({"quantization_config": bnb_config})

        # Pipeline initialization.
        self.llm = pipeline(
            task="text-generation",
            model=AutoModelForCausalLM.from_pretrained(self.model_name, **model_params),
            tokenizer=AutoTokenizer.from_pretrained(self.model_name),
            torch_dtype=torch.bfloat16,
            **self.cfg.pipeline_params,
        )

    def __call__(self, text: str, qa_data: QAData, *args, **kwargs) -> LLMResult:
        if self.cfg.system_prompt is None:
            prompt = text
        else:
            prompt = [
                {"role": "system", "content": self.cfg.system_prompt},
                {"role": "user", "content": text}
            ]
        output = self.llm(prompt, **self.cfg.generation_params)
        if self.cfg.system_prompt is None:
            generated_text = output[0]['generated_text']
        else:
            generated_text = output[0]['generated_text'][-1]["content"]
        return LLMResult(generated_text, self.parser(generated_text, qa_data))
