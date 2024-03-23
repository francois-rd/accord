from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..components import LLM, LLMResult


@dataclass
class TransformersConfig:
    # RNG seed for replication of results.
    seed: int = 314159

    # The prompt format to represent system instructions. Should contain a "{}"
    # wherever the specific text of the __call__() method ought to be inserted.
    system_prompt_format: str = "{}"

    # Model quantization options for bitsandbytes.
    quantization: Optional[str] = None

    # See transformers.AutoModelForCausalLM.from_pretrained for details.
    # NOTE: Skip 'quantization_config', which is handled specially.
    model_params: Dict[str, Any] = field(
        default_factory=lambda: {"trust_remote_code": True},
    )

    # See transformers.AutoTokenizer.from_pretrained for details.
    tokenizer_params: Dict[str, Any] = field(default_factory=dict)

    # See transformers.Pipeline for details.
    # NOTE: Skip 'task', 'model', 'tokenizer', 'torch_dtype', which are handled
    # specially.
    pipeline_params: Dict[str, Any] = field(default_factory=dict)

    # See transformers.GenerationConfig for details.
    generation_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "return_full_text": False,
            "max_new_tokens": 5,
            "num_return_sequences": 1,
        },
    )


class TransformersLLM(LLM):
    def __init__(self, model_name: str, cfg: TransformersConfig):
        # Delayed imports.
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            GenerationConfig,
            pipeline,
            set_seed,
        )

        set_seed(cfg.seed)
        super().__init__(model_name)
        self.cfg = cfg
        self.gen_cfg = GenerationConfig(**self.cfg.generation_config)

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
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                **self.cfg.model_params,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name, **self.cfg.model_params,
            )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, **self.cfg.tokenizer_params,
        )
        self.llm = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            **self.cfg.pipeline_params,
        )

    def __call__(self, text: str, *args, **kwargs) -> LLMResult:
        text = self.cfg.system_prompt_format.format(text)
        output = self.llm(text, **self.gen_cfg.to_dict())
        return LLMResult(output[0]['generated_text'])
