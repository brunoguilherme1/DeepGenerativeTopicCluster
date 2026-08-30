"""Lazy Hugging Face causal-LM client for LLM-based cluster refinement.

LAZY LOADING: the tokenizer/model are constructed ONLY inside
`_ensure_loaded()`, called from the first `.generate()` invocation -
NEVER at `LLMClient.__init__()` or at import time. An experiment run
where every edge point already cache-hits (llm/cache.py), or where
`--edge-fraction` selects zero points, never downloads or loads the LLM
at all.

Designed for Google Colab GPU execution: 4-bit quantization via
`bitsandbytes`/`transformers.BitsAndBytesConfig` when a CUDA device is
available (the default, `--quantization 4bit`); `--quantization none`
runs full precision (correct but impractically slow for a 7B model on
CPU - never silently substituted with a smaller model instead).

DETERMINISM: `do_sample=False` (greedy decoding) is always used;
`temperature` is recorded in run metadata as `0` (the user-facing,
"deterministic" value) but is NOT passed to `generate()` - passing an
explicit `temperature` alongside `do_sample=False` triggers a
`transformers` warning ("temperature is set but do_sample is False") in
recent versions, since greedy decoding ignores it entirely; omitting it
avoids that warning without changing behavior.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    completion_tokens: int


class LLMClient:
    #: Recorded in run metadata; NOT passed to `generate()` - see module docstring.
    TEMPERATURE = 0.0

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        device: str = "auto",
        quantization: str = "4bit",  # "4bit" | "none"
        max_new_tokens: int = 64,
    ):
        if quantization not in ("4bit", "none"):
            raise ValueError(f"quantization must be '4bit' or 'none', got {quantization!r}")
        self.model_name = model_name
        self.device = device
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self._tokenizer = None
        self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        model_kwargs: dict = {"device_map": self.device}
        if self.quantization == "4bit":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "quantization='4bit' requires a CUDA device (bitsandbytes) - none detected. "
                    "Pass --quantization none to run in full precision on CPU (very slow for a 7B "
                    "model) or run on a GPU runtime (e.g. Google Colab)."
                )
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        self._model.eval()

    def generate(self, prompt: str) -> GenerationResult:
        """Deterministic (greedy, `do_sample=False`, single beam)
        generation - loads the model on first call (see module
        docstring)."""
        self._ensure_loaded()
        import torch

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        prompt_token_count = int(inputs["input_ids"].shape[1])
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        completion_ids = output_ids[0][prompt_token_count:]
        text = self._tokenizer.decode(completion_ids, skip_special_tokens=True)
        return GenerationResult(
            text=text,
            prompt_tokens=prompt_token_count,
            completion_tokens=int(completion_ids.shape[0]),
        )
