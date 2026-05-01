from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import json
import os
import urllib.error
import urllib.request


class TextBackend(Protocol):
    def generate(self, prompt: str) -> str: ...


@dataclass
class EchoBackend:
    def generate(self, prompt: str) -> str:
        return prompt


@dataclass
class FixedBackend:
    response: str

    def generate(self, prompt: str) -> str:
        return self.response


@dataclass
class OpenAICompatibleBackend:
    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: int = 120

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI-compatible request failed with HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenAI-compatible request failed: {exc.reason}") from exc
        return data["choices"][0]["message"]["content"]


@dataclass
class HuggingFacePipelineBackend:
    model_name: str
    max_new_tokens: int = 256

    def generate(self, prompt: str) -> str:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install the 'local' extras to use Hugging Face backends") from exc

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def build_backend(kind: str, model: str | None = None) -> TextBackend:
    if kind == "echo":
        return EchoBackend()
    if kind == "openai":
        api_key = os.environ["OPENAI_API_KEY"]
        return OpenAICompatibleBackend(api_key=api_key, model=model or "gpt-4o-mini")
    if kind == "hf":
        if not model:
            raise ValueError("model is required for hf backend")
        return HuggingFacePipelineBackend(model_name=model)
    raise ValueError(f"Unknown backend kind: {kind}")
