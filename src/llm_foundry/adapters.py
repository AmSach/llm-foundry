from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
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
class SequenceBackend:
    responses: list[str] = field(default_factory=list)
    default_response: str = ""
    index: int = 0

    def generate(self, prompt: str) -> str:
        if self.index < len(self.responses):
            response = self.responses[self.index]
            self.index += 1
            return response
        return self.default_response


@dataclass
class ScriptedBackend:
    responses: dict[str, str]
    default_response: str = ""

    def generate(self, prompt: str) -> str:
        return self.responses.get(prompt, self.default_response)


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


@dataclass
class OpenAICompatibleBackend:
    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: int = 120
    temperature: float = 0.2

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
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
class AnthropicBackend:
    api_key: str
    model: str
    base_url: str = "https://api.anthropic.com"
    timeout_seconds: int = 120
    temperature: float = 0.2
    max_tokens: int = 1024

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        req = urllib.request.Request(
            f"{self.base_url}/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Anthropic request failed with HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Anthropic request failed: {exc.reason}") from exc
        parts = [block.get("text", "") for block in data.get("content", []) if isinstance(block, dict) and block.get("type") == "text"]
        return "".join(parts)


@dataclass(frozen=True)
class ApiEndpointSpec:
    name: str
    provider: str
    base_url: str
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: int = 120
    temperature: float = 0.2
    max_tokens: int = 1024


@dataclass
class MultiEndpointBackend:
    candidates: list[TextBackend] = field(default_factory=list)
    backend_labels: list[str] = field(default_factory=list)
    strategy: str = "failover"
    index: int = 0

    def generate(self, prompt: str) -> str:
        if not self.candidates:
            raise RuntimeError("No API endpoints configured")
        pairs = list(zip(self.backend_labels, self.candidates))
        if self.strategy == "round_robin":
            start = self.index % len(self.candidates)
            self.index = (self.index + 1) % len(self.candidates)
            ordered = pairs[start:] + pairs[:start]
        else:
            ordered = pairs

        errors: list[str] = []
        for label, backend in ordered:
            try:
                response = backend.generate(prompt)
                if response.strip():
                    return response
            except Exception as exc:
                errors.append(f"{label}: {exc}")
        raise RuntimeError("All configured API endpoints failed: " + " | ".join(errors) if errors else "All configured API endpoints failed")

    @property
    def backend_names(self) -> list[str]:
        return list(self.backend_labels)


def load_api_endpoint_specs(api_endpoints_file: str | None = None, api_endpoints_json: str | None = None) -> list[ApiEndpointSpec]:
    if not api_endpoints_file and not api_endpoints_json:
        return []
    if api_endpoints_file:
        data = json.loads(Path(api_endpoints_file).read_text())
    else:
        data = json.loads(api_endpoints_json or "[]")
    if isinstance(data, dict):
        data = data.get("endpoints", [])
    if not isinstance(data, list):
        raise ValueError("API endpoint config must be a JSON list or an object with an 'endpoints' list")
    specs: list[ApiEndpointSpec] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Each API endpoint config entry must be an object")
        provider = str(item.get("provider") or item.get("kind") or "openai-compatible")
        api_key_env = str(item.get("api_key_env", "OPENAI_API_KEY"))
        if provider == "anthropic" and api_key_env == "OPENAI_API_KEY":
            api_key_env = "ANTHROPIC_API_KEY"
        specs.append(
            ApiEndpointSpec(
                name=str(item.get("name", f"endpoint-{len(specs) + 1}")),
                provider=provider,
                base_url=str(item.get("base_url") or ("https://api.anthropic.com" if provider == "anthropic" else "https://api.openai.com/v1")),
                model=str(item["model"]),
                api_key_env=api_key_env,
                timeout_seconds=int(item.get("timeout_seconds", 120)),
                temperature=float(item.get("temperature", 0.2)),
                max_tokens=int(item.get("max_tokens", 1024)),
            )
        )
    return specs


def build_multi_endpoint_backend(api_endpoints_file: str | None = None, api_endpoints_json: str | None = None, strategy: str = "failover") -> MultiEndpointBackend:
    specs = load_api_endpoint_specs(api_endpoints_file=api_endpoints_file, api_endpoints_json=api_endpoints_json)
    if not specs:
        raise ValueError("At least one API endpoint is required")
    candidates: list[TextBackend] = []
    labels: list[str] = []
    for spec in specs:
        labels.append(spec.name)
        if spec.provider == "anthropic":
            api_key = os.environ.get(spec.api_key_env)
            if not api_key:
                raise ValueError(f"Missing API key environment variable: {spec.api_key_env}")
            candidates.append(
                AnthropicBackend(
                    api_key=api_key,
                    model=spec.model,
                    base_url=spec.base_url,
                    timeout_seconds=spec.timeout_seconds,
                    temperature=spec.temperature,
                    max_tokens=spec.max_tokens,
                )
            )
        else:
            api_key = os.environ.get(spec.api_key_env, "") if spec.api_key_env else ""
            candidates.append(
                OpenAICompatibleBackend(
                    api_key=api_key,
                    model=spec.model,
                    base_url=spec.base_url,
                    timeout_seconds=spec.timeout_seconds,
                    temperature=spec.temperature,
                )
            )
    return MultiEndpointBackend(candidates=candidates, backend_labels=labels, strategy=strategy)


def build_backend(
    kind: str,
    model: str | None = None,
    *,
    api_endpoints_file: str | None = None,
    api_endpoints_json: str | None = None,
    api_strategy: str = "failover",
) -> TextBackend:
    if api_endpoints_file or api_endpoints_json or kind in {"multi", "openai-multi"}:
        return build_multi_endpoint_backend(
            api_endpoints_file=api_endpoints_file,
            api_endpoints_json=api_endpoints_json,
            strategy=api_strategy,
        )
    if kind == "echo":
        return EchoBackend()
    if kind == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        return OpenAICompatibleBackend(api_key=api_key, model=model or "gpt-4o-mini")
    if kind == "anthropic":
        api_key = os.environ["ANTHROPIC_API_KEY"]
        return AnthropicBackend(api_key=api_key, model=model or "claude-sonnet-4-20250514")
    if kind == "hf":
        if not model:
            raise ValueError("model is required for hf backend")
        return HuggingFacePipelineBackend(model_name=model)
    raise ValueError(f"Unknown backend kind: {kind}")
