from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import json
import math
import os
import urllib.request


class TextBackend(Protocol):
    def generate(self, prompt: str) -> str: ...


@dataclass
class EchoBackend:
    def generate(self, prompt: str) -> str:
        return prompt


@dataclass
class OpenAICompatibleBackend:
    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"

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
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]


@dataclass
class SafetyScore:
    delayed_harm_risk: float
    causal_credit: float
    confidence: float


class SafetyLayer:
    def __init__(self, harm_threshold: float = 0.5) -> None:
        self.harm_threshold = harm_threshold

    def score(self, prompt: str, response: str) -> SafetyScore:
        joined = f"{prompt}\n{response}".lower()
        harm_words = ["harm", "attack", "exploit", "steal", "deceive", "poison"]
        hits = sum(1 for word in harm_words if word in joined)
        delayed_harm_risk = min(1.0, 0.15 * hits)
        causal_credit = max(0.0, 1.0 - delayed_harm_risk)
        confidence = 0.5 + 0.1 * hits
        return SafetyScore(delayed_harm_risk, causal_credit, min(confidence, 1.0))

    def allow(self, score: SafetyScore) -> bool:
        return score.delayed_harm_risk < self.harm_threshold


class ReflectionEngine:
    def __init__(self, backend: TextBackend) -> None:
        self.backend = backend

    def answer(self, prompt: str) -> dict:
        draft = self.backend.generate(prompt)
        critique = self.backend.generate(
            "Review the draft for factual errors, logic errors, and unsafe claims:\n\n"
            f"PROMPT:\n{prompt}\n\nDRAFT:\n{draft}"
        )
        revised = self.backend.generate(
            "Revise the draft using the critique. Keep it concise and correct.\n\n"
            f"PROMPT:\n{prompt}\n\nDRAFT:\n{draft}\n\nCRITIQUE:\n{critique}"
        )
        return {"draft": draft, "critique": critique, "final": revised}


class CounterfactualVerifier:
    def __init__(self, backend: TextBackend) -> None:
        self.backend = backend

    def verify(self, prompt: str, answer: str) -> str:
        return self.backend.generate(
            "Check whether the answer is still supported under counterfactual alternatives.\n\n"
            f"PROMPT:\n{prompt}\n\nANSWER:\n{answer}"
        )


class RewardShaper:
    def __init__(self, safety: SafetyLayer) -> None:
        self.safety = safety

    def shaped_reward(self, prompt: str, response: str, base_reward: float) -> float:
        score = self.safety.score(prompt, response)
        return base_reward * score.causal_credit - score.delayed_harm_risk


def build_backend(kind: str, model: str | None = None) -> TextBackend:
    if kind == "echo":
        return EchoBackend()
    if kind == "openai":
        api_key = os.environ["OPENAI_API_KEY"]
        return OpenAICompatibleBackend(api_key=api_key, model=model or "gpt-4o-mini")
    raise ValueError(f"Unknown backend kind: {kind}")
