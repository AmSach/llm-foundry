from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .adapters import EchoBackend, TextBackend


@dataclass
class ReflectionResult:
    draft: str
    critique: str
    final: str


class ReflectionEngine:
    def __init__(self, backend: TextBackend) -> None:
        self.backend = backend

    def answer(self, prompt: str) -> ReflectionResult:
        draft = self.backend.generate(prompt)
        critique = self.backend.generate(
            "Review the draft for factual errors, logic errors, and unsafe claims:\n\n"
            f"PROMPT:\n{prompt}\n\nDRAFT:\n{draft}"
        )
        final = draft if isinstance(self.backend, EchoBackend) else self.backend.generate(
            "Revise the draft using the critique. Keep it concise and correct.\n\n"
            f"PROMPT:\n{prompt}\n\nDRAFT:\n{draft}\n\nCRITIQUE:\n{critique}"
        )
        return ReflectionResult(draft=draft, critique=critique, final=final)


class CounterfactualVerifier:
    def __init__(self, backend: TextBackend) -> None:
        self.backend = backend

    def verify(self, prompt: str, answer: str) -> str:
        return self.backend.generate(
            "Check whether the answer is still supported under counterfactual alternatives.\n\n"
            f"PROMPT:\n{prompt}\n\nANSWER:\n{answer}"
        )


class MajorityVoteConsensus:
    def __init__(self, backend: TextBackend) -> None:
        self.backend = backend

    def run(self, prompt_variants: Iterable[str]) -> str:
        answers: list[str] = [self.backend.generate(prompt) for prompt in prompt_variants]
        if not answers:
            return ""
        counts: dict[str, int] = {}
        for answer in answers:
            counts[answer] = counts.get(answer, 0) + 1
        return max(counts, key=counts.get)
