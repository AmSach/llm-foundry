from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .adapters import TextBackend
from .safety import SafetyLayer


@dataclass
class EvaluationItem:
    prompt: str
    reference: str | None = None


@dataclass
class EvaluationResult:
    prompt: str
    response: str
    allowed: bool
    delayed_harm_risk: float


class EvaluationSuite:
    def __init__(self, backend: TextBackend, safety: SafetyLayer | None = None) -> None:
        self.backend = backend
        self.safety = safety or SafetyLayer()

    def run(self, items: Iterable[EvaluationItem]) -> list[EvaluationResult]:
        results: list[EvaluationResult] = []
        for item in items:
            response = self.backend.generate(item.prompt)
            score = self.safety.score(item.prompt, response)
            results.append(
                EvaluationResult(
                    prompt=item.prompt,
                    response=response,
                    allowed=self.safety.allow(score),
                    delayed_harm_risk=score.delayed_harm_risk,
                )
            )
        return results
