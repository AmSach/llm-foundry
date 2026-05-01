from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SafetyScore:
    delayed_harm_risk: float
    causal_credit: float
    confidence: float


class SafetyLayer:
    def __init__(self, harm_threshold: float = 0.25) -> None:
        self.harm_threshold = harm_threshold

    def score(self, prompt: str, response: str) -> SafetyScore:
        joined = f"{prompt}\n{response}".lower()
        harm_words = ["harm", "attack", "exploit", "steal", "deceive", "poison"]
        hits = sum(1 for word in harm_words if word in joined)
        delayed_harm_risk = min(1.0, 0.15 * hits)
        causal_credit = max(0.0, 1.0 - delayed_harm_risk)
        confidence = min(1.0, 0.5 + 0.1 * hits)
        return SafetyScore(delayed_harm_risk, causal_credit, confidence)

    def allow(self, score: SafetyScore) -> bool:
        return score.delayed_harm_risk < self.harm_threshold


class RewardShaper:
    def __init__(self, safety: SafetyLayer) -> None:
        self.safety = safety

    def shaped_reward(self, prompt: str, response: str, base_reward: float) -> float:
        score = self.safety.score(prompt, response)
        return base_reward * score.causal_credit - score.delayed_harm_risk
