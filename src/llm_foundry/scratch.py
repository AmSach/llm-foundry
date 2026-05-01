from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import random
from typing import Iterable


@dataclass
class ScratchConfig:
    vocab_size: int
    context_size: int
    d_model: int = 128
    learning_rate: float = 0.05


class TinyCausalLM:
    def __init__(self, config: ScratchConfig) -> None:
        self.config = config
        self.weights = [0.0] * config.vocab_size

    def predict_logits(self, token_id: int) -> list[float]:
        base = self.weights[token_id % len(self.weights)]
        return [base + (i / max(1, self.config.vocab_size - 1)) for i in range(self.config.vocab_size)]

    def train_step(self, token_id: int, target_id: int) -> float:
        logits = self.predict_logits(token_id)
        max_logit = max(logits)
        exps = [math.exp(x - max_logit) for x in logits]
        denom = sum(exps)
        probs = [x / denom for x in exps]
        loss = -math.log(max(probs[target_id], 1e-9))
        grad = probs[target_id] - 1.0
        self.weights[token_id % len(self.weights)] -= self.config.learning_rate * grad
        return loss

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"config": self.config.__dict__, "weights": self.weights}, indent=2))


class ToyTokenizer:
    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.token_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_token = {i: ch for ch, i in self.token_to_id.items()}

    def encode(self, text: str) -> list[int]:
        return [self.token_to_id[ch] for ch in text if ch in self.token_to_id]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.id_to_token[i] for i in ids if i in self.id_to_token)


def train_from_text(corpus_path: str, steps: int, context: int, d_model: int) -> dict:
    text = Path(corpus_path).read_text()
    tokenizer = ToyTokenizer(text)
    config = ScratchConfig(vocab_size=max(2, len(tokenizer.token_to_id)), context_size=context, d_model=d_model)
    model = TinyCausalLM(config)
    token_ids = tokenizer.encode(text)
    if len(token_ids) < 2:
        return {"loss": None, "message": "corpus too small"}
    losses = []
    for step in range(steps):
        idx = step % (len(token_ids) - 1)
        loss = model.train_step(token_ids[idx], token_ids[idx + 1])
        losses.append(loss)
    return {"final_loss": sum(losses) / len(losses), "vocab_size": config.vocab_size}
