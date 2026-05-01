from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json
import math
import random


@dataclass
class ScratchTrainingConfig:
    context_size: int
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1
    learning_rate: float = 3e-4
    batch_size: int = 8
    steps: int = 100
    device: str = "cpu"


class CharacterTokenizer:
    def __init__(self, text: str) -> None:
        chars = sorted(set(text))
        self.token_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_token = {i: ch for ch, i in self.token_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str) -> list[int]:
        return [self.token_to_id[ch] for ch in text if ch in self.token_to_id]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.id_to_token[i] for i in ids if i in self.id_to_token)


class TinyCausalLM:
    def __init__(self, vocab_size: int, learning_rate: float = 0.05) -> None:
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.weights = [0.0] * vocab_size

    def predict_logits(self, token_id: int) -> list[float]:
        base = self.weights[token_id % len(self.weights)]
        return [base + (i / max(1, self.vocab_size - 1)) for i in range(self.vocab_size)]

    def train_step(self, token_id: int, target_id: int) -> float:
        logits = self.predict_logits(token_id)
        max_logit = max(logits)
        exps = [math.exp(x - max_logit) for x in logits]
        denom = sum(exps)
        probs = [x / denom for x in exps]
        loss = -math.log(max(probs[target_id], 1e-9))
        grad = probs[target_id] - 1.0
        self.weights[token_id % len(self.weights)] -= self.learning_rate * grad
        return loss

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"vocab_size": self.vocab_size, "weights": self.weights}, indent=2))


class TorchUnavailableError(RuntimeError):
    pass


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise TorchUnavailableError("Install the 'local' extras to train the transformer model") from exc
    return torch, nn, F


class CausalSelfAttentionBlock:
    pass


class TransformerLM:
    def __init__(self, vocab_size: int, config: ScratchTrainingConfig) -> None:
        torch, nn, _ = _require_torch()
        self.torch = torch
        self.nn = nn
        self.config = config
        self.model = self._build(vocab_size)

    def _build(self, vocab_size: int):
        torch, nn, F = self.torch, self.nn, self.torch.nn.functional

        class Block(nn.Module):
            def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
                super().__init__()
                self.ln1 = nn.LayerNorm(d_model)
                self.q = nn.Linear(d_model, d_model)
                self.k = nn.Linear(d_model, d_model)
                self.v = nn.Linear(d_model, d_model)
                self.proj = nn.Linear(d_model, d_model)
                self.ln2 = nn.LayerNorm(d_model)
                self.ff = nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Linear(d_ff, d_model),
                )
                self.n_heads = n_heads
                self.d_model = d_model
                self.dropout = nn.Dropout(dropout)

            def forward(self, x):
                b, t, c = x.shape
                h = self.n_heads
                x1 = self.ln1(x)
                q = self.q(x1).view(b, t, h, c // h).transpose(1, 2)
                k = self.k(x1).view(b, t, h, c // h).transpose(1, 2)
                v = self.v(x1).view(b, t, h, c // h).transpose(1, 2)
                att = (q @ k.transpose(-2, -1)) / math.sqrt(c // h)
                mask = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool))
                att = att.masked_fill(~mask, float("-inf"))
                att = torch.softmax(att, dim=-1)
                y = (att @ v).transpose(1, 2).contiguous().view(b, t, c)
                x = x + self.dropout(self.proj(y))
                x = x + self.dropout(self.ff(self.ln2(x)))
                return x

        class LM(nn.Module):
            def __init__(self, vocab_size: int, cfg: ScratchTrainingConfig):
                super().__init__()
                self.token_emb = nn.Embedding(vocab_size, cfg.d_model)
                self.pos_emb = nn.Embedding(cfg.context_size, cfg.d_model)
                self.blocks = nn.ModuleList([
                    Block(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
                    for _ in range(cfg.n_layers)
                ])
                self.ln = nn.LayerNorm(cfg.d_model)
                self.head = nn.Linear(cfg.d_model, vocab_size, bias=False)

            def forward(self, idx, targets=None):
                b, t = idx.shape
                pos = torch.arange(t, device=idx.device)
                x = self.token_emb(idx) + self.pos_emb(pos)[None, :, :]
                for block in self.blocks:
                    x = block(x)
                x = self.ln(x)
                logits = self.head(x)
                loss = None
                if targets is not None:
                    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                return logits, loss

        return LM(vocab_size, self.config)

    def train(self, token_ids: list[int]) -> list[float]:
        torch, nn, _ = self.torch, self.nn, self.torch.nn.functional
        if len(token_ids) <= self.config.context_size:
            raise ValueError("corpus too small for selected context size")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        losses: list[float] = []
        self.model.train()
        for _ in range(self.config.steps):
            starts = [random.randint(0, len(token_ids) - self.config.context_size - 1) for _ in range(self.config.batch_size)]
            x = torch.tensor([token_ids[s : s + self.config.context_size] for s in starts], dtype=torch.long)
            y = torch.tensor([token_ids[s + 1 : s + self.config.context_size + 1] for s in starts], dtype=torch.long)
            _, loss = self.model(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        return losses

    def save(self, path: str | Path) -> None:
        torch, nn, _ = self.torch, self.nn, self.torch.nn.functional
        torch.save({"state_dict": self.model.state_dict(), "config": self.config.__dict__}, path)


@dataclass
class TrainingReport:
    mode: str
    vocab_size: int
    final_loss: float | None = None
    message: str = ""


def train_from_text(corpus_path: str, steps: int, context: int, d_model: int) -> dict:
    text = Path(corpus_path).read_text()
    tokenizer = CharacterTokenizer(text)
    if tokenizer.vocab_size < 2:
        return {"loss": None, "message": "corpus too small"}
    try:
        model = TransformerLM(tokenizer.vocab_size, ScratchTrainingConfig(context_size=context, d_model=d_model, steps=steps))
        losses = model.train(tokenizer.encode(text))
        return {
            "mode": "transformer",
            "vocab_size": tokenizer.vocab_size,
            "final_loss": sum(losses[-10:]) / min(len(losses), 10),
        }
    except TorchUnavailableError:
        model = TinyCausalLM(tokenizer.vocab_size)
        token_ids = tokenizer.encode(text)
        if len(token_ids) < 2:
            return {"loss": None, "message": "corpus too small"}
        losses = []
        for step in range(steps):
            idx = step % (len(token_ids) - 1)
            loss = model.train_step(token_ids[idx], token_ids[idx + 1])
            losses.append(loss)
        return {
            "mode": "toy",
            "vocab_size": tokenizer.vocab_size,
            "final_loss": sum(losses) / len(losses),
        }
