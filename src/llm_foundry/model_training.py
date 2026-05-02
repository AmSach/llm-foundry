from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
import json
from typing import Iterable

from .config import ModelConfig, TokenizerConfig
from .tokenizer import TokenizerConfigView, build_tokenizer
from .training import ScratchTrainingConfig, TinyCausalLM, TorchUnavailableError, TransformerLM


@dataclass
class TrainingRun:
    mode: str
    corpus_path: str
    tokenizer_kind: str
    vocab_size: int
    token_count: int
    config: dict
    losses: list[float] = field(default_factory=list)
    final_loss: float | None = None
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        return path


@dataclass
class ModelBundle:
    config: ModelConfig
    training: TrainingRun
    tokenizer_path: str = ""
    checkpoint_path: str = ""

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "training": self.training.to_dict(),
            "tokenizer_path": self.tokenizer_path,
            "checkpoint_path": self.checkpoint_path,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        return path


@dataclass
class TokenStatistics:
    characters: int
    bytes: int
    estimated_tokens: int
    tokens: int
    vocab_size: int


class ModelTrainer:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def prepare_tokenizer(self, corpus_text: str):
        tokenizer_config = TokenizerConfigView(
            kind=self.config.tokenizer.kind,
            model_name=self.config.tokenizer.model_name,
            vocab_path=self.config.tokenizer.vocab_path,
            lowercase=self.config.tokenizer.lowercase,
            add_bos=self.config.tokenizer.add_bos,
            add_eos=self.config.tokenizer.add_eos,
        )
        if tokenizer_config.kind == "character":
            return build_tokenizer(tokenizer_config, corpus_text=corpus_text)
        return build_tokenizer(tokenizer_config)

    def token_stats(self, corpus_text: str, tokenizer) -> TokenStatistics:
        token_ids = tokenizer.encode(corpus_text)
        return TokenStatistics(
            characters=len(corpus_text),
            bytes=len(corpus_text.encode("utf-8")),
            estimated_tokens=max(1, len(corpus_text.encode("utf-8")) // 4),
            tokens=len(token_ids),
            vocab_size=tokenizer.vocab_size,
        )

    def train(self, corpus_path: str | Path) -> TrainingRun:
        corpus_path = Path(corpus_path)
        corpus_text = corpus_path.read_text()
        tokenizer = self.prepare_tokenizer(corpus_text)
        token_ids = tokenizer.encode(corpus_text)
        stats = self.token_stats(corpus_text, tokenizer)
        if len(token_ids) < 2:
            return TrainingRun(
                mode="empty",
                corpus_path=str(corpus_path),
                tokenizer_kind=self.config.tokenizer.kind,
                vocab_size=tokenizer.vocab_size,
                token_count=len(token_ids),
                config=self.config.to_dict(),
                notes="corpus too small",
            )
        try:
            context = min(max(2, self.config.context_length // 2), max(2, len(token_ids) - 1))
            scratch_cfg = ScratchTrainingConfig(
                context_size=context,
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
                n_layers=self.config.n_layers,
                d_ff=self.config.d_ff,
                dropout=self.config.dropout,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                steps=self.config.training_steps or 20,
            )
            model = TransformerLM(tokenizer.vocab_size, scratch_cfg)
            losses = model.train(token_ids)
            final_loss = sum(losses[-10:]) / min(10, len(losses))
            return TrainingRun(
                mode="transformer",
                corpus_path=str(corpus_path),
                tokenizer_kind=self.config.tokenizer.kind,
                vocab_size=tokenizer.vocab_size,
                token_count=len(token_ids),
                config={**self.config.to_dict(), "token_statistics": asdict(stats)},
                losses=losses,
                final_loss=final_loss,
                notes="trained with local torch backend",
            )
        except TorchUnavailableError:
            model = TinyCausalLM(tokenizer.vocab_size)
            losses: list[float] = []
            for step in range(self.config.training_steps or 20):
                idx = step % (len(token_ids) - 1)
                losses.append(model.train_step(token_ids[idx], token_ids[idx + 1]))
            final_loss = sum(losses[-10:]) / min(10, len(losses))
            return TrainingRun(
                mode="toy",
                corpus_path=str(corpus_path),
                tokenizer_kind=self.config.tokenizer.kind,
                vocab_size=tokenizer.vocab_size,
                token_count=len(token_ids),
                config={**self.config.to_dict(), "token_statistics": asdict(stats)},
                losses=losses,
                final_loss=final_loss,
                notes="torch unavailable so the toy fallback was used",
            )


def train_model_from_corpus(corpus_path: str | Path, config: ModelConfig) -> TrainingRun:
    return ModelTrainer(config).train(corpus_path)
