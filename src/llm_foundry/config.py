from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json


@dataclass
class TokenizerConfig:
    kind: str = "byte"
    model_name: str | None = None
    vocab_path: str | None = None
    lowercase: bool = False
    add_bos: bool = True
    add_eos: bool = True


@dataclass
class ModelConfig:
    name: str = "llm-foundry"
    backend: str = "openai-compatible"
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    context_length: int = 8192
    max_output_tokens: int = 1024
    d_model: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    d_ff: int = 11008
    dropout: float = 0.0
    attention_dropout: float = 0.0
    vocab_size: int = 0
    use_memory: bool = True
    memory_store_dir: str = "memory"
    compression_target_tokens: int = 512
    use_tools: bool = True
    max_tool_calls: int = 8
    use_reasoning_cascade: bool = True
    use_self_reflection: bool = True
    training_steps: int = 0
    batch_size: int = 8
    learning_rate: float = 3e-4
    seed: int = 42
    notes: str = ""

    def estimated_parameters(self) -> int:
        embed = self.vocab_size * self.d_model
        position = self.context_length * self.d_model
        per_block = 4 * self.d_model * self.d_model + 2 * self.d_model * self.d_ff + 4 * self.d_model
        output = self.d_model * self.vocab_size
        return embed + position + output + self.n_layers * per_block

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        tokenizer = TokenizerConfig(**(data.get("tokenizer") or {}))
        payload = dict(data)
        payload["tokenizer"] = tokenizer
        return cls(**payload)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass
class TrainingConfig:
    steps: int = 100
    batch_size: int = 8
    learning_rate: float = 3e-4
    device: str = "cpu"
    gradient_clip_norm: float = 1.0
    warmup_steps: int = 0


@dataclass
class RuntimeConfig:
    use_memory: bool = True
    use_tools: bool = True
    use_reasoning: bool = True
    context_budget_tokens: int = 4096
    max_tool_calls: int = 8
    memory_notes_top_k: int = 6


@dataclass
class BenchmarkConfig:
    backend: str = "echo"
    model: str | None = None
    output_dir: str = "reports"
    include_reasoning: bool = True
    include_coding: bool = True
    include_tool_use: bool = True
    include_memory: bool = True
