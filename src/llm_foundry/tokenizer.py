from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol
import json


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: Iterable[int]) -> str: ...

    @property
    def vocab_size(self) -> int: ...

    def save(self, path: str | Path) -> Path: ...


@dataclass
class ByteTokenizer:
    lowercase: bool = False
    add_bos: bool = True
    add_eos: bool = True

    PAD = 0
    BOS = 1
    EOS = 2
    UNK = 3
    OFFSET = 4

    @property
    def vocab_size(self) -> int:
        return self.OFFSET + 256

    def encode(self, text: str) -> list[int]:
        source = text.lower() if self.lowercase else text
        ids = [byte + self.OFFSET for byte in source.encode("utf-8")]
        if self.add_bos:
            ids.insert(0, self.BOS)
        if self.add_eos:
            ids.append(self.EOS)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        raw = bytearray()
        for token_id in ids:
            if token_id < self.OFFSET:
                continue
            raw.append(token_id - self.OFFSET)
        return raw.decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"kind": "byte", "lowercase": self.lowercase, "add_bos": self.add_bos, "add_eos": self.add_eos},
                indent=2,
                ensure_ascii=False,
            )
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ByteTokenizer":
        data = json.loads(Path(path).read_text())
        return cls(
            lowercase=bool(data.get("lowercase", False)),
            add_bos=bool(data.get("add_bos", True)),
            add_eos=bool(data.get("add_eos", True)),
        )


@dataclass
class CharacterTokenizer:
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]
    lowercase: bool = False
    add_bos: bool = True
    add_eos: bool = True

    PAD = 0
    BOS = 1
    EOS = 2
    UNK = 3
    OFFSET = 4

    @classmethod
    def from_text(cls, text: str, lowercase: bool = False, add_bos: bool = True, add_eos: bool = True) -> "CharacterTokenizer":
        source = text.lower() if lowercase else text
        chars = sorted(set(source))
        token_to_id = {ch: i + cls.OFFSET for i, ch in enumerate(chars)}
        id_to_token = {i + cls.OFFSET: ch for i, ch in enumerate(chars)}
        return cls(token_to_id=token_to_id, id_to_token=id_to_token, lowercase=lowercase, add_bos=add_bos, add_eos=add_eos)

    @property
    def vocab_size(self) -> int:
        return self.OFFSET + len(self.token_to_id)

    def encode(self, text: str) -> list[int]:
        source = text.lower() if self.lowercase else text
        ids = [self.token_to_id.get(ch, self.UNK) for ch in source]
        if self.add_bos:
            ids.insert(0, self.BOS)
        if self.add_eos:
            ids.append(self.EOS)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.id_to_token.get(token_id, "") for token_id in ids if token_id >= self.OFFSET)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "kind": "character",
                    "token_to_id": self.token_to_id,
                    "lowercase": self.lowercase,
                    "add_bos": self.add_bos,
                    "add_eos": self.add_eos,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "CharacterTokenizer":
        data = json.loads(Path(path).read_text())
        token_to_id = {str(key): int(value) for key, value in data["token_to_id"].items()}
        id_to_token = {int(value): str(key) for key, value in token_to_id.items()}
        return cls(
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            lowercase=bool(data.get("lowercase", False)),
            add_bos=bool(data.get("add_bos", True)),
            add_eos=bool(data.get("add_eos", True)),
        )


@dataclass
class HuggingFaceTokenizer:
    model_name: str
    add_bos: bool = True
    add_eos: bool = True

    def __post_init__(self) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install the 'local' extras to use Hugging Face tokenizers") from exc
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    @property
    def vocab_size(self) -> int:
        return int(getattr(self._tokenizer, "vocab_size", 0))

    def encode(self, text: str) -> list[int]:
        ids = self._tokenizer.encode(text, add_special_tokens=self.add_bos or self.add_eos)
        return list(ids)

    def decode(self, ids: Iterable[int]) -> str:
        return self._tokenizer.decode(list(ids), skip_special_tokens=True)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"kind": "huggingface", "model_name": self.model_name}, indent=2, ensure_ascii=False))
        return path


@dataclass
class TokenizerConfigView:
    kind: str = "byte"
    model_name: str | None = None
    vocab_path: str | None = None
    lowercase: bool = False
    add_bos: bool = True
    add_eos: bool = True


def build_tokenizer(config: TokenizerConfigView, corpus_text: str | None = None) -> Tokenizer:
    if config.kind == "byte":
        return ByteTokenizer(lowercase=config.lowercase, add_bos=config.add_bos, add_eos=config.add_eos)
    if config.kind == "character":
        if not corpus_text:
            raise ValueError("corpus_text is required to build a character tokenizer")
        return CharacterTokenizer.from_text(corpus_text, lowercase=config.lowercase, add_bos=config.add_bos, add_eos=config.add_eos)
    if config.kind == "huggingface":
        if not config.model_name:
            raise ValueError("model_name is required for a huggingface tokenizer")
        return HuggingFaceTokenizer(config.model_name, add_bos=config.add_bos, add_eos=config.add_eos)
    if config.kind == "saved-byte":
        if not config.vocab_path:
            raise ValueError("vocab_path is required for saved-byte tokenizer")
        return ByteTokenizer.load(config.vocab_path)
    if config.kind == "saved-character":
        if not config.vocab_path:
            raise ValueError("vocab_path is required for saved-character tokenizer")
        return CharacterTokenizer.load(config.vocab_path)
    raise ValueError(f"Unknown tokenizer kind: {config.kind}")


def estimate_token_count(text: str) -> int:
    return max(1, len(text.encode("utf-8")) // 4)
