"""Train-only character vocabulary with special sequence tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.utils import load_json, save_json

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
EOS_TOKEN = "<EOS>"
SPECIAL_TOKENS = (PAD_TOKEN, UNK_TOKEN, EOS_TOKEN)


@dataclass
class CharacterVocabulary:
    """Bidirectional mapping for characters and special tokens."""

    char_to_idx: dict[str, int]
    idx_to_char: dict[int, str]

    @classmethod
    def build(cls, texts: Iterable[str]) -> "CharacterVocabulary":
        characters = sorted({character for text in texts for character in text})
        tokens = [*SPECIAL_TOKENS, *characters]
        char_to_idx = {token: index for index, token in enumerate(tokens)}
        return cls(char_to_idx=char_to_idx, idx_to_char=dict(enumerate(tokens)))

    @property
    def size(self) -> int:
        return len(self.char_to_idx)

    @property
    def pad_idx(self) -> int:
        return self.char_to_idx[PAD_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.char_to_idx[UNK_TOKEN]

    @property
    def eos_idx(self) -> int:
        return self.char_to_idx[EOS_TOKEN]

    def encode(self, text: str, append_eos: bool = False) -> list[int]:
        encoded = [self.char_to_idx.get(character, self.unk_idx) for character in text]
        if append_eos:
            encoded.append(self.eos_idx)
        return encoded

    def decode(self, token_ids: Iterable[int], skip_special: bool = True) -> str:
        decoded: list[str] = []
        for token_id in token_ids:
            token = self.idx_to_char[int(token_id)]
            if token == EOS_TOKEN:
                break
            if skip_special and token in (PAD_TOKEN, UNK_TOKEN):
                continue
            decoded.append(token)
        return "".join(decoded)

    def save(self, path: str) -> None:
        save_json({"tokens": [self.idx_to_char[index] for index in range(self.size)]}, path)

    @classmethod
    def load(cls, path: str) -> "CharacterVocabulary":
        data = load_json(path)
        tokens = data["tokens"]
        char_to_idx = {token: index for index, token in enumerate(tokens)}
        if tuple(tokens[: len(SPECIAL_TOKENS)]) != SPECIAL_TOKENS:
            raise ValueError("Fișierul vocabularului nu conține tokenurile speciale așteptate.")
        return cls(char_to_idx=char_to_idx, idx_to_char=dict(enumerate(tokens)))