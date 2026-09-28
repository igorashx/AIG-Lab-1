"""Configurable character-level LSTM language model."""

from __future__ import annotations

import torch
from torch import nn


class CharacterLSTM(nn.Module):
    """Embedding, LSTM, and vocabulary projection returning raw logits."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        pad_idx: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self, token_ids: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        embeddings = self.embedding(token_ids)
        states, hidden = self.lstm(embeddings, hidden)
        return self.output(states), hidden