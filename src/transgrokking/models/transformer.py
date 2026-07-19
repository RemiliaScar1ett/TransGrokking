"""A small, explicit, hook-friendly causal Transformer."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F

CacheFilter = Callable[[str], bool] | set[str] | list[str] | tuple[str, ...] | None


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return attention output `[B,S,D]` and projected head outputs `[B,S,H,D]`."""
        batch, seq_len, d_model = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.n_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query, key, value = (tensor.transpose(1, 2) for tensor in (query, key, value))
        scores = query @ key.transpose(-2, -1) / self.head_dim**0.5
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
        pattern = F.softmax(scores, dim=-1)
        pattern = F.dropout(pattern, self.dropout, self.training)
        raw_heads = pattern @ value
        weight = self.output.weight.view(d_model, self.n_heads, self.head_dim)
        head_output = torch.einsum("bhsd,ohd->bsho", raw_heads, weight)
        output = head_output.sum(dim=2)
        if self.output.bias is not None:
            output = output + self.output.bias
        return output, head_output


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_mlp: int, dropout: float, activation: str):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp_in = nn.Linear(d_model, d_mlp)
        self.mlp_out = nn.Linear(d_mlp, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu


class TransparentTransformer(nn.Module):
    """Causal Transformer mapping tokens `[B,S]` to logits `[B,S,V]`."""

    def __init__(
        self,
        vocab_size: int,
        sequence_length: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_mlp: int,
        dropout: float,
        activation: str,
        norm_first: bool,
    ) -> None:
        super().__init__()
        if not norm_first:
            raise ValueError("M0 transparent blocks currently require norm_first=true")
        self.sequence_length = sequence_length
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(sequence_length, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_mlp, dropout, activation)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.unembedding = nn.Linear(d_model, vocab_size)

    @staticmethod
    def _wanted(names_filter: CacheFilter, name: str) -> bool:
        if names_filter is None:
            return True
        if callable(names_filter):
            return bool(names_filter(name))
        return name in names_filter

    def _run(
        self, tokens: torch.Tensor, names_filter: CacheFilter
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if tokens.ndim != 2 or tokens.shape[1] > self.sequence_length:
            raise ValueError(f"tokens must have shape [batch, seq<= {self.sequence_length}]")
        cache: dict[str, torch.Tensor] = {}

        def save(name: str, value: torch.Tensor) -> None:
            if self._wanted(names_filter, name):
                cache[name] = value

        positions = torch.arange(tokens.shape[1], device=tokens.device)
        token_embed = self.token_embedding(tokens)
        position_embed = self.position_embedding(positions).unsqueeze(0)
        save("embed.token", token_embed)
        save("embed.position", position_embed)
        residual = token_embed + position_embed
        save("residual.pre", residual)
        for index, block in enumerate(self.blocks):
            attention_output, heads = block.attention(block.ln1(residual))
            save(f"blocks.{index}.attention.head_output", heads)
            residual = residual + block.dropout(attention_output)
            save(f"blocks.{index}.residual.mid", residual)
            mlp_pre = block.mlp_in(block.ln2(residual))
            mlp_post = block.activation(mlp_pre)
            save(f"blocks.{index}.mlp.pre", mlp_pre)
            save(f"blocks.{index}.mlp.post", mlp_post)
            residual = residual + block.dropout(block.mlp_out(mlp_post))
            save(f"blocks.{index}.residual.post", residual)
        normalized = self.final_norm(residual)
        save("residual.final", normalized)
        logits = self.unembedding(normalized)
        save("logits", logits)
        return logits, cache

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return logits `[batch, sequence, vocab]` without retaining activations."""
        return self._run(tokens, set())[0]

    def run_with_cache(
        self, tokens: torch.Tensor, names_filter: CacheFilter = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return logits and selected named activations with documented stable names."""
        return self._run(tokens, names_filter)
