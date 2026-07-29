from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 50_304
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = False


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        if cfg.n_embd % cfg.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.out_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self.has_sdpa = hasattr(F, "scaled_dot_product_attention")
        if not self.has_sdpa:
            mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
            self.register_buffer(
                "causal_mask",
                mask.view(1, 1, cfg.block_size, cfg.block_size),
                persistent=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels = x.shape
        head_size = channels // self.n_head
        q = self.q_proj(x).view(
            batch_size, sequence_length, self.n_head, head_size
        ).transpose(1, 2)
        k = self.k_proj(x).view(
            batch_size, sequence_length, self.n_head, head_size
        ).transpose(1, 2)
        v = self.v_proj(x).view(
            batch_size, sequence_length, self.n_head, head_size
        ).transpose(1, 2)

        if self.has_sdpa:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            attention = (q @ k.transpose(-2, -1)) / math.sqrt(head_size)
            attention = attention.masked_fill(
                self.causal_mask[:, :, :sequence_length, :sequence_length] == 0,
                float("-inf"),
            )
            attention = F.softmax(attention, dim=-1)
            attention = F.dropout(
                attention,
                p=self.dropout,
                training=self.training,
            )
            y = attention @ v

        y = y.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, channels
        )
        return self.resid_dropout(self.out_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x), approximate="tanh")))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)
        residual_std = 0.02 / math.sqrt(2 * cfg.n_layer)
        for name, parameter in self.named_parameters():
            if name.endswith("attn.out_proj.weight") or name.endswith(
                "mlp.proj.weight"
            ):
                nn.init.normal_(parameter, mean=0.0, std=residual_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, *, exclude_position_embeddings: bool = False) -> int:
        count = sum(parameter.numel() for parameter in self.parameters())
        if exclude_position_embeddings:
            count -= self.position_embedding.weight.numel()
        return count

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, sequence_length = idx.shape
        if sequence_length > self.cfg.block_size:
            raise ValueError(
                f"sequence length {sequence_length} exceeds block_size "
                f"{self.cfg.block_size}"
            )
        positions = torch.arange(sequence_length, device=idx.device)
        x = self.drop(
            self.token_embedding(idx) + self.position_embedding(positions)
        )
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss

    def spectral_matrices(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield only transformer block matrices used for alpha tracking."""

        for index, block in enumerate(self.blocks):
            prefix = f"L{index:02d}"
            yield f"{prefix}_W_Q", block.attn.q_proj.weight
            yield f"{prefix}_W_K", block.attn.k_proj.weight
            yield f"{prefix}_W_V", block.attn.v_proj.weight
            yield f"{prefix}_W_O", block.attn.out_proj.weight
            yield f"{prefix}_W_MLP_IN", block.mlp.fc.weight
            yield f"{prefix}_W_MLP_OUT", block.mlp.proj.weight
