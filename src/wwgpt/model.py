from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from wwgpt.config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.n_embd % cfg.n_head:
            raise ValueError("n_embd must divide n_head")
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.key = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.linear_bias)
        self.query = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.linear_bias)
        self.value = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.linear_bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.linear_bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q = self.query(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = self.key(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
            scale=self.head_dim ** -0.5,
        )
        return self.resid_dropout(self.proj(y.transpose(1, 2).contiguous().view(b, t, c)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.layernorm_bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.layernorm_bias)
        act: nn.Module = nn.GELU() if cfg.activation == "gelu" else nn.ReLU()
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, cfg.mlp_mult * cfg.n_embd, bias=cfg.linear_bias), act,
            nn.Linear(cfg.mlp_mult * cfg.n_embd, cfg.n_embd, bias=cfg.linear_bias), nn.Dropout(cfg.dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


@dataclass(frozen=True)
class ParameterReport:
    total_parameters: int
    trainable_parameters: int
    token_embedding_parameters: int
    position_embedding_parameters: int
    output_head_parameters: int
    embedding_parameters: int
    non_embedding_parameters: int
    attention_heads: int
    transformer_blocks: int
    context_length: int
    vocabulary_size: int


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.layernorm_bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.wte.weight
        if cfg.init_mode == "nanogpt_normal_0p02":
            self.apply(self._init_weights)
            self._init_residual_projections()
        elif cfg.init_mode != "pytorch_default":
            raise ValueError(f"unknown init_mode: {cfg.init_mode}")

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _init_residual_projections(self) -> None:
        std = 0.02 / (2 * self.cfg.n_layer) ** 0.5
        for name, module in self.named_modules():
            if name.endswith(("attn.proj", "mlp.2")) and isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        _, t = idx.shape
        if t > self.cfg.block_size:
            raise ValueError("sequence length exceeds block_size")
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def parameter_report(self) -> ParameterReport:
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        tok = self.wte.weight.numel()
        pos = self.wpe.weight.numel()
        head = self.lm_head.weight.numel()
        emb = tok + pos
        return ParameterReport(total, train, tok, pos, head, emb, total - emb, self.cfg.n_head, self.cfg.n_layer, self.cfg.block_size, self.cfg.vocab_size)

    def report_dict(self) -> dict[str, int]:
        return asdict(self.parameter_report())
