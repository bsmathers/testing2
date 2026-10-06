"""CPU-safe sliding-window attention for Tauros collectors.

AMAGO's FlashAttention inference path is CUDA-only.  CPU collectors use this
stateless drop-in replacement with the same causal 32-left / 0-right attention
window and KV-cache interface.  It adds no parameters, so policy checkpoints
remain architecture-compatible.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from amago.nets import transformer


class CPUSlidingWindowAttention(transformer.SelfAttention):
    def __init__(
        self,
        causal: bool,
        dropout: float,
        window_size: tuple[int, int] = (32, 0),
    ):
        super().__init__(causal=causal, dropout=dropout)
        self.dropout_layer = nn.Dropout(dropout)
        self.window_size = tuple(int(x) for x in window_size)
        if self.window_size[1] != 0:
            raise ValueError("CPU collector attention expects right window = 0")

    def _mask(self, q_len: int, kv_len: int, device: torch.device) -> torch.Tensor:
        q = torch.arange(q_len, device=device)[:, None]
        k = torch.arange(kv_len, device=device)[None, :]
        left, right = self.window_size
        allowed = (k >= q - left) & (k <= q + right)
        if self.causal:
            allowed &= k <= q
        return ~allowed

    def _forward_no_cache(self, qkv: torch.Tensor) -> torch.Tensor:
        q, k, v = torch.unbind(qkv, dim=2)
        _, l, _, e = q.shape
        scale = 1.0 / math.sqrt(e)
        scores = scale * torch.einsum("blhe,bshe->bhls", q, k)
        mask = self._mask(l, l, q.device)
        scores.masked_fill_(mask[None, None, :, :], -torch.inf)
        attn = self.dropout_layer(torch.softmax(scores, dim=-1))
        return torch.einsum("bhls,bshd->blhd", attn, v)

    def _forward_cache(
        self,
        qkv: torch.Tensor,
        key_cache: torch.Tensor,
        val_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        q, k, v = torch.unbind(qkv, dim=2)
        b, l, _, e = q.shape
        assert l == 1

        cache_idxs = torch.arange(b, device=key_cache.device)
        key_cache[cache_idxs, cache_seqlens] = k[:, 0]
        val_cache[cache_idxs, cache_seqlens] = v[:, 0]

        end = cache_seqlens + 1
        max_len = int(end.max().item())
        keys = torch.nan_to_num(key_cache[:, :max_len])
        vals = torch.nan_to_num(val_cache[:, :max_len])

        scale = 1.0 / math.sqrt(e)
        scores = scale * torch.einsum("blhe,bshe->bhs", q, keys)

        positions = torch.arange(max_len, device=cache_seqlens.device)[None, :]
        left = self.window_size[0]
        start = torch.clamp(end - 1 - left, min=0)
        allowed = (positions >= start[:, None]) & (positions < end[:, None])
        scores.masked_fill_(~allowed[:, None, :], -torch.inf)

        attn = self.dropout_layer(torch.softmax(scores, dim=-1))
        out = torch.einsum("bhs,bshd->bhd", attn, vals).unsqueeze(1)
        return out

    @torch.compiler.disable
    def forward(self, qkv, key_cache=None, val_cache=None, cache_seqlens=None):
        if key_cache is None or val_cache is None or cache_seqlens is None:
            return self._forward_no_cache(qkv)
        if self.training:
            raise RuntimeError("KV-cache inference requires eval mode")
        return self._forward_cache(qkv, key_cache, val_cache, cache_seqlens)
