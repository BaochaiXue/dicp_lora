from dataclasses import dataclass
from typing import Literal, Optional, Type

import torch


@dataclass
class Config:
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    rotary_percentage: float
    parallel_residual: bool
    bias: bool
    dropout: float
    attention_dropout: float

    shared_attention_norm: bool
    _norm_class: Literal["LayerNorm", "RMSNorm", "FusedRMSNorm"]
    _mlp_class: Literal["GptNeoxMLP", "LLaMAMLP"]
    n_query_groups: Optional[int] = None
    norm_eps: float = 1e-5
    intermediate_size: Optional[int] = None
    condense_ratio: int = 1
    flash_attn: bool = True

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0
        if self.n_query_groups is not None:
            assert self.n_head % self.n_query_groups == 0
        else:
            self.n_query_groups = self.n_head
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.n_embd

    @property
    def head_size(self) -> int:
        return self.n_embd // self.n_head

    @property
    def norm_class(self) -> Type:
        if self._norm_class == "RMSNorm":
            from .rmsnorm import RMSNorm

            return RMSNorm
        elif self._norm_class == "FusedRMSNorm":
            from .rmsnorm import FusedRMSNorm

            return FusedRMSNorm
        return getattr(torch.nn, self._norm_class)
