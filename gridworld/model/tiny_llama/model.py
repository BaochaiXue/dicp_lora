import math
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
# from flash_attn import flash_attn_func
from lightning_utilities.core.imports import RequirementCache
from xformers.ops import SwiGLU

from .config import Config
from .fused_rotary_embedding import apply_rotary_emb_func
from ..lora import LoRALinear

from torch.nn.attention import sdpa_kernel, SDPBackend

RoPECache = Tuple[torch.Tensor, torch.Tensor]
KVCache = Tuple[torch.Tensor, torch.Tensor]
FlashAttention2Available = RequirementCache("flash-attn>=2.0.0.post1")


class Transformer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = Config(
            block_size=3*config['n_transit'],
            n_layer=config['tf_n_layer'],
            n_head=config['tf_n_head'],
            n_embd=config['tf_n_embd'],
            bias=True,
            rotary_percentage=1.0,
            parallel_residual=False,
            shared_attention_norm=False,
            _norm_class="FusedRMSNorm",
            _mlp_class="LLaMAMLP",
            dropout=config['tf_dropout'],
            attention_dropout=config['tf_attn_dropout'],
            intermediate_size=config['tf_n_inner'],
            flash_attn=config['flash_attn'],
        )
        self.config.lora_r = config.get('lora_r', 0)
        self.config.lora_alpha = config.get('lora_alpha', 1.0)
        self.config.lora_dropout = config.get('lora_dropout', 0.0)
        self.device = config['device']
        self.blocks = nn.ModuleList(Block(self.config) for _ in range(config['tf_n_layer']))
        self.rope_cache_fp16 = self.build_rope_cache(device=self.device, dtype=torch.float16)
        self.rope_cache_bf16 = self.build_rope_cache(device=self.device, dtype=torch.bfloat16)
        self.rope_cache_fp32 = self.build_rope_cache(device=self.device, dtype=torch.float32)

    def forward(self,
                x: torch.Tensor, 
                max_seq_length: int,
                mask: Optional[torch.Tensor] = None, 
                dtype="bf16") -> Tuple[torch.Tensor, Optional[KVCache]]:

        if dtype == "bf16":
            cos, sin = self.rope_cache_bf16
        elif dtype == "fp16":
            cos, sin = self.rope_cache_fp16
        elif dtype == "fp32":
            cos, sin = self.rope_cache_fp32
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
                
        for block in self.blocks:
            x, *_ = block(x,
                          (cos[:x.size(1)], sin[:x.size(1)]),
                          max_seq_length,
                          mask)
        return x

    def build_rope_cache(self, device, dtype) -> RoPECache:
        return build_rope_cache(
            seq_len=self.config.block_size,
            n_elem=int(self.config.rotary_percentage * self.config.head_size),
            dtype=dtype,
            device=device,
            condense_ratio=self.config.condense_ratio,
            )



class Block(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.norm_1 = config.norm_class(
            config.n_embd, eps=config.norm_eps, dropout=config.dropout
        )
        self.attn = CausalSelfAttention(config)
        if not config.shared_attention_norm:
            self.norm_2 = config.norm_class(
                config.n_embd, eps=config.norm_eps, dropout=config.dropout
            )
        self.mlp = getattr(sys.modules[__name__], config._mlp_class)(config)
        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        max_seq_length: int,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        n_1 = self.norm_1(x)
        h, new_kv_cache = self.attn(
            n_1, rope, max_seq_length, mask, input_pos, kv_cache
        )
        if self.config.parallel_residual:
            n_2 = n_1 if self.config.shared_attention_norm else self.norm_2(x)
            x = x + h + self.mlp(n_2)
        else:
            if self.config.shared_attention_norm:
                raise NotImplementedError(
                    "No checkpoint amongst the ones we support uses this configuration"
                    " (non-parallel residual and shared attention norm)."
                )

            x = x + h
            x = x + self.mlp(self.norm_2(x))
        return x, new_kv_cache


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        if getattr(config, "lora_r", 0) > 0:
            self.attn = LoRALinear(
                config.n_embd,
                shape,
                r=getattr(config, "lora_r", 0),
                lora_alpha=getattr(config, "lora_alpha", 1.0),
                lora_dropout=getattr(config, "lora_dropout", 0.0),
                bias=config.bias,
            )
            self.proj = LoRALinear(
                config.n_embd,
                config.n_embd,
                r=getattr(config, "lora_r", 0),
                lora_alpha=getattr(config, "lora_alpha", 1.0),
                lora_dropout=getattr(config, "lora_dropout", 0.0),
                bias=config.bias,
            )
        else:
            self.attn = nn.Linear(config.n_embd, shape, bias=config.bias)
            self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        max_seq_length: int,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        (
            B,
            T,
            C,
        ) = x.size()

        qkv = self.attn(x)

        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2
        qkv = qkv.view(
            B, T, self.config.n_query_groups, total_qkv, self.config.head_size
        )

        q, k, v = qkv.split((q_per_kv, 1, 1), dim=-2)

        q = q.reshape(B, T, -1, self.config.head_size)  # (B, T, nh_q, hs)
        k = k.reshape(B, T, -1, self.config.head_size)
        v = v.reshape(B, T, -1, self.config.head_size)

        cos, sin = rope
        q = apply_rotary_emb_func(q, cos, sin, False, True)
        k = apply_rotary_emb_func(k, cos, sin, False, True)

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            cache_k, cache_v = cache_k.to(dtype=k.dtype), cache_v.to(dtype=v.dtype)
            if input_pos[-1] >= max_seq_length:
                input_pos = torch.tensor(max_seq_length - 1, device=input_pos.device)
                cache_k = torch.roll(cache_k, -1, dims=2)
                cache_v = torch.roll(cache_v, -1, dims=2)
            k = cache_k.index_copy_(2, input_pos, k)
            v = cache_v.index_copy_(2, input_pos, v)
            kv_cache = k, v

        y = self.scaled_dot_product_attention(q, k, v, mask=mask)

        y = y.reshape(B, T, C)

        y = self.proj(y)

        return y, kv_cache

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        scale = 1.0 / math.sqrt(self.config.head_size)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if q.size() != k.size():
            k = k.repeat_interleave(q.shape[1]//k.shape[1], dim=1)
            v = v.repeat_interleave(q.shape[1]//v.shape[1], dim=1)
        
        if (
            FlashAttention2Available
            and mask is None
            and q.device.type == "cuda"
            and q.dtype in (torch.float16, torch.bfloat16)
            and self.config.flash_attn
        ):
            attn_type = SDPBackend.FLASH_ATTENTION
        else:
            attn_type = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
            
        with sdpa_kernel(attn_type):
                y = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, 
                    attn_mask=mask, 
                    dropout_p=self.config.attention_dropout if self.training else 0.0, 
                    scale=scale,
                    is_causal=mask is None
                )
                
        
        return y.transpose(1, 2)


class GptNeoxMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = torch.nn.functional.gelu(x)
        return self.proj(x)


class LLaMAMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.swiglu = SwiGLU(
            config.n_embd, config.intermediate_size, bias=False, _pack_weights=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swiglu(x)


def build_rope_cache(
    seq_len: int,
    n_elem: int,
    dtype: torch.dtype,
    device: torch.device,
    base: int = 10000,
    condense_ratio: int = 1,
) -> RoPECache:
    
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device) / n_elem))

    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    idx_theta = torch.outer(seq_idx, theta)

    cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)

    if dtype == torch.bfloat16:
        return cos.bfloat16(), sin.bfloat16()
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        return cos.half(), sin.half()
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
    x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    roped = (x * cos) + (rotated * sin)
    return roped.type_as(x)
