import torch
import torch.nn as nn

from peft.tuners.lora import LoraLinear


class LoRALinear(LoraLinear):
    """Thin wrapper around ``peft.tuners.lora.LoraLinear``.

    The previous version of this repository provided a minimal fallback
    implementation when ``peft`` was unavailable. The fallback added
    complexity and behaved differently from the official implementation, so it
    has been removed in favor of always depending on ``peft``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
        )

