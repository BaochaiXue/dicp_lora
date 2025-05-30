import torch
import torch.nn as nn

"""Utility wrapper around the LoRA linear layer.

This repository originally relied on ``peft.tuners.lora.LoraLinear``.  Recent
``peft`` releases renamed the class to ``Linear`` and some environments might
not provide ``peft`` at all.  We therefore try to import whichever variant is
available and fall back to a minimal implementation if none is found.
"""

try:
    from peft.tuners.lora import LoraLinear as _PeftLoraLinear
except Exception:
    try:
        from peft.tuners.lora import Linear as _PeftLoraLinear
    except Exception:
        _PeftLoraLinear = None

import math


if _PeftLoraLinear is None:
    class _PeftLoraLinear(nn.Linear):
        """Minimal implementation used when ``peft`` is missing."""

        def __init__(self, in_features, out_features, r=0, lora_alpha=1.0, lora_dropout=0.0, bias=True):
            super().__init__(in_features, out_features, bias=bias)
            self.r = r
            if r > 0:
                self.lora_A = nn.Linear(in_features, r, bias=False)
                self.lora_B = nn.Linear(r, out_features, bias=False)
                self.scaling = lora_alpha / r
                self.lora_dropout = nn.Dropout(p=lora_dropout)
                nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B.weight)
                self.weight.requires_grad = False
            else:
                self.lora_A = None
                self.lora_B = None
                self.scaling = 0.0
                self.lora_dropout = nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            result = super().forward(x)
            if self.r > 0:
                update = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
                result = result + update
            return result


class LoRALinear(_PeftLoraLinear):
    """Compatibility wrapper used throughout the repository."""

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

