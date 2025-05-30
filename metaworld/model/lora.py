import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from peft.tuners.lora import LoraLinear as PeftLoRALinear
except Exception:  # pragma: no cover - PEFT may be unavailable
    PeftLoRALinear = None

if PeftLoRALinear is not None:
    class LoRALinear(PeftLoRALinear):
        """LoRA linear layer using the HuggingFace PEFT implementation."""

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
else:
    class LoRALinear(nn.Linear):
        """Fallback LoRA linear layer if PEFT is unavailable."""

        def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: float = 1.0,
            lora_dropout: float = 0.0,
            bias: bool = True,
        ) -> None:
            super().__init__(in_features, out_features, bias=bias)
            self.r = r
            if r > 0:
                self.lora_A = nn.Parameter(torch.zeros(r, in_features))
                self.lora_B = nn.Parameter(torch.zeros(out_features, r))
                self.scaling = lora_alpha / r
                self.dropout = nn.Dropout(p=lora_dropout)
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B)
                self.weight.requires_grad = False
                if self.bias is not None:
                    self.bias.requires_grad = False
            else:
                self.lora_A = None
                self.lora_B = None
                self.scaling = None
                self.dropout = nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            result = F.linear(x, self.weight, self.bias)
            if self.r > 0:
                lora = self.dropout(x) @ self.lora_A.t()
                lora = lora @ self.lora_B.t()
                result = result + lora * self.scaling
            return result
