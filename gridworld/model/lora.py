import torch
import torch.nn as nn

from peft.tuners.lora import LoraLinear


class LoRALinear(LoraLinear):
    """Thin wrapper around ``peft.tuners.lora.LoraLinear``.

    This project previously shipped a lightweight fallback implementation when
    ``peft`` was not installed. To simplify the codebase and ensure consistent
    behavior, the fallback has been removed and ``peft`` is now required.
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

