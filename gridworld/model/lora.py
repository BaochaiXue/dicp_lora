import torch
import torch.nn as nn

"""Utility wrapper around the LoRA linear layer.

The upstream ``peft`` library has gone through a few API revisions.  Some
versions expose the LoRA-augmented linear layer under
``peft.tuners.lora.LoraLinear`` while newer releases simply call it
``Linear``.  Older environments might even lack ``peft`` entirely.  The
training scripts in this repository expect a class compatible with the
``LoraLinear`` implementation, so we attempt to import it from ``peft`` and
fall back to a lightweight local implementation if necessary.
"""

try:  # PEFT >= 0.4
    from peft.tuners.lora import LoraLinear as _PeftLoraLinear
except Exception:  # PEFT >= 0.7 uses ``Linear``
    try:
        from peft.tuners.lora import Linear as _PeftLoraLinear
    except Exception:  # PEFT not available -> use simple fallback
        _PeftLoraLinear = None

import math


if _PeftLoraLinear is None:
    class _PeftLoraLinear(nn.Linear):
        """Minimal LoRA linear layer used as a fallback when ``peft`` is missing.

        This implementation supports the ``LoraLinear`` API used in the
        repository. It simply injects a low-rank update ``BA`` into the base
        linear layer while keeping the original weights frozen.
        """

        def __init__(self, in_features, out_features, r=0, lora_alpha=1.0, lora_dropout=0.0, bias=True, **kwargs):
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
    """Compatibility wrapper used throughout the codebase."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
        bias: bool = True,
        adapter_name: str = "default",
    ) -> None:
        # ``peft`` has changed the ``LoraLinear`` constructor multiple times
        # so we inspect the parent signature to see whether ``adapter_name`` is
        # expected.  Passing it conditionally avoids ``TypeError`` on older
        # releases while still working on 0.11+.
        import inspect

        parent_init = super().__init__
        params = inspect.signature(parent_init).parameters

        kwargs = dict(
            in_features=in_features,
            out_features=out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
        )
        if "adapter_name" in params:
            kwargs["adapter_name"] = adapter_name

        parent_init(**kwargs)

