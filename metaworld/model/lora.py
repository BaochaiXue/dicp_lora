import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Linear):
    def __init__(self, in_features, out_features, r=0, lora_alpha=1.0, lora_dropout=0.0, bias=True):
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
