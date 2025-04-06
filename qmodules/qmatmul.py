import torch, torch.nn as nn, torch.nn.functional as F
from quant_utils import build_quantizer


class QMatMul(nn.Module):
    def __init__(self, x_bit=8, **kwargs):
        super().__init__()
        self.x1_quantizer = build_quantizer(
            dict(
                calib_metric="percent-0.99999",
                dtype=f"uint{x_bit}",
                learnable=True,
                symmetric=False,
            )
        )
        self.x2_quantizer = build_quantizer(
            dict(
                calib_metric="percent-0.99999",
                dtype=f"uint{x_bit}",
                learnable=True,
                symmetric=False,
            )
        )

    def forward(self, x1, x2):
        x1 = self.x1_quantizer(x1)
        x2 = self.x2_quantizer(x2)
        return torch.matmul(x1, x2)
