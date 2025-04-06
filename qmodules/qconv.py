from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .qlinear import QLinear, QMZBLinear

import torch.autograd as ag


from quant_utils import build_quantizer

eps = torch.finfo(torch.float32).eps * 10


class STERound(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
    ):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def fake_quantize(x, scale, zp=None, bitwidth=8, sim=True):
    if zp is None:
        x_sim = STERound.apply(x / scale).clamp(
            -(2 ** (bitwidth - 1)), 2 ** (bitwidth - 1) - 1
        )
    else:
        x_sim = STERound.apply(x / scale + zp).clamp(
            -(2 ** (bitwidth - 1)), 2 ** (bitwidth - 1) - 1
        )
    if sim:  # TODO
        if zp is not None:
            x_sim -= zp
        x_sim *= scale
    return x_sim


class QConv(nn.Conv2d):
    w_scale: torch.Tensor
    x_scale: torch.Tensor
    x_zp: torch.Tensor

    @property
    def calibrated(self) -> bool:
        return self.w_scale.abs().sum() > 0

    def __init__(self, raw_conv: nn.Conv2d, w_bit=8, x_bit=8) -> None:
        super().__init__(
            raw_conv.in_channels,
            raw_conv.out_channels,
            raw_conv.kernel_size,
            raw_conv.stride,
            raw_conv.padding,
            raw_conv.dilation,
            raw_conv.groups,
            raw_conv.bias is not None,
            raw_conv.padding_mode,
        )
        device = raw_conv.weight.device
        factory_kwargs = dict(device=device)
        self.weight = nn.Parameter(raw_conv.weight.data)
        if raw_conv.bias is not None:
            self.bias = nn.Parameter(raw_conv.bias.data)
        else:
            self.bias = None
        self.register_buffer("w_scale", torch.zeros([self.out_channels, 1, 1, 1]))
        self.register_buffer("x_scale", torch.zeros(1))
        self.register_buffer("x_zp", torch.zeros(1))
        self.w_bit = w_bit
        self.x_bit = x_bit
        self.w_qmin = -(1 << (w_bit - 1))
        self.w_qmax = (1 << (w_bit - 1)) - 1

        self.x_quantizer = build_quantizer(
            dict(
                calib_metric="percent-0.99999",
                dtype=f"uint{x_bit}",
                learnable=True,
                symmetric=False,
            )
        )

        self._register_load_state_dict_pre_hook(self._pre_load_state_dict_hook)
        self.to(device)

    @torch.no_grad()
    def calibration(self, x):
        neg_scale = (
            torch.amin(self.weight, [1, 2, 3], keepdim=True).clip(max=0) / self.w_qmin
        )
        pos_scale = (
            torch.amax(self.weight, [1, 2, 3], keepdim=True).clip(min=0) / self.w_qmax
        )
        w_scale = torch.max(neg_scale, pos_scale).clip(min=eps)
        self.w_scale.data.copy_(w_scale.detach())

    def _pre_load_state_dict_hook(
        self,
        state_dict: dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if hasattr(self, "w_scale"):
            w_scale = state_dict.get(prefix + "w_scale")
            w_scale.resize_(self.w_scale.shape)
        if hasattr(self, "x_scale"):
            x_scale = state_dict.get(prefix + "x_scale")
            x_scale.resize_(self.x_scale.shape)
        if hasattr(self, "x_zp"):
            x_zp = state_dict.get(prefix + "x_zp")
            x_zp.resize_(self.x_zp.shape)

    def forward(self, x):
        x_sim = self.x_quantizer(x)
        if self.w_scale.sum() == 0:
            self.calibration(x.detach())
            w_sim = self.weight
        else:
            # x_sim = self.x_quantizer(x)
            # x_sim = fake_quantize(x, self.x_scale, self.x_zp,bitwidth=self.x_bit)
            self.weight.data.clamp_(
                min=self.w_scale * self.w_qmin, max=self.w_scale * self.w_qmax
            )
            w_sim = fake_quantize(self.weight, self.w_scale, bitwidth=self.w_bit)

        return self._conv_forward(x_sim, w_sim, self.bias)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.weight.shape} w={self.w_bit} x={self.x_bit})"


class QMZBConv(QConv):
    w_scale: torch.Tensor
    x_scale: torch.Tensor
    x_zp: torch.Tensor

    def __init__(self, raw_conv: nn.Conv2d, w_bit=8, x_bit=8, shift_value=7) -> None:
        super().__init__(raw_conv, w_bit, x_bit)
        self.shift_value = shift_value
        self.mode = "normal"

    @torch.no_grad()
    def calibration(self, x):
        neg_scale = (
            torch.amin(self.weight, [1, 2, 3], keepdim=True).clip(max=0) / self.w_qmin
        )
        pos_scale = (
            torch.amax(self.weight, [1, 2, 3], keepdim=True).clip(min=0) / self.w_qmax
        )
        w_scale = torch.max(neg_scale, pos_scale).clip(min=eps)
        self.w_scale.data.copy_(w_scale.detach())

    def _pre_load_state_dict_hook(
        self,
        state_dict: dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if hasattr(self, "w_scale"):
            w_scale = state_dict.get(prefix + "w_scale")
            w_scale.resize_(self.w_scale.shape)
        if hasattr(self, "x_scale"):
            x_scale = state_dict.get(prefix + "x_scale")
            x_scale.resize_(self.x_scale.shape)
        if hasattr(self, "x_zp"):
            x_zp = state_dict.get(prefix + "x_zp")
            x_zp.resize_(self.x_zp.shape)

    @torch.no_grad()
    def shift_forward(self, x_sim):
        assert not self.w_scale.sum() == 0, "must be calibrated"
        W = fake_quantize(self.weight, self.w_scale, bitwidth=self.w_bit, sim=False)
        W = (W + self.shift_value).clip(self.w_qmin, self.w_qmax)
        w_sim = W * self.w_scale
        out = self._conv_forward(x_sim, w_sim, self.bias)
        w_correction = -torch.ones_like(w_sim) * self.shift_value * self.w_scale
        correction = self._conv_forward(x_sim, w_correction, None)
        out = out + correction
        return out

    def forward(self, x):
        x_sim = self.x_quantizer(x)
        if self.mode == "shift":  # 为了MZB来进行设计的
            return self.shift_forward(x_sim)

        if self.w_scale.sum() == 0:
            self.calibration(x.detach())
            w_sim = self.weight
        else:
            # x_sim = self.x_quantizer(x)
            # x_sim = fake_quantize(x, self.x_scale, self.x_zp,bitwidth=self.x_bit)
            self.weight.data.clamp_(
                min=self.w_scale * self.w_qmin, max=self.w_scale * self.w_qmax
            )
            w_sim = fake_quantize(self.weight, self.w_scale, bitwidth=self.w_bit)

        return self._conv_forward(x_sim, w_sim, self.bias)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.weight.shape} w={self.w_bit} x={self.x_bit})"


if __name__ == "__main__":
    import torch.optim as opt

    torch.manual_seed(0)
    device = torch.device("cuda:0")
    device = torch.device("cpu")

    # test_mzb
    raw_conv = nn.Conv2d(3, 3, 1)
    conv = QMZBConv(raw_conv)
    x = torch.randn(1, 3, 4, 4)
    y = raw_conv(x)
    y0 = conv(x)
    y1 = conv(x)
    conv.mode = "shift"
    y3 = conv(x)
    pass

    conv = torch.nn.Conv2d(3, 64, 3, bias=True, padding=1).to(device)
    qconv = QMZBConv(conv, w_bit=8, x_bit=8).to(device)
    # qconv(torch.randn(1, 3, 224, 224))

    # test high_zero_conv
    # x = torch.randn(1,3,224,224)
    # conv = torch.nn.Conv2d(3,64,3)
    # qconv = HighZeroQConv(conv)
    # qconv(x)
    # qconv(x)
    # qconv(x).sum().backward()

    # test_acc &
    x = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        y = conv(x) + 1
    optimizer = opt.SGD(qconv.parameters(), lr=1e-4, weight_decay=0.0001)
    for i in range(1000):
        y_pred = qconv(x)
        loss = F.smooth_l1_loss(y_pred, y)
        optimizer.zero_grad()
        torch.cuda.synchronize()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(loss.item())
    pass
