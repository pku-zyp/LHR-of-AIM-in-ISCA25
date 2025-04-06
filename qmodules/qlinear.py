import torch, torch.nn as nn, torch.nn.functional as F

from .quant_utils import build_quantizer

eps = torch.finfo(torch.float32).eps * 10


class STERound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
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


class QLinear(nn.Linear):
    def __init__(self, raw_linear: nn.Linear, w_bit=8, x_bit=8) -> None:
        super().__init__(
            raw_linear.in_features,
            raw_linear.out_features,
            raw_linear.bias is not None,
            raw_linear.weight.device,
        )
        self.weight.data.copy_(raw_linear.weight.data)
        if self.bias is not None:
            self.bias.data.copy_(raw_linear.bias.data)
        self.register_buffer(
            "w_scale",
            torch.zeros(
                [
                    self.out_features,
                    1,
                ]
            ),
        )
        self.register_buffer("x_scale", torch.zeros(1))
        self.register_buffer("x_zp", torch.zeros(1))
        self.w_bit = w_bit
        self.x_bit = x_bit
        self.w_qmin = -(1 << (w_bit - 1))
        self.w_qmax = (1 << (w_bit - 1)) - 1
        self.calibrated = False

        self.x_quantizer = build_quantizer(
            dict(
                calib_metric="percent-0.99999",
                bitwidth=f"int{w_bit}",
                learnable=True,
                symmetric=False,
            )
        )

        self._register_load_state_dict_pre_hook(self._pre_load_state_dict_hook)

    @torch.no_grad()
    def calibration(self, x):
        neg_scale = (
            torch.amin(
                self.weight,
                [
                    1,
                ],
                keepdim=True,
            ).clip(max=0)
            / self.w_qmin
        )
        pos_scale = (
            torch.amax(
                self.weight,
                [
                    1,
                ],
                keepdim=True,
            ).clip(min=0)
            / self.w_qmax
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
            self.calibrated = True
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
            self.calibrated = True
        else:
            self.weight.data.clamp_(
                min=self.w_scale * self.w_qmin, max=self.w_qmax * self.w_scale
            )
            w_sim = fake_quantize(self.weight, self.w_scale, self.w_bit)
        return F.linear(x_sim, w_sim, self.bias)

    def __repr__(self):
        return f"QLinear{self.weight.shape},w={self.w_bit},x={self.x_bit}"


class QMZBLinear(QLinear):
    def __init__(self, raw_linear: nn.Conv2d, w_bit=8, x_bit=8, shift_value=7):
        super().__init__(raw_linear, w_bit, x_bit)
        self.shift_value = shift_value
        self.mode = "normal"

    @torch.no_grad()
    def shift_forward(self, x_sim):
        assert not self.w_scale.sum() == 0, "must be calbirated"
        W = fake_quantize(self.weight, self.w_scale, bitwidth=self.w_bit, sim=False)
        W = (W + self.shift_value).clip(self.w_qmin, self.w_qmax)
        w_sim = W * self.w_scale
        out = F.linear(x_sim, w_sim, self.bias)
        w_correction = -torch.ones_like(w_sim) * self.shift_value * self.w_scale
        correction = F.linear(x_sim, w_correction, None)
        out = out + correction
        return out

    def forward(self, x):
        x_sim = self.x_quantizer(x)
        if self.mode == "shift":
            return self.shift_forward(x_sim)
        if self.w_scale.sum() == 0:
            self.calibration(x.detach())
            w_sim = self.weight
            self.calibrated = True
        else:
            self.weight.data.clamp_(
                min=self.w_scale * self.w_qmin, max=self.w_qmax * self.w_scale
            )
            w_sim = fake_quantize(self.weight, self.w_scale, self.w_bit)
        return F.linear(x_sim, w_sim, self.bias)


class HighZeroLinear(QLinear):
    def __init__(
        self,
        raw_linear: nn.Linear,
        w_bit=8,
        x_bit=8,
        mask_type="linear",
        high_zero_thresh=7,
        l1_norm=1000,
        clip=False,
    ):
        super().__init__(raw_linear, w_bit, x_bit)
        self.mask_type = mask_type
        self.high_zero_thresh = high_zero_thresh
        self.l1_norm = l1_norm
        if not clip:
            self.weight.register_hook(self._weight_backward_hook)
        else:
            self.register_forward_pre_hook(self._clip_weight)

    @torch.no_grad()
    def _clip_weight(self, module, input):
        if self.w_scale.sum() == 0:
            return
        s_w = self.w_scale
        if self.mask_type == "ic":
            clip_max = s_w * self.high_zero_thresh
            clip_min = torch.zeros_like(clip_max)
            self.weight.data[:, ::4].clamp_(min=clip_min, max=clip_max)
        elif self.mask_type == "linear":
            clip_max = (
                (s_w * self.high_zero_thresh).expand_as(self.weight).reshape(-1)[::4]
            )
            clip_min = torch.zeros_like(clip_max)
            self.weight.data.view(-1)[::4].clamp_(min=clip_min, max=clip_max)
        else:
            raise NotImplemented

    # 注册在weight上的hook，用于改变weight对应的梯度从而能够实现 split和
    # @torch.no_grad()
    def _weight_backward_hook(self, grad):
        if self.w_scale.sum() == 0:
            return
        s_w = self.w_scale
        w = self.weight.clone()
        if self.mask_type == "ic":
            clip_max = s_w * self.high_zero_thresh
            clip_min = torch.zeros_like(clip_max)
            w[:, ::4].clamp_(min=clip_min, max=clip_max)
        elif self.mask_type == "linear":
            clip_max = (s_w * self.high_zero_thresh).expand_as(w).reshape(-1)[::4]
            clip_min = torch.zeros_like(clip_max)
            w.view(-1)[::4].clamp_(
                min=clip_min, max=clip_max
            )  # 有bug 因为clamp_min和clamp_max是一个per_channel的数字
        else:
            raise NotImplemented
        outlier = self.weight - w
        grad.add_(self.l1_norm * outlier)  # 梯度应该跟符号的方向一致
        # TODO 将weight的截断过程 ,同样使用L1正则化来解决

    @torch.no_grad()
    def get_exceed_sum(self):
        W = fake_quantize(self.weight, self.w_scale, bitwidth=self.w_bit, sim=False)
        if self.mask_type == "linear":
            return (
                (
                    W.view(-1)[::4]
                    - W.view(-1)[::4].clip(min=0, max=self.high_zero_thresh)
                )
                .abs()
                .sum()
            )
        else:
            return (
                (W[:, ::4] - W[:, ::4].clip(min=0, max=self.high_zero_thresh))
                .abs()
                .sum()
            )


if __name__ == "__main__":
    linear = nn.Linear(3, 64)
    qlinear = QLinear(linear)
    x = torch.randn(1000, 3)
    y0 = linear(x)
    y1 = qlinear(x)
    err = torch.abs(y0 - y1)
    print()
