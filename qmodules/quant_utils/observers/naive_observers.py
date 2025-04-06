import abc
from logging import getLogger
import sys
from typing import Tuple
import torch
import torch.distributed as dist
from torch.distributed import ReduceOp
from ._register import *

eps = torch.tensor(torch.finfo(torch.float32).eps)


def analysis_dim(granularities):
    """解析granularity, 并翻译为具体的channel数值, -1表示per-tensor, list将提取具体的通道数组成list, dim开头提取其后的通道数.

    Args:
        granularities (str or list): tensor or dimx, or [dim0, dim1, ...]

    Returns:
        int or list: 通道id
    """
    ch_axis = None
    if isinstance(granularities, list):
        ch_axis = []
        for granularity in granularities:
            ch_axis.append(int(granularity[3:]))
        for ch in ch_axis:
            assert ch >= 0, "for stability"
    elif granularities == "tensor":
        ch_axis = -1
    elif granularities[:3] == "dim":
        ch_axis = [
            int(granularities[3:]),
        ]
    return ch_axis


class ObserverABC(abc.ABC, torch.nn.Module):
    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(
        self,
        bitwidth: int = 8,
        granularity="tensor",
        symmetric: bool = True,
        asymmetric_signed: bool = False,
    ):
        super().__init__()
        self.bitwidth = bitwidth
        self.granularity = granularity
        self.symmetric = symmetric
        self.asymmetric_signed = asymmetric_signed
        self.quant_min, self.quant_max = self._calculate_qmin_qmax()
        self._ch_axis = analysis_dim(granularities=granularity)
        self.device = None
        self.eps = None
        self.manager: ObserverABC = None
        self.register_buffer("min_val", torch.tensor([]))
        self.register_buffer("max_val", torch.tensor([]))
        self._register_load_state_dict_pre_hook(self._pre_load_state_dict_hook)

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
        min_val = state_dict.get(prefix + "min_val", None)
        max_val = state_dict.get(prefix + "max_val", None)
        if min_val is not None:
            self.min_val.resize_(min_val.shape).copy_(min_val)
        if max_val is not None:
            self.max_val.resize_(max_val.shape).copy_(max_val)

    @property
    def ch_axis(self):
        return self._ch_axis

    def _calculate_qmin_qmax(self) -> Tuple[int, int]:
        r"""Calculates actual qmin and qmax based on the quantization range,
        observer datatype and if range is reduced.
        """
        if self.symmetric:
            quant_min = -(1 << (self.bitwidth - 1))
            quant_max = (1 << (self.bitwidth - 1)) - 1
        else:
            if self.asymmetric_signed:
                quant_min = -(1 << (self.bitwidth - 1))
                quant_max = (1 << (self.bitwidth - 1)) - 1
            else:
                quant_min = 0
                quant_max = (1 << self.bitwidth) - 1
        return quant_min, quant_max

    @torch.no_grad()
    def calculate_scale_zero_point(self):
        min_val, max_val = self.cal_min_max()
        assert min_val is not None and max_val is not None
        quant_min, quant_max = self.quant_min, self.quant_max
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

        device = min_val_neg.device
        scale = torch.ones(min_val_neg.size(), dtype=torch.float32, device=device)
        zero_point = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

        if self.symmetric:
            scale = torch.max(
                torch.abs(min_val_neg / self.quant_min),
                torch.abs(max_val_pos / self.quant_max),
            )
            scale = torch.max(scale, eps.to(scale.device))
        else:
            scale = ((max_val_pos - min_val_neg) / float(quant_max - quant_min)).abs()
            scale = torch.max(scale, eps.to(scale.device))
            zero_point = quant_min - torch.round(min_val_neg / scale).to(torch.int)
            zero_point = torch.clamp(zero_point, quant_min, quant_max)
        return scale, zero_point

    @abc.abstractmethod
    def _cal_min_max(self):
        return self.min_val, self.max_val

    def cal_min_max(self):
        min_val, max_val = self._cal_min_max()
        if dist.is_initialized():
            dist.all_reduce(min_val, op=ReduceOp.MIN)
            dist.all_reduce(max_val, op=ReduceOp.MAX)
        return min_val, max_val

    @abc.abstractmethod
    def _update_(self, tensor: torch.Tensor):
        pass

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        if self.manager is not None:
            self.manager.update(x)
        self._update_(x)

    def forward(self, x: torch.Tensor):
        self.update(x)
        return x


@ObserverRegister.add("minmax", "MinMax", "MINMAX")
class MinMaxObserver(ObserverABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observer_name = "MinMax"

    def _cal_min_max(self):
        return super()._cal_min_max()

    def _update_(self, x: torch.Tensor):
        dims = tuple(range(x.dim()))
        if self.ch_axis != -1:
            dims = [dim for dim in dims if dim not in self.ch_axis]
        max_val = torch.amax(x, dim=dims, keepdim=False)
        min_val = torch.amin(x, dim=dims, keepdim=False)
        if max_val.dim() == 0 or min_val.dim() == 0:
            assert max_val.dim() == min_val.dim()
            max_val = max_val.reshape(-1)
            min_val = min_val.reshape(-1)
        if self.max_val.numel() == 0:
            self.max_val.resize_(max_val.shape).fill_(0)
        if self.min_val.numel() == 0:
            self.min_val.resize_(min_val.shape).fill_(0)
        self.max_val.data.copy_(torch.max(self.max_val, max_val))
        self.min_val.data.copy_(torch.min(self.min_val, min_val))


@ObserverRegister.add("ema", "EMA", "Ema")
class EMAMinMaxObserver(ObserverABC):
    def __init__(
        self,
        bitwidth: int = 8,
        granularity="tensor",
        symmetric: bool = True,
        asymmetric_signed: bool = False,
        averaging_constant=0.05,
    ):
        super().__init__(
            bitwidth=bitwidth,
            granularity=granularity,
            symmetric=symmetric,
            asymmetric_signed=asymmetric_signed,
        )
        self.averaging_constant = averaging_constant
        self.observer_name = "EmaMinMax"

    def _cal_min_max(self):
        return super()._cal_min_max()

    def _update_(self, x: torch.Tensor):
        dims = tuple(range(x.dim()))
        if self.ch_axis != -1:
            dims = [dim for dim in dims if dim not in self.ch_axis]
        max_val_cur = torch.amax(x, dim=dims, keepdim=False)
        min_val_cur = torch.amin(x, dim=dims, keepdim=False)
        if max_val_cur.dim() == 0 or min_val_cur.dim() == 0:
            max_val_cur = max_val_cur.reshape(-1)
            min_val_cur = min_val_cur.reshape(-1)
        if self.max_val.numel() == 0:
            self.max_val.resize_(max_val_cur.shape).copy_(max_val_cur)
        if self.min_val.numel() == 0:
            self.min_val.resize_(min_val_cur.shape).copy_(min_val_cur)

        self.max_val.copy_(
            self.max_val + self.averaging_constant * (max_val_cur - self.max_val)
        )
        self.min_val.copy_(
            self.min_val + self.averaging_constant * (min_val_cur - self.min_val)
        )


@ObserverRegister.add("fixed", "FIXED", "Fixed")
class FixedObserver(ObserverABC):
    def __init__(
        self,
        bitwidth: int = 8,
        granularity="tensor",
        symmetric: bool = True,
        asymmetric_signed: bool = False,
        min=-1,
        max=1,
    ):
        super().__init__(
            bitwidth=bitwidth,
            granularity=granularity,
            symmetric=symmetric,
            asymmetric_signed=asymmetric_signed,
        )
        min = torch.tensor(min)
        max = torch.tensor(max)

        if min.dim() == 0:
            min = min.reshape(-1)
            max = max.reshape(-1)
        self.min_val.resize_(min.shape).copy_(min)
        self.max_val.resize_(max.shape).copy_(max)

    def _cal_min_max(self):
        return super()._cal_min_max()

    def _update_(self, x):
        pass
