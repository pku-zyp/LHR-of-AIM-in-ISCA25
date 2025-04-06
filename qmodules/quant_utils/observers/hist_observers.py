import torch, numpy as np
from functools import partial
from copy import deepcopy
from typing import Union
from ._register import ObserverRegister
from .naive_observers import ObserverABC, eps
from .hist_manager import HistManager


@ObserverRegister.add("percent")
class PercentileObserver(ObserverABC):
    ch_shapes: list

    def __init__(
        self,
        bitwidth=8,
        granularity="tensor",
        symmetric: bool = True,
        hist_bin_num: int = 2048,
        percent: Union[float, list] = 1.0,
        percentile_mode: str = "line",
        asymmetric_signed: bool = True,
    ):
        super().__init__(bitwidth, granularity, symmetric, asymmetric_signed)
        if self.symmetric:
            assert isinstance(
                percent, float
            ), "The percent must be a float when symmetric."
            self.left_percent = percent
            self.right_percent = percent
        else:
            if isinstance(percent, list):
                self.left_percent = percent[0]
                self.right_percent = percent[1]
            elif isinstance(percent, float):
                self.left_percent = self.right_percent = percent
            else:
                raise TypeError("Input must be an float or a list.")
        self.percentile_mode = percentile_mode
        self.hist_manager = HistManager(num_bins=hist_bin_num)
        self.ch_shapes = 1

    def _update_(self, tensor: torch.Tensor):
        hist_manager = self.hist_manager
        if self.ch_axis == -1:
            x = tensor.contiguous().view(1, -1)
        elif isinstance(self.ch_axis, list):
            self.ch_shapes = [tensor.shape[i] for i in self.ch_axis]
            dims = list(range(tensor.dim()))  # self.ch_shapes =
            permute_dims = deepcopy(self.ch_axis)
            for dim in dims:
                if dim not in permute_dims:
                    permute_dims.append(dim)
            x = tensor.permute(permute_dims)
            x = x.reshape(int(np.prod(self.ch_shapes)), -1)  # (#channels, -1)
        else:
            raise NotImplementedError("ch axis must be int or list.")
        hist_manager.collect(data=x)

    def percentile(self):
        if self.left_percent >= 1.0 or self.right_percent >= 1.0:
            assert (
                self.percentile_mode == "line"
            ), "If percent is 1.0, must use line for no loss."
        min_clip_tensor, max_clip_tensor = self.hist_manager.percentile(
            left_percent=self.left_percent,
            right_percent=self.right_percent,
            mode=self.percentile_mode,
        )
        return min_clip_tensor, max_clip_tensor

    def _cal_min_max(self):
        min_val, max_val = self.percentile()
        min_val = min_val.reshape(self.ch_shapes)
        max_val = max_val.reshape(self.ch_shapes)
        self.min_val.resize_(min_val.shape).copy_(min_val)
        self.max_val.resize_(max_val.shape).copy_(max_val)
        return self.min_val, self.max_val


ObserverRegister.add("percent-0.9999")(partial(PercentileObserver, percent=0.9999))
ObserverRegister.add("percent-0.999")(partial(PercentileObserver, percent=0.999))
ObserverRegister.add("percent-0.99995")(partial(PercentileObserver, percent=0.99995))
ObserverRegister.add("percent-0.99997")(partial(PercentileObserver, percent=0.99997))
ObserverRegister.add("percent-0.99999")(partial(PercentileObserver, percent=0.99999))
ObserverRegister.add("percent-0.999995")(partial(PercentileObserver, percent=0.999995))
ObserverRegister.add("percent-0.999999")(partial(PercentileObserver, percent=0.999999))
