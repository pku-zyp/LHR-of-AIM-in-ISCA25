import torch
import torch.nn as nn
from quant_utils.observers import ObserverABC, build_observer
import abc

# from hmquant.ptq.nn_layers.observers import build_observer
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed as dist
from torch.distributed import ReduceOp

eps = torch.finfo(torch.float32).eps * 10


def grad_scale(t, scale):
    return (t - (t * scale)).detach() + (t * scale)


def _fake_quantize_learnable_per_channel_affine_training(
    x, scale, zero_point, ch_axis, quant_min, quant_max, grad_factor
):
    zero_point = (zero_point.round() - zero_point).detach() + zero_point
    new_shape = [1] * len(x.shape)
    new_shape[ch_axis] = x.shape[ch_axis]
    scale = grad_scale(scale, grad_factor).reshape(new_shape)
    zero_point = grad_scale(zero_point, grad_factor).reshape(new_shape)
    x = x / scale + zero_point
    x = (x.round() - x).detach() + x
    x = torch.clamp(x, quant_min, quant_max)
    return (x - zero_point) * scale


class QuantizerABC(abc.ABC, nn.Module):
    fake_quant_enabled: torch.Tensor
    observer_enabled: torch.Tensor
    calibed_enabled: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    observer: ObserverABC

    def __init__(self):
        super().__init__()
        self.register_buffer("fake_quant_enabled", torch.tensor([0], dtype=torch.uint8))
        self.register_buffer("observer_enabled", torch.tensor([0], dtype=torch.uint8))
        self.register_buffer("calibed_enabled", torch.tensor([0], dtype=torch.uint8))
        self.register_buffer("learnable_enabled", torch.tensor([0], dtype=torch.uint8))
        self._observer_enable = False
        self._fake_quant_enable = False
        self._calibrated = False
        self._learnable = False
        self.observer = None
        # TODO 是否更优雅的实现calib_iters
        self.calib_iters = 0
        self._register_load_state_dict_pre_hook(hook=self._pre_load_state_dict_hook)

    @abc.abstractmethod
    def forward(self, x):
        pass

    """1. fake_quant_flag"""

    def disable_fake_quant(self):
        self.enable_fake_quant(False)

    def enable_fake_quant(self, enabled: bool = True):
        self._fake_quant_enable = enabled
        self.fake_quant_enabled[0] = 1 if enabled else 0

    @property
    def fake_quant_enable(self):
        return self._fake_quant_enable

    @fake_quant_enable.setter
    def fake_quant_enable(self, value: bool):
        self._fake_quant_enable = value
        self.fake_quant_enabled[0] = 1 if value else 0

    """2. observer_flag"""

    def enable_observer(self, enabled: bool = True):
        self._observer_enable = enabled
        self.observer_enabled[0] = 1 if enabled else 0

    def disable_observer(self):
        # TODO 自动执行一次计算scale和zero_point
        self.enable_observer(False)

    @property
    def observer_enable(self):
        return self._observer_enable

    @observer_enable.setter
    def observer_enable(self, value: bool):
        self._observer_enable = value
        self.observer_enabled[0] = 1 if value else 0

    """3. calibrated_flag"""

    @property
    def calibrated(self):
        return self._calibrated

    @calibrated.setter
    def calibrated(self, value: bool):
        self._calibrated = value
        self.calibed_enabled[0] = 1 if value else 0

    """4. enable_learn_flag"""

    @property
    def learnable_enable(self):
        return self._learnable

    @learnable_enable.setter
    def learnable_enable(self, value: bool):
        self._learnable = value
        self.learnable_enabled[0] = 1 if value else 0
        self.scale.requires_grad_(value)
        if self.zero_point and (not self.symmetric):
            self.zero_point.requires_grad_(value)

    def enable_learn(self, enable=True):
        self.learnable_enable = enable

    def disable_learn(self):
        self.enable_learn(False)

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
        v = state_dict.get(prefix + "observer_enabled", None)
        if v is not None:
            self.observer_enable = v[0].item() == 1
        v = state_dict.get(prefix + "fake_quant_enabled", None)
        if v is not None:
            self.fake_quant_enable = v[0].item() == 1
        v = state_dict.get(prefix + "calibed_enabled", None)
        if v is not None:
            self.calibrated = v[0].item() == 1
        v = state_dict.get(prefix + "learnable_enabled", None)
        if v is not None:
            self.learnable_enable = v[0].item() == 1

    def __repr__(self) -> str:
        if self.scale is not None:
            scale = self.scale.view(-1).cpu()[:6]
        else:
            scale = "not calibrate"
        s = f"""{self.__class__.__name__} {self.bitwidth}bit scale={scale} \
zero_point={self.zero_point} metric={self.metric} \
learnable={self.learnable_enable} symmetric={self.symmetric} granularity={self.granularity}"""
        return s

    """"""


class UnionQuantizer(QuantizerABC):
    def __init__(
        self,
        bitwidth,
        granularity,
        calib_metric="minmax",
        symmetric=True,
        learnable=False,
        tensor_shape=None,
        calib_iters=0,
    ):
        super().__init__()
        self.bitwidth = bitwidth
        self.granularity = granularity
        self.metric = calib_metric
        self.qmin = int(-(1 << (bitwidth - 1)))
        self.qmax = int((1 << (bitwidth - 1)) - 1)
        self.symmetric = symmetric
        if isinstance(granularity, (tuple, list)):
            if len(granularity) == 0:
                granularity = "tensor"
            else:
                granularity = ",".join([f"dim{_}" for _ in granularity])
        self.granularity = granularity
        if granularity is None or granularity == "tensor":
            self.granularity_dims = []
        else:
            dims = granularity.split(",")
            self.granularity_dims = [int(_[3:]) for _ in dims]
        assert (
            len(self.granularity_dims) <= 1
        ), "Only support 1 dimension quantization now"
        if tensor_shape is None:
            shape = [1]
        else:
            shape = [
                _ for i, _ in enumerate(tensor_shape) if i in self.granularity_dims
            ]
        self.scale = torch.nn.Parameter(torch.ones(shape) * torch.nan)
        self.zero_point = None
        if not symmetric:
            self.zero_point = torch.nn.Parameter(torch.zeros_like(self.scale))

        self.learnable_enable = learnable
        if isinstance(calib_metric, str):
            observer_cfg = dict(
                type=calib_metric,
                bitwidth=bitwidth,
                granularity=granularity,
                symmetric=symmetric,
            )
        else:
            observer_cfg = calib_metric
            observer_cfg.update(
                bitwidth=bitwidth, granularity=granularity, symmetric=symmetric
            )
        self.observer: ObserverABC = build_observer(observer_cfg)
        self.calib_iters = calib_iters
        if self.calib_iters >= 1:
            self.enable_observer()
            self.disable_fake_quant()

    @torch.no_grad()
    def compute_quant_params(self):
        scale, zero_point = self.observer.calculate_scale_zero_point()
        if dist.is_initialized():  # scale synchronization
            dist.all_reduce(scale, op=ReduceOp.MAX)
        self.scale.data.copy_(scale.reshape(self.scale.shape))
        self.scale.data.clamp_min_(eps)
        if self.zero_point is not None:  # TODO support dist synchronization
            if dist.is_initialized():
                zero_point = zero_point / dist.get_world_size()
                dist.all_reduce(zero_point)
            self.zero_point.data.copy_(zero_point.reshape(self.zero_point.shape))
        self.calibrated = True

    def calib_iter_trigger(self):
        if not self.calibrated and self.calib_iters >= 1:
            if self.calib_iters == 1:
                self.enable_fake_quant()
                self.disable_observer()
                self.compute_quant_params()
            self.calib_iters -= 1

    def change_scale(self, new_scale):
        self.scale.data.copy_(new_scale.data)

    @property
    def range_max(self):
        if self.symmetric:
            return self.qmax * self.scale.detach()
        else:
            return (self.qmax - self.zero_point) * self.scale.detach()

    @property
    def range_min(self):
        if self.symmetric:
            return self.qmin * self.scale.detach()
        else:
            return (self.qmin - self.zero_point) * self.scale.detach()

    def forward(self, x: torch.Tensor):
        if self.observer_enable:
            self.observer.update(x.detach())
        if self.fake_quant_enable:
            if not self.calibrated:
                self.calibrated = True
                self.compute_quant_params()

            self.scale.data.clamp_min_(eps)
            if self.zero_point is not None:
                zero_point = self.zero_point
            else:
                zero_point = torch.zeros_like(self.scale)

            if len(self.granularity_dims) == 0:
                grad_factor = 1.0 / ((x.numel() * self.qmax) ** 0.5)
                x = torch._fake_quantize_learnable_per_tensor_affine(
                    x,
                    self.scale,
                    zero_point,
                    self.qmin,
                    self.qmax,
                    grad_factor=grad_factor,
                )
            else:
                grad_factor = (
                    1.0
                    / (x.numel() / x.shape[self.granularity_dims[0]] * self.quant_max)
                    ** 0.5
                )
                x = _fake_quantize_learnable_per_channel_affine_training(
                    x,
                    self.scale,
                    zero_point,
                    self.granularity_dims[0],
                    self.qmin,
                    self.qmax,
                    grad_factor,
                )
            x.quantizer = self
        # 用来根据iters进行不同的触发
        self.calib_iter_trigger()
        return x


def enable_learn(model: nn.Module):
    for module in model.modules():
        if isinstance(module, QuantizerABC):
            module.enable_learn()


def disable_learn(model: nn.Module):
    for module in model.modules():
        if isinstance(module, QuantizerABC):
            module.disable_learn()


def enable_observer(model: nn.Module):
    for module in model.modules():
        if isinstance(module, QuantizerABC):
            module.enable_observer()


def disable_observer(model: nn.Module):
    for module in model.modules():
        if isinstance(module, QuantizerABC):
            module.disable_observer()


def enable_fakequant(model: nn.Module):
    for module in model.modules():
        if isinstance(module, QuantizerABC):
            module.enable_fake_quant()


def disable_fakequant(model: nn.Module):
    for module in model.modules():
        if isinstance(module, QuantizerABC):
            module.disable_fake_quant()


def set_calib(model: nn.Module):
    enable_observer(model)
    disable_fakequant(model)


def set_fake_quant(model: nn.Module):
    disable_observer(model)
    enable_fakequant(model)


if __name__ == "__main__":
    q = UnionQuantizer(8, "tensor", "minmax", True, True)
    q.observer_enable = True
    q.enable_observer()
    q.disable_learn()
    x = torch.randn(1, 3, 224, 224)
    y1 = q(x)

    q.enable_learn()
    q.enable_fake_quant()
    x.requires_grad_(True)
    y2 = q(x)
    y2.sum().backward()

    q1 = UnionQuantizer(8, "tensor", "minmax", True, True)
    q1.load_state_dict(q.state_dict())
    y3 = q1(x)
    print()
