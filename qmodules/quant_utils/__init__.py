import torch, torch.nn as nn
from torch.nn import Identity
from .observers import *
from .quantizer import QuantizerABC, UnionQuantizer

from .quantizer import (
    UnionQuantizer,
    enable_learn,
    disable_learn,
    enable_observer,
    disable_observer,
    enable_fakequant,
    disable_fakequant,
    set_calib,
    set_fake_quant,
)
from .img_utils import *


def build_quantizer(cfg, tensor_shape=None):
    if cfg is None:
        return Identity()
    if isinstance(cfg, dict):
        dtype = cfg.get("dtype", "int8")
        if dtype[:3] == "int":
            bitwidth = int(dtype[3:])
        elif dtype[:4] == "uint":
            bitwidth = int(dtype[4:])
        else:
            raise ValueError(f"Invalid dtype: {dtype}")
        granularity = cfg.get("granularity", "tensor")
        calib_metric = cfg.get("calib_metric", "minmax")
        calib_iters = cfg.get("calib_iters", 1)
        learnable = cfg.get("learnable", False)
        symmetric = cfg.get("symmetric", True)
        return UnionQuantizer(
            bitwidth,
            granularity=granularity,
            calib_metric=calib_metric,
            calib_iters=calib_iters,
            learnable=learnable,
            tensor_shape=tensor_shape,
            symmetric=symmetric,
        )
    raise ValueError(f"Invalid quantizer config: {cfg}")
