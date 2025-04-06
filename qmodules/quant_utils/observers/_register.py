from typing import Union
from .utils import BaseRegister
from .utils import filter_args_of_cfg

ObserverRegister = BaseRegister(key_type=str, add_str_name=True)


# deprecated
def build_observer_for_quantizer(
    cls_type, bitwidth=8, granularity="tensor", symmetric=True, **kwargs
):
    cls = ObserverRegister.get(cls_type)
    return cls(bit_num=bitwidth, granularity=granularity, symmetric=symmetric, **kwargs)


def build_observer(cfg: Union[str, dict] = dict(), **kwargs):
    """Universal Observer Builder, it can build a observer by str or dict

    Args:
        cfg (Union[str, dict], optional): cfg of observer ,it can be a str or dict. Defaults to dict().

    Returns:
        ObserverABC
    """
    if isinstance(cfg, str):
        cfg = dict(type=cfg)
    assert isinstance(cfg, dict), "cfg must be dict"
    cfg.update(**kwargs)
    cls_type = cfg.pop("type", "minmax")
    cls = ObserverRegister.get(cls_type)
    return cls(**filter_args_of_cfg(cfg, cls))
