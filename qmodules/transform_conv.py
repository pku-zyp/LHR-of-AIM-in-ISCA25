import torch
import torch.nn as nn
from .qconv import QConv, fake_quantize, QMZBConv
from .qlinear import HighZeroLinear, QMZBLinear

MZBQConv = None

from .qlinear import QLinear
from .qmatmul import QMatMul


def add_sub_module(self, target, m):
    *prefix, field = target.split(".")
    mod: torch.nn.Module = self

    for item in prefix:

        submod = getattr(mod, item, None)

        if submod is None:
            submod = torch.nn.Module()
            setattr(mod, item, submod)

        if not isinstance(submod, torch.nn.Module):
            return False

        mod = submod

    mod.add_module(field, m)
    return True


def transform_to_qnet(
    model: torch.nn.Module, quant_linear=False, use_mzb=False, kwargs={}
) -> None:
    # for name, module in module_name_dict.items():
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if use_mzb:
                layer = QMZBConv(module, **kwargs)
            else:
                layer = QConv(module, **kwargs)
        elif isinstance(module, nn.Linear) and quant_linear:
            if use_mzb:
                layer = QMZBLinear(module, **kwargs)
        else:
            continue
        print(f"replace {name} with {layer}")
        add_sub_module(model, name, layer)


def get_quantized_data(model: torch.nn.Module, img):

    ret_dict = dict()

    def get_hook(name):
        def hook(m: QConv, i, o):
            x = i[0]
            layer_dict = dict()
            layer_dict["w_scale"] = m.w_scale
            layer_dict["x_scale"] = m.x_scale
            layer_dict["x_zp"] = m.x_zp
            layer_dict["x"] = x
            layer_dict["qx"] = (
                fake_quantize(x, m.x_scale, m.x_zp, m.x_bit) / m.x_scale + m.x_zp
            )
            layer_dict["weight"] = m.weight
            layer_dict["qweight"] = (
                fake_quantize(m.weight, m.w_scale, bitwidth=m.w_bit) / m.w_scale
            )
            layer_dict["y"] = o
            ret_dict[name] = layer_dict

        return hook

    hooks = list()
    for name, module in model.named_modules():
        if isinstance(module, QConv, QLinear):
            hooks.append(module.register_forward_hook(get_hook(name)))
    model(img)
    for hook in hooks:
        hook.remove()
    return ret_dict
