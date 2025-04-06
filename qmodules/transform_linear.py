import torch, torch.nn as nn
from torch.fx.graph_module import GraphModule

GraphModule.add_submodule
from .qlinear import QLinear


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


def transform_to_qnet(model: torch.nn.Module, w_bit=8, x_bit=8):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            qlinear = QLinear(module, w_bit, x_bit)
            add_sub_module(model, name, qlinear)

    return model
