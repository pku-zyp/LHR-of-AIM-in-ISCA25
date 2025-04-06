from copy import deepcopy
import os, inspect
from typing import List
import torch
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader


def filter_args_of_cfg(args, callable):
    args = deepcopy(args)
    var_names = set()
    for k, v in inspect.signature(callable).parameters.items():
        if v.kind == v.VAR_KEYWORD:
            return args
        else:
            var_names.add(k)
    ret_args = {k: args.get(k) for k in var_names if args.get(k) is not None}
    return ret_args


class BaseRegister:
    """Register to map a str and a obj"""

    def __init__(
        self, D: dict = None, key_type=None, value_type=None, add_str_name=True
    ):
        if D is None:
            D = dict()
        self.D = D
        self.key_type = key_type
        self.value_type = value_type
        self.add_str_name = add_str_name

    def __repr__(self):
        return self.D.__repr__()

    def add(self, *keys, **info):
        def insert(value):
            if self.value_type:
                assert isinstance(value, self.value_type) or issubclass(
                    value, self.value_type
                ), "must matching"
            names = keys
            names = set(names)
            if hasattr(value, "__name__") and self.add_str_name:
                names.add(value.__name__)
            for name in names:
                if self.key_type:
                    assert isinstance(
                        name, (self.key_type, str)
                    ), f"key of register must be {self.key_type}, not {type(name)} "
                if len(info):
                    self.D[name] = (value, info)
                else:
                    self.D[name] = value
            return value

        return insert

    def get(self, key):
        if key not in self.D.keys():
            raise Exception(f"key: {key} not exists")
        return self.D[key]

    def has(self, key):
        return key in self.D.keys()

    def build(self, key, **kwargs):
        cls = self.D[key]
        return cls(**kwargs)

    def return_when_exist(self, key):
        if key in self.D.keys():
            return self.D[key]
        else:
            return None
