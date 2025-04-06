import torch, torch.nn as nn
import numpy as np, matplotlib.pyplot as plt


class HammingLoss(nn.Module):
    def __init__(self, bit=8):
        self.bit = bit
        if bit == 8:
            nums = list(range(-128, 128, 1))
            hms = [
                np.binary_repr(i, width=8).replace("0b", "").count("1") for i in nums
            ]
        elif bit == 4:
            nums = list(range(-8, 8, 1))
            hms = [
                np.binary_repr(i, width=4).replace("0b", "").count("1") for i in nums
            ]
        else:
            raise NotImplemented
        super().__init__()
        self.register_buffer("hms", torch.tensor(hms).float())

    def forward(self, x, reduce="sum"):
        if self.hms.device != x.device:
            self.to(x.device)
        if self.bit == 8:
            assert x.max() <= 127.5 and x.min() >= -128.5, ""
            x += 128
        else:
            assert x.max() <= 7.5 and x.min() >= -8.5, ""
            x += 8
        low = torch.floor(x).long().clip_(0)
        high = torch.ceil(x).long().clip_(2 ** (self.bit) - 1)
        low_val = self.hms[low]
        high_val = self.hms[high]
        frac = x - low.float()
        ret = torch.lerp(low_val, high_val, frac)
        if reduce == "sum":
            return ret.sum()
        else:
            return ret.mean()


if __name__ == "__main__":
    x = torch.rand(1, 3, 224, 224) * 127
    ret = HammingLoss()(x)
    print()
