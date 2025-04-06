import math
from copy import deepcopy
from typing import Union
import torch
from logging import getLogger

eps = torch.finfo(torch.float32).eps * 10


def percentile_center_onedirection(
    hists_mat: torch.Tensor, centers: torch.Tensor, percent: float
):
    """根据参数对直方图的频度进行center模式(中线)的percentile操作.
        注意, 是对尾端percentile. 如果想做头端percentile请对传入的hist和centers进入这个功能提前做reversed().

    Args:
        hists_mat (torch.Tensor): 直方图构成的矩阵.
        centers (torch.Tensor): 直方图各个bin的中线.
        percent (float): 保留前percent的bin, percent属于[0., 1.]
    Returns:
        torch.Tensor: 各个通道最小值组成的tensor, 表示尾端的clip界限.
    """
    assert hists_mat.dim() == 2 and centers.dim() == 2, "must"
    hists_sum = hists_mat.sum(dim=-1, keepdim=True)  # (B, 1)
    hists_cum = hists_mat.cumsum(dim=-1)  # (B,N)
    target = percent * hists_sum  # (B,1)
    idx = (hists_cum - target).abs().argmin(dim=-1)  # distance:(B,), idx(B,)
    target_centers = centers.gather(dim=-1, index=idx.reshape(-1, 1))
    return target_centers.reshape(-1)


def percentile_linear_onedirection(
    hists_mat: torch.Tensor, bin_edges_mat: torch.Tensor, percent: float
):
    """根据参数对直方图的频度进行linear模式(线性插值)的percentile操作.
        注意, 是对尾端percentile. 如果想做头端percentile请对传入的hist和bin_edges进入这个功能提前做reversed().

    Args:
        hists_mat (torch.Tensor): 直方图.
        bin_edges_mat (torch.Tensor): 直方图各个bin的edges.
        percent (float): 保留前percent的bin, percent属于[0., 1.]

    Returns:
        torch.Tensor: 只有一个值的tensor, 表示尾端的clip界限.
    """
    assert hists_mat.dim() == 2 and bin_edges_mat.dim() == 2, "must"
    hists_sum = hists_mat.sum(dim=-1, keepdim=True)  # (B, 1)
    hists_cum = hists_mat.cumsum(dim=-1)  # (B,N)
    target = percent * hists_sum  # (B,1)
    idx = ((hists_cum - target) >= 0).int().argmax(dim=-1)  # first idx that >= target
    r_csum = hists_cum.gather(-1, idx.reshape(-1, 1))
    l_csum = hists_cum.gather(-1, (idx - 1).clip_(0).reshape(-1, 1))
    p = (r_csum - target) / (r_csum - l_csum)
    p = p.clip(0, 1)
    return (
        bin_edges_mat.gather(-1, idx.reshape(-1, 1)) * p
        + bin_edges_mat.gather(-1, (idx + 1).reshape(-1, 1)) * (1 - p)
    ).reshape(-1)


def _percentile_center_onedirection(
    hists_mat: torch.Tensor, centers: torch.Tensor, percent: float
):
    """根据参数对直方图的频度进行center模式(中线)的percentile操作.
        注意, 是对尾端percentile. 如果想做头端percentile请对传入的hist和centers进入这个功能提前做reversed().

    Args:
        hists_mat (torch.Tensor): 直方图构成的矩阵.
        centers (torch.Tensor): 直方图各个bin的中线.
        percent (float): 保留前percent的bin, percent属于[0., 1.]
    Returns:
        torch.Tensor: 各个通道最小值组成的tensor, 表示尾端的clip界限.
    """
    frec_sum = hists_mat.sum(dim=-1)
    step_sum = hists_mat.sum(dim=-1)  # (a, b) or (a)
    flipped_hists_mat = torch.flip(input=hists_mat, dims=(-1,))
    flipped_centers = torch.flip(input=centers, dims=(-1,))
    # flipped_hists_mat, flipped_centers = roll(flipped_hists_mat,flipped_centers) # TODO for stablibity not use it
    # 如果flipped_centers的绝对值最大的值是正数, 初始化为-inf, 如果是负数, 初始化为inf.
    clip_record_tensor = (
        torch.ones(size=list(hists_mat.shape[:-1]), device=hists_mat.device)
        * float("-inf")
        * torch.sign(flipped_centers.contiguous().view(-1)[0])
    )
    for i in range(hists_mat.shape[-1]):
        frec_tensor = flipped_hists_mat[..., i]
        center_tensor = flipped_centers[..., i]
        step_sum -= frec_tensor
        step_mask = step_sum <= (frec_sum * percent)
        step_none_inf_mask = clip_record_tensor == (
            torch.ones(size=list(hists_mat.shape[:-1]), device=hists_mat.device)
            * float("-inf")
            * torch.sign(flipped_centers.contiguous().view(-1)[0])
        )
        step_mask = step_mask & step_none_inf_mask
        clip_record_tensor[step_mask] = center_tensor[step_mask]
        if torch.all(
            clip_record_tensor
            != (
                torch.ones(size=list(hists_mat.shape[:-1]), device=hists_mat.device)
                * float("-inf")
                * torch.sign(flipped_centers.contiguous().view(-1)[0])
            )
        ):
            break
    return clip_record_tensor


def _percentile_linear_onedirection(
    hists_mat: torch.Tensor, bin_edges_mat: torch.Tensor, percent: float
):
    """根据参数对直方图的频度进行linear模式(线性插值)的percentile操作.
        注意, 是对尾端percentile. 如果想做头端percentile请对传入的hist和bin_edges进入这个功能提前做reversed().

    Args:
        hists_mat (torch.Tensor): 直方图.
        bin_edges_mat (torch.Tensor): 直方图各个bin的edges.
        percent (float): 保留前percent的bin, percent属于[0., 1.]

    Returns:
        torch.Tensor: 只有一个值的tensor, 表示尾端的clip界限.
    """
    frec_sum = hists_mat.sum(dim=-1)
    # 做全0channel的检查,
    zero_channel_mask = frec_sum == 0

    step_sum = hists_mat.sum(dim=-1)  # (a, b) or (a)
    flipped_hists_mat = torch.flip(input=hists_mat, dims=(-1,))
    flipped_bin_edges_mat = torch.flip(input=bin_edges_mat, dims=(-1,))
    # roll to avoid zero
    # flipped_hists_mat,flipped_bin_edges_mat = roll(flipped_hists_mat,flipped_bin_edges_mat)
    # roll to avoid zero
    clip_record_tensor = (
        torch.ones(size=list(hists_mat.shape[:-1]), device=hists_mat.device)
        * float("-inf")
        * torch.sign(flipped_bin_edges_mat.contiguous().view(-1)[0])
    )
    clip_record_tensor[zero_channel_mask] = eps * torch.sign(
        flipped_bin_edges_mat.contiguous().view(-1)[0]
    )
    for i in range(hists_mat.shape[-1]):
        frec_tensor = flipped_hists_mat[..., i]
        step_sum -= frec_tensor
        step_mask = step_sum <= (frec_sum * percent)  # (a, b) or (a)
        right_edge = flipped_bin_edges_mat[..., i + 1]
        left_edge = flipped_bin_edges_mat[..., i]
        # frec_tensor中有可能有0, 所以要排除freq_tensor中的0.
        none_zero_mask = frec_tensor != 0
        step_none_zero_mask = step_mask & none_zero_mask
        step_none_inf_mask = clip_record_tensor == (
            torch.ones(size=list(hists_mat.shape[:-1]), device=hists_mat.device)
            * float("-inf")
            * torch.sign(flipped_bin_edges_mat.contiguous().view(-1)[0])
        )
        step_none_zero_mask = step_none_zero_mask & step_none_inf_mask
        extra_frec = (frec_sum[step_none_zero_mask] * percent) - step_sum[
            step_none_zero_mask
        ]
        ratio = extra_frec / frec_tensor[step_none_zero_mask]
        ratio[ratio > 1.0] = 1.0
        linear = (
            left_edge[step_none_zero_mask] * ratio
            + right_edge[step_none_zero_mask] * (ratio - 1.0).abs()
        )
        clip_record_tensor[step_none_zero_mask] = linear
        if torch.all(
            clip_record_tensor
            != (
                torch.ones(size=list(hists_mat.shape[:-1]), device=hists_mat.device)
                * float("-inf")
                * torch.sign(flipped_bin_edges_mat.contiguous().view(-1)[0])
            )
        ):
            if any(zero_channel_mask):
                hists_mat[zero_channel_mask][0] -= 1
            break
    return clip_record_tensor


def torch_kl_stable(pred: torch.Tensor, ref: torch.Tensor):
    """计算伪量化后的hist与量化前浮点直方图之间的KL散度.

    Args:
        pred (torch.Tensor): 伪量化后的hist.
        ref (torch.Tensor): 量化前的hist.

    Returns:
        torch.Tensor: KL散度.
    """
    mask = ref != 0
    if pred[-1] == 0:  # for numerical stability
        pred[-1] = 1
    pred = pred.to(torch.float)[mask]
    ref = ref.to(torch.float)[mask]
    psum = pred.sum()
    rsum = ref.sum()
    p_sum = (ref * torch.log(psum * pred)).sum()
    r_sum = (ref * torch.log(rsum * ref)).sum()
    return (r_sum - p_sum) / rsum


def binary_search(x: torch.Tensor, value: float):
    """二分查找, 用于找到bin_edges中0.两侧edges的索引.

    Args:
        x (torch.Tensor): 被搜索的tensor, 在KLObserver中是bin_edges.
        value (float): 要搜索的值. 在KLObserver中是0.

    Returns:
        int: value 右侧的值的索引, 或value本身的索引, 但是由于ObserverBase更新hist的方式, 所以0.一定不在edge上, 一定是右侧索引.
    """
    left, right = 0, len(x) - 1
    while left <= right:
        mid = (left + right) // 2
        if x[mid] < value:
            left = mid + 1
        elif x[mid] > value:
            right = mid - 1
        else:
            return mid
    return left


@torch.no_grad()
def get_kl_threshold_onedirection(
    bit_num: int, stride: int, hist: torch.Tensor, edges: torch.Tensor
):
    """只有正半轴有bin的hist进行KL最低的threshold搜索.

    Args:
        bit_num (int): 要量化的bit数. 这个是按照对称量化考虑的, 所以如果正半轴量化到127个bin, 这里应该给8.
        stride (int): 搜索步长.
        hist (torch.Tensor): 只有正半轴有值的hist.
        edges (torch.Tensor): hist对应的edges, 如果一个edge为负, 其余edges为正.

    Returns:
        torch.Tensor: threshold
    """
    quant_range = 2 ** (bit_num - 1) - 1
    start = quant_range
    if hist.numel() > 0 and hist.shape[0] > start:
        ret_device = hist.device
        edges = edges.to(ret_device)
        bin_width = edges[-1] - edges[-2]
        n_hist = hist.shape[0]
        hist[: int(n_hist * 0.001)] = 0  # optional ,让前面几个值为0
        rets = list()
        for i in range(start, n_hist + 1, stride):
            ref = torch.clone(hist[:i])
            ref[-1] += torch.sum(hist[i:])
            space = torch.linspace(0, bin_width * i, quant_range + 1, device=ret_device)
            hb2space = (
                torch.bucketize(bin_width / 2 + edges[:i], space)[None, :] - 1
            )  # (1,i)
            to_judge = torch.arange(quant_range, device=ret_device)[:, None]  # (127,1)
            mask = (hb2space == to_judge) & ((hist[:i] != 0))[None, :]
            values = hist[:i][None, :].repeat(quant_range, 1)  # (127,i)
            values[~mask] = 0
            sum_histv_perbin = torch.sum(values, dim=-1, keepdim=True)
            sum_hist_perbin = torch.sum(mask, -1, keepdim=True)
            sum_hist_perbin[sum_histv_perbin == 0] = 1
            mean = (sum_histv_perbin / sum_hist_perbin).repeat(1, i)  # (127,i)
            mean[~mask] = 0
            cand = torch.sum(mean, 0)  # (,i)
            rets.append((torch_kl_stable(cand, ref), bin_width * i))
        m = min(rets, key=lambda t: t[0])
        return m[1].to(ret_device)
    elif hist.numel() > 0:
        print(f"The amount of collected data is too small.")
        return edges[-1]
    else:
        raise ValueError("Histogram is empty!")


class HistManager:
    def __init__(self, num_bins: int) -> None:
        """初始化直方图管理器. (用于Observer的update方法.)

        Args:
            num_bins (int): 直方图的bin个数.
        """
        self.hists_mat = None
        self.bin_edges_mat = None
        self.num_bins = num_bins
        self.left_bins_num = 0
        self.right_bins_num = 0

    def collect(self, data: torch.Tensor) -> list:
        """根据data更新得到一个最新的直方图.

        Args:
            data (torch.Tensor): 新进入的tensor数据. 一定要保证传入的data应该是根据设定的dims已经做过flatten了. shape=(#channel, #elements).

        Returns:
            list: self.hists_mat, self.bin_edges_mat
        """
        assert data.dim() == 2, "batch hist_manager only support dim=2"
        data_range_perchannel = data.abs().amax(dim=-1, keepdim=True)  # (B,1)
        data_range_perchannel_mask = data_range_perchannel == 0  # unnecessary
        data_range_perchannel[data_range_perchannel_mask] = eps  # unnecessary
        if self.hists_mat is None:
            self.bin_width = data_range_perchannel / (
                self.num_bins // 2
            )  # init bin_widht : (B,1)
            self.bin_width.clip(min=eps)
            # 初始化中心0bin.
            self.hists_mat = torch.zeros(
                size=(data.shape[0], 1), device=data.device
            )  # init hists_mat : (B,1)
            self.bin_edges_mat = (
                torch.tensor([-0.5, 0.5], device=data.device).repeat(data.shape[0], 1)
                * self.bin_width
            )
        B, N = self.hists_mat.shape
        normalized_data = data / self.bin_width  # align measurement-unit
        max_bound = max(
            normalized_data.abs().amax().item(), self.hists_mat.shape[-1] // 2
        )  # (1,)
        edge_bound = math.ceil(max_bound + 0.5) - 0.5  # (1,)
        num_bins = round(edge_bound * 2)
        shift_vec = torch.arange(
            start=0, end=data.shape[0], step=1, device=data.device
        ).unsqueeze(1) * (edge_bound * 2)
        normalized_data = normalized_data + shift_vec
        hists_vec = torch.histc(
            input=normalized_data,
            bins=num_bins * B,
            min=-edge_bound,
            max=(2 * B - 1) * edge_bound,
        )
        hists_mat = hists_vec.reshape(B, -1)  # (B,Nnew)
        start_idx = int((num_bins - N) / 2)
        hists_mat[:, start_idx : start_idx + N] += self.hists_mat
        """update self.hists_mat & self.bin_edges_mat"""
        self.hists_mat = hists_mat
        self.bin_edges_mat = (
            torch.arange(
                -edge_bound, edge_bound + 0.5, step=1, device=data.device
            ).repeat(B, 1)
            * self.bin_width
        )
        self.left_bins_num = self.right_bins_num = (num_bins + 1) // 2
        """"""
        return [self.hists_mat, self.bin_edges_mat]

    def percentile(
        self, left_percent: float, right_percent: float, mode: str = "center"
    ):
        """根据参数对直方图的频度进行percentile操作.
        注意, 是对两边分别percentile同样的percent.
        这个method具体参考了PaddleSlim/paddleslim/quant/observers/hist.py中的PercentHistObserverLayer.

        Args:
            left_percent (float): 对头部做截断.保留前percent的bin, percent属于[0., 1.]
            right_percent (float): 对尾部做截断.保留前percent的bin, percent属于[0., 1.]
            mode (str, optional): 以percent落在的bin的中线为界(center)还是线性插值(line). Defaults to "center".

        Returns:
            torch.Tensor, torch.Tensor: 两个tensor, 表示clip的min和max界限.
        """
        assert (0.0 <= left_percent <= 1.0) and (
            0.0 <= right_percent <= 1.0
        ), "Error Percent setting."
        assert mode in ["center", "line"], "Only support center or line."
        if mode == "center":
            centers = (
                self.bin_edges_mat[..., 1:] + self.bin_edges_mat[..., :-1]
            ) / 2  # 算出每个bin的中线.
            min_clip_bound = centers[..., 0]
            max_clip_bound = centers[..., -1]
            # 从左向右, 给出截掉max端的percent的clip_value.
            max_clip_bound = percentile_center_onedirection(
                hists_mat=self.hists_mat[..., (self.left_bins_num - 1) :],
                centers=centers[..., (self.left_bins_num - 1) :],
                percent=right_percent,
            )
            # 从右向左, 给出截掉min端的percent的clip_value.
            min_clip_bound = percentile_center_onedirection(
                hists_mat=torch.flip(
                    self.hists_mat[..., : self.left_bins_num], dims=(-1,)
                ),
                centers=torch.flip(centers[..., : self.left_bins_num], dims=(-1,)),
                percent=left_percent,
            )
            return min_clip_bound, max_clip_bound
        else:
            min_clip_bound = self.bin_edges_mat[..., 0]
            max_clip_bound = self.bin_edges_mat[..., -1]
            # 从左向右, 给出截掉max端的percent的clip_value.
            max_clip_bound = percentile_linear_onedirection(
                hists_mat=self.hists_mat[..., (self.left_bins_num - 1) :],
                bin_edges_mat=self.bin_edges_mat[..., (self.left_bins_num - 1) :],
                percent=right_percent,
            )
            min_clip_bound = percentile_linear_onedirection(
                hists_mat=torch.flip(
                    self.hists_mat[..., : self.left_bins_num], dims=(-1,)
                ),
                bin_edges_mat=torch.flip(
                    self.bin_edges_mat[..., : (self.left_bins_num + 1)], dims=(-1,)
                ),
                percent=left_percent,
            )
            return min_clip_bound, max_clip_bound

    def find_lowest_kl_bound(self, bit_num: int, stride: int):
        """搜索出KL最小的min_bound, max_bound.
        1. 以0.所在bin为起点, 分别向左向右搜索能够使得KL散度最小的thresholds.
        2. 两个thresholds拼起来作为min_bound, max_bound.

        Args:
            bit_num (int): 量化位数.
            stride (int): 搜索步长.
        Returns:
            torch.Tensor, torch.Tensor: 搜索得到的min_bound和max_bound.
        """
        # 找到搜索的起点bin.
        # 由于搜集方式的原因, 我们一定有0.在某个bin的中线, 这个bin的左侧是负edge, 右侧是正edge.
        right_edge_idx = binary_search(x=self.bin_edges_mat, value=0.0)
        left_edge_idx = right_edge_idx - 1
        pos_bin_edges = self.bin_edges_mat[left_edge_idx:].clone()
        pos_hist = self.hists_mat[-(len(pos_bin_edges) - 1) :].clone()
        neg_bin_edges = self.bin_edges_mat[: right_edge_idx + 1].clone()
        neg_hist = self.hists_mat[: len(neg_bin_edges) - 1].clone()
        right_threshold = get_kl_threshold_onedirection(
            bit_num=bit_num, stride=stride, hist=pos_hist, edges=pos_bin_edges
        )
        left_threshold = get_kl_threshold_onedirection(
            bit_num=bit_num,
            stride=stride,
            hist=reversed(neg_hist),
            edges=-reversed(neg_bin_edges),
        )
        left_threshold = -left_threshold
        return left_threshold, right_threshold


if __name__ == "__main__":
    from hmquant.ptq.nn_layers.observers.utils import gen_rand_feat

    num_bins = 12
    bin_width_tensor = torch.ones(size=(4, 4), device="cuda").float()
    hist_manager1 = HistManager(num_bins=num_bins)
    hist_manager1.manual_init_binwidth(bin_width=bin_width_tensor)
    hist_manager2 = HistManager(num_bins=num_bins)
    hist_manager2.manual_init_binwidth(bin_width=bin_width_tensor)
    hist_manager3 = HistManager(num_bins=num_bins)
    hist_manager3.manual_init_binwidth(bin_width=bin_width_tensor)
    feat = gen_rand_feat(
        min=3,
        max=12,
        batch_size=4,
        feat_chs=4,
        feat_height=4,
        feat_width=4,
        device="cuda",
    )
    feat_2dims = feat.flatten(start_dim=2)
    print(feat_2dims)
    hists, edges = hist_manager1.collect(feat_2dims)
    print("hist 1:")
    print(hists)
    print("edges 1")
    print(edges)
    feat = gen_rand_feat(
        min=-5,
        max=10,
        batch_size=4,
        feat_chs=4,
        feat_height=4,
        feat_width=4,
        device="cuda",
    )
    feat_2dims = feat.flatten(start_dim=2)
    print(feat_2dims)
    hists, edges = hist_manager1.collect(feat_2dims)
    print("hist 2")
    print(hists)
    print("edges 2")
    print(edges)
    feat = gen_rand_feat(
        min=-3,
        max=12,
        batch_size=4,
        feat_chs=4,
        feat_height=4,
        feat_width=4,
        device="cuda",
    )
    feat_2dims = feat.flatten(start_dim=2)
    print(feat_2dims)
    hists, edges = hist_manager2.collect(feat_2dims)
    print("hist 3")
    print(hists)
    print("edges 3")
    print(edges)
    feat = gen_rand_feat(
        min=-7,
        max=-1,
        batch_size=4,
        feat_chs=4,
        feat_height=4,
        feat_width=4,
        device="cuda",
    )
    feat_2dims = feat.flatten(start_dim=2)
    print(feat_2dims)
    hists, edges = hist_manager2.collect(feat_2dims)
    print("hist 4")
    print(hists)
    print("edges 4")
    print(edges)
    feat = gen_rand_feat(
        min=-3,
        max=6,
        batch_size=4,
        feat_chs=4,
        feat_height=4,
        feat_width=4,
        device="cuda",
    )
    feat_2dims = feat.flatten(start_dim=2)
    print(feat_2dims)
    hists, edges = hist_manager3.collect(feat_2dims)
    print("hist 5")
    print(hists)
    print("edges 5")
    print(edges)
    feat = gen_rand_feat(
        min=-7,
        max=12,
        batch_size=4,
        feat_chs=4,
        feat_height=4,
        feat_width=4,
        device="cuda",
    )
    feat_2dims = feat.flatten(start_dim=2)
    print(feat_2dims)
    hists, edges = hist_manager3.collect(feat_2dims)
    print("hist 6")
    print(hists)
    print("edges 6")
    print(edges)
    # from hmquant.ptq.nn_layers.speed_observers.utils import merge_hists_of_observers # TODO fix it

    merge_hists_of_observers(
        hist_manager_list=[hist_manager1, hist_manager2, hist_manager3]
    )

    # clip_value1 = hist_manager3.percentile(left_percent=0.7895, right_percent=0.95, mode="center")
    # print(clip_value1)
    # clip_value2 = hist_manager3.percentile(left_percent=0.7895, right_percent=0.95, mode="line")
    # print(clip_value2)
