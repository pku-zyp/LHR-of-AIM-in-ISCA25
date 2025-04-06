import sys

sys.path.append(".")
import torch, timm
import argparse
import os
import random
import shutil
import time, datetime
import warnings
import torch, torch.nn as nn, torch.fx as fx, torch.nn.functional as F, operator
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from torch.fx.graph import Graph
from torch.fx.node import Node
from torch.fx.graph_module import GraphModule
from loguru import logger

logger.remove()
logger.add(sys.stdout, format="{message}")

# from transform_net import transform_to_qnet
from qmodules import transform_to_qnet, QConv, QLinear, QMZBConv, QMZBLinear

# from mzb_quant import calc_mean_zb_ratio
from qmodules.hamming import HammingLoss

best_acc1 = 0


def get_vit(x_bit=8, w_bit=8):
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    model = fx._symbolic_trace.symbolic_trace(model)
    node: Node
    model: GraphModule
    # replace all matmul nodes
    for node in model.graph.nodes:
        if node.op == "call_function":
            func = node.target
            if func == torch.matmul or func == operator.matmul:
                node.op = "call_module"
                model.add_submodule(node.name, QMatMul(x_bit=x_bit))
                node.target = node.name
    model.graph.eliminate_dead_code()
    model.graph.lint()
    model.recompile()
    # replace all linear nodes

    for k, v in model.named_modules():
        if isinstance(v, nn.Linear):
            model.add_submodule(k, QMZBLinear(v, w_bit=w_bit, x_bit=x_bit))
    return model


def main():
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
    parser.add_argument("--data", type=str, default="data/imagenet")
    parser.add_argument("--w_bit", default=8, type=int)
    parser.add_argument("--x_bit", default=8, type=int)
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=1e-5,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=8,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=128,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256), this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distribu Data Parallel",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=30,
        type=int,
        metavar="N",
        help="print frequency",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument("--quad", type=bool, default=True)
    parser.add_argument("--ratio", default=0.01, type=int)
    args = parser.parse_args()
    args.gpu = 0
    args.seed = 2023
    args.start_epoch = 0
    global best_acc1
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    model = get_vit(w_bit=args.w_bit, x_bit=args.x_bit).cuda(args.gpu)
    work_dir = f"output/vit_mzb_w{args.w_bit}_a{args.x_bit}{time.strftime('%Y%m%d%H%M', time.localtime())}"
    logger.add(f"{work_dir}/log.txt", format="{message}")
    os.makedirs(work_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss().cuda()
    from torch.optim.lr_scheduler import LambdaLR
    import math

    warmup_iterations = 100
    total_iterations = 10000 * args.epochs

    def lr_lambda(current_iteration):
        if current_iteration < warmup_iterations:
            # print(f"lr warmup {current_iteration}/{warmup_iterations}")
            return float(current_iteration) / float(max(1, warmup_iterations))
        progress = float(current_iteration - warmup_iterations) / float(
            max(1, total_iterations - warmup_iterations)
        )
        lr_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        # print(f"lr decay {lr_decay}")
        return lr_decay

    optimizer = torch.optim.Adam(
        model.parameters(),
        args.lr,
        # weight_decay=args.weight_decay,
    )
    scheduler = LambdaLR(optimizer, lr_lambda)
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            best_acc1 = checkpoint["best_acc1"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))
    traindir = os.path.join(args.data, "train")
    valdir = os.path.join(args.data, "val")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    writer = SummaryWriter("output/runs/{}_{}".format("vit", int(time.time())))
    writer.global_steps = 0

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            valdir,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    save_init(model, val_loader, work_dir)
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, scheduler, epoch, args, writer)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_acc1": best_acc1,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            filename=f"{work_dir}/epoch_{epoch}.pth.tar",
        )


@torch.enable_grad()
def train(train_loader, model, criterion, optimizer, scheduler, epoch, args, writer):
    # reguralizar
    conv_modules = [m for m in model.modules() if isinstance(m, (QConv, QLinear))]
    tot_nums = sum([m.weight.numel() for m in conv_modules])
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    losseshamming = AverageMeter("hamming", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, losseshamming, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    hamming = HammingLoss().cuda(args.gpu)

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        optimizer.zero_grad()
        # compute
        loss = criterion(output, target)
        loss_item = loss.item()
        if args.quad:
            hamming_loss = 0
            layer_max_hamming = torch.tensor(0.0, device=loss.device)
            for module in conv_modules:
                if module.calibrated:
                    per_layer_mean_hamming = hamming(
                        module.weight / module.w_scale, reduce="mean"
                    )
                    layer_max_hamming = torch.max(
                        per_layer_mean_hamming, layer_max_hamming
                    )
                    hamming_loss += per_layer_mean_hamming**2
            loss += hamming_loss * args.ratio
            hamming_show = layer_max_hamming.item()
        else:
            hamming_loss = 0
            for module in conv_modules:
                if module.calibrated:
                    hamming_loss += hamming(module.weight / module.w_scale) / tot_nums
            loss += hamming_loss * 0.1
            hamming_show = hamming_loss.item()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss_item, images.size(0))
        losseshamming.update(hamming_show)
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        loss.backward()
        # grad clip to avoid exploding gradient
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        scheduler.step()

        writer.add_scalar("Loss/train", loss.item(), writer.global_steps)
        writer.global_steps += 1

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            # breakpoint()


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        logger.info(
            " * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5)
        )

    return top1.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        return
        shutil.copyfile(filename, "model_best.pth.tar")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


@torch.no_grad()
def save_init(model, loader, work_dir: str):
    for data, label in loader:
        data = data.cuda()
        model(data)
        break
    save_checkpoint(
        {
            "epoch": 0,
            "state_dict": model.state_dict(),
            "best_acc1": None,
        },
        False,
        filename=f"{work_dir}/init.pth.tar",
    )


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


if __name__ == "__main__":
    main()
