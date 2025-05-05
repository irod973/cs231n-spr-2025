import argparse

import torchvision.models as models
from loguru import logger

# import argparse
# import os
# import random
# import shutil
# import time
# import warnings
# from enum import Enum
#
# import torch
# import torch.backends.cudnn as cudnn
# import torch.distributed as dist
# import torch.multiprocessing as mp
# import torch.nn as nn
# import torch.nn.parallel
# import torch.optim
# import torch.utils.data
# import torch.utils.data.distributed
# import torchvision.datasets as datasets
# import torchvision.models as models
# import torchvision.transforms as transforms
# from torch.optim.lr_scheduler import StepLR
# from torch.utils.data import Subset


model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)
model_names_str = " | ".join(model_names)
logger.debug(f"Supported models: {model_names_str}")

def parse_args() -> argparse.Namespace:
    """Process cli arguments"""
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

    # Model architecture
    parser.add_argument(
        "-a",
        "--model-arch",
        metavar="ARCH",
        default="resnet18",
        choices=model_names,
        help=f"Model architecture, default resnet18, options include {model_names_str}",
    )
    parser.add_argument(
        "--pretrained", dest="pretrained", action="store_true", help="Use pre-trained model"
    )

    # Workers
    parser.add_argument(
        "-j",
        "-data-workers",
        default=4,
        type=int,
        metavar="N",
        help="Number of data loading workers (default: 4)",
    )

    # Epochs
    parser.add_argument(
        "--epochs", default=90, type=int, metavar="N", help="number of total epochs to run"
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )

    # Hyperparams
    parser.add_argument(
        "-b",
        "--batch-size",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256), this is the total "
             "batch size of all GPUs on the current node when "
             "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )

    # Distributed training
    parser.add_argument(
        "--world-size", default=-1, type=int, help="number of nodes for distributed training"
    )
    parser.add_argument("--rank", default=-1, type=int, help="node rank for distributed training")
    parser.add_argument(
        "--dist-url",
        default="tcp://224.66.41.62:23456",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--seed", default=None, type=int, help="seed for initializing training. ")
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    parser.add_argument(
        "--multiprocessing-distributed",
        action="store_true",
        help="Use multi-processing distributed training to launch "
        "N processes per node, which has N GPUs. This is the "
        "fastest way to use PyTorch for either single node or "
        "multi node data parallel training",
    )

    # Dummy data
    parser.add_argument("--dummy", action="store_true", help="use fake data to benchmark")

    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parse_args()

