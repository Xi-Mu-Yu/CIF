import math
from typing import Collection, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

Floats = Union[float, Sequence[float]]


def listify(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def bn_drop_lin(ni, no, bn=True, p=0.0, actn=None):
    layers = []
    if bn:
        layers.append(nn.BatchNorm1d(ni))
    if p > 0:
        layers.append(nn.Dropout(p))
    layers.append(nn.Linear(ni, no))
    if actn is not None:
        layers.append(actn)
    return layers


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def _conv1d(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, act="relu", bn=True, drop_p=0):
    lst = []
    if drop_p > 0:
        lst.append(nn.Dropout(drop_p))
    lst.append(
        nn.Conv1d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            dilation=dilation,
            bias=not bn,
        )
    )
    if bn:
        lst.append(nn.BatchNorm1d(out_planes))
    if act == "relu":
        lst.append(nn.ReLU(True))
    elif act == "elu":
        lst.append(nn.ELU(True))
    elif act == "prelu":
        lst.append(nn.PReLU(True))
    return nn.Sequential(*lst)


class AdaptiveConcatPool1d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or 1
        self.ap = nn.AdaptiveAvgPool1d(sz)
        self.mp = nn.AdaptiveMaxPool1d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class SqueezeExcite1d(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        channels_reduced = channels // reduction
        self.w1 = nn.Parameter(torch.randn(channels_reduced, channels).unsqueeze(0))
        self.w2 = nn.Parameter(torch.randn(channels, channels_reduced).unsqueeze(0))

    def forward(self, x):
        z = torch.mean(x, dim=2, keepdim=True)
        intermed = F.relu(torch.matmul(self.w1, z))
        s = F.sigmoid(torch.matmul(self.w2, intermed))
        return s * x


def create_head1d(
    nf,
    nc,
    lin_ftrs=None,
    ps=0.5,
    bn_final=False,
    bn=True,
    act="relu",
    concat_pooling=True,
):
    lin_ftrs = [2 * nf if concat_pooling else nf, nc] if lin_ftrs is None else [2 * nf if concat_pooling else nf] + lin_ftrs + [nc]
    ps = listify(ps)
    if len(ps) == 1:
        ps = [ps[0] / 2] * (len(lin_ftrs) - 2) + ps
    actns = [nn.ReLU(inplace=True) if act == "relu" else nn.ELU(inplace=True)] * (len(lin_ftrs) - 2) + [None]
    layers = [AdaptiveConcatPool1d() if concat_pooling else nn.MaxPool1d(2), Flatten()]
    for ni, no, p, actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, actns):
        layers += bn_drop_lin(ni, no, bn, p, actn)
    if bn_final:
        layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    return nn.Sequential(*layers)
