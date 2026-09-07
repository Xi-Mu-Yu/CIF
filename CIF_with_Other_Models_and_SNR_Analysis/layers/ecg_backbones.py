import math
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.basic_conv1d import create_head1d


def conv(in_planes, out_planes, kernel_size=3, stride=1):
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=(kernel_size - 1) // 2,
        bias=False,
    )


def noop(x):
    return x


class InceptionBlock1d(nn.Module):
    def __init__(self, ni, nb_filters, kss, stride=1, bottleneck_size=32):
        super().__init__()
        self.bottleneck = conv(ni, bottleneck_size, 1, stride) if bottleneck_size > 0 else noop
        self.convs = nn.ModuleList(
            [conv(bottleneck_size if bottleneck_size > 0 else ni, nb_filters, ks) for ks in kss]
        )
        self.conv_bottle = nn.Sequential(nn.MaxPool1d(3, stride, padding=1), conv(ni, nb_filters, 1))
        self.bn_relu = nn.Sequential(nn.BatchNorm1d((len(kss) + 1) * nb_filters), nn.ReLU())

    def forward(self, x):
        bottled = self.bottleneck(x)
        out = self.bn_relu(torch.cat([c(bottled) for c in self.convs] + [self.conv_bottle(x)], dim=1))
        return out


class Shortcut1d(nn.Module):
    def __init__(self, ni, nf):
        super().__init__()
        self.act_fn = nn.ReLU(True)
        self.conv = conv(ni, nf, 1)
        self.bn = nn.BatchNorm1d(nf)

    def forward(self, inp, out):
        return self.act_fn(out + self.bn(self.conv(inp)))


class InceptionBackbone(nn.Module):
    def __init__(self, input_channels, kss, depth, bottleneck_size, nb_filters, use_residual):
        super().__init__()
        self.depth = depth
        assert depth % 3 == 0
        self.use_residual = use_residual
        n_ks = len(kss) + 1
        self.im = nn.ModuleList(
            [
                InceptionBlock1d(
                    input_channels if d == 0 else n_ks * nb_filters,
                    nb_filters=nb_filters,
                    kss=kss,
                    bottleneck_size=bottleneck_size,
                )
                for d in range(depth)
            ]
        )
        self.sk = nn.ModuleList(
            [
                Shortcut1d(input_channels if d == 0 else n_ks * nb_filters, n_ks * nb_filters)
                for d in range(depth // 3)
            ]
        )

    def forward(self, x):
        input_res = x
        for d in range(self.depth):
            x = self.im[d](x)
            if self.use_residual and d % 3 == 2:
                x = self.sk[d // 3](input_res, x)
                input_res = x.clone()
        return x


class Inception1dBackbone(nn.Module):
    def __init__(
        self,
        num_classes,
        input_channels,
        kernel_size=40,
        depth=6,
        bottleneck_size=32,
        nb_filters=32,
        use_residual=True,
        ps_head=0.5,
        concat_pooling=True,
    ):
        super().__init__()
        kernel_size = max(15, min(kernel_size, 40))
        if kernel_size % 2 == 0:
            kernel_size -= 1
        kss = [kernel_size, max(kernel_size // 2, 7), max(kernel_size // 4, 3)]
        kss = [k if k % 2 == 1 else k - 1 for k in kss]
        n_ks = len(kss) + 1
        self.backbone = InceptionBackbone(
            input_channels=input_channels,
            kss=kss,
            depth=depth,
            bottleneck_size=bottleneck_size,
            nb_filters=nb_filters,
            use_residual=use_residual,
        )
        self.head = create_head1d(
            n_ks * nb_filters,
            nc=num_classes,
            ps=ps_head,
            concat_pooling=concat_pooling,
        )

    def forward(self, x):
        return self.head(self.backbone(x))


class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, kernel_size=(3, 3), downsample=None):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size // 2 + 1]
        self.conv1 = conv(inplanes, planes, stride=stride, kernel_size=kernel_size[0])
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv(planes, planes, kernel_size=kernel_size[1])
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class ResNet1dBackbone(nn.Sequential):
    def __init__(
        self,
        block,
        layers,
        kernel_size=3,
        num_classes=2,
        input_channels=3,
        inplanes=64,
        fix_feature_dim=True,
        kernel_size_stem=None,
        stride_stem=2,
        pooling_stem=True,
        stride=2,
        ps_head=0.5,
        concat_pooling=True,
    ):
        self.inplanes = inplanes
        layers_tmp = []
        if kernel_size_stem is None:
            kernel_size_stem = kernel_size[0] if isinstance(kernel_size, list) else kernel_size
        layers_tmp.append(
            nn.Conv1d(
                input_channels,
                inplanes,
                kernel_size=kernel_size_stem,
                stride=stride_stem,
                padding=(kernel_size_stem - 1) // 2,
                bias=False,
            )
        )
        layers_tmp.append(nn.BatchNorm1d(inplanes))
        layers_tmp.append(nn.ReLU(inplace=True))
        if pooling_stem:
            layers_tmp.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        for i, l in enumerate(layers):
            if i == 0:
                layers_tmp.append(self._make_layer(block, inplanes, layers[0], kernel_size=kernel_size))
            else:
                layers_tmp.append(
                    self._make_layer(
                        block,
                        inplanes if fix_feature_dim else (2**i) * inplanes,
                        layers[i],
                        stride=stride,
                        kernel_size=kernel_size,
                    )
                )
        head = create_head1d(
            (inplanes if fix_feature_dim else (2 ** len(layers)) * inplanes) * block.expansion,
            nc=num_classes,
            ps=ps_head,
            concat_pooling=concat_pooling,
        )
        layers_tmp.append(head)
        super().__init__(*layers_tmp)

    def _make_layer(self, block, planes, blocks, stride=1, kernel_size=3):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layer_list = [block(self.inplanes, planes, stride, kernel_size, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layer_list.append(block(self.inplanes, planes))
        return nn.Sequential(*layer_list)


def resnet1d18(**kwargs):
    return ResNet1dBackbone(BasicBlock1d, [2, 2, 2, 2], **kwargs)


def resnet1d_wang(**kwargs):
    if "kernel_size" not in kwargs:
        kwargs["kernel_size"] = [5, 3]
    if "kernel_size_stem" not in kwargs:
        kwargs["kernel_size_stem"] = 7
    if "stride_stem" not in kwargs:
        kwargs["stride_stem"] = 1
    if "pooling_stem" not in kwargs:
        kwargs["pooling_stem"] = False
    if "inplanes" not in kwargs:
        kwargs["inplanes"] = 128
    return ResNet1dBackbone(BasicBlock1d, [1, 1, 1], **kwargs)


NormType = Enum("NormType", "Batch BatchZero Weight Spectral Instance InstanceZero")


def _conv_func(ndim=2, transpose=False):
    return getattr(nn, f'Conv{"Transpose" if transpose else ""}{ndim}d')


def init_default(m, func=nn.init.kaiming_normal_):
    if func and hasattr(m, "weight"):
        func(m.weight)
    with torch.no_grad():
        if getattr(m, "bias", None) is not None:
            m.bias.fill_(0.0)
    return m


def _get_norm(prefix, nf, ndim=2, zero=False, **kwargs):
    bn = getattr(nn, f"{prefix}{ndim}d")(nf, **kwargs)
    if bn.affine:
        bn.bias.data.fill_(1e-3)
        bn.weight.data.fill_(0.0 if zero else 1.0)
    return bn


def BatchNorm(nf, ndim=2, norm_type=NormType.Batch, **kwargs):
    return _get_norm("BatchNorm", nf, ndim, zero=norm_type == NormType.BatchZero, **kwargs)


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        ni,
        nf,
        ks=3,
        stride=1,
        padding=None,
        bias=None,
        ndim=2,
        norm_type=NormType.Batch,
        bn_1st=True,
        act_cls=nn.ReLU,
        transpose=False,
        init=nn.init.kaiming_normal_,
        xtra=None,
        **kwargs,
    ):
        if padding is None:
            padding = (ks - 1) // 2 if not transpose else 0
        bn = norm_type in (NormType.Batch, NormType.BatchZero)
        if bias is None:
            bias = not bn
        conv_func = _conv_func(ndim, transpose=transpose)
        conv = init_default(
            conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding, **kwargs),
            init,
        )
        layers = [conv]
        act_bn = []
        if act_cls is not None:
            act_bn.append(act_cls())
        if bn:
            act_bn.append(BatchNorm(nf, norm_type=norm_type, ndim=ndim))
        if bn_1st:
            act_bn.reverse()
        layers += act_bn
        if xtra:
            layers.append(xtra)
        super().__init__(*layers)


def AvgPool(ks=2, stride=None, padding=0, ndim=2, ceil_mode=False):
    return getattr(nn, f"AvgPool{ndim}d")(ks, stride=stride, padding=padding, ceil_mode=ceil_mode)


class ResBlock(nn.Module):
    def __init__(
        self,
        expansion,
        ni,
        nf,
        stride=1,
        kernel_size=3,
        groups=1,
        reduction=None,
        nh1=None,
        nh2=None,
        dw=False,
        g2=1,
        sa=False,
        sym=False,
        norm_type=NormType.Batch,
        act_cls=nn.ReLU,
        ndim=2,
        pool=AvgPool,
        pool_first=True,
        **kwargs,
    ):
        super().__init__()
        norm2 = (
            NormType.BatchZero
            if norm_type == NormType.Batch
            else NormType.InstanceZero
            if norm_type == NormType.Instance
            else norm_type
        )
        if nh2 is None:
            nh2 = nf
        if nh1 is None:
            nh1 = nh2
        nf, ni = nf * expansion, ni * expansion
        k0 = dict(norm_type=norm_type, act_cls=act_cls, ndim=ndim, **kwargs)
        k1 = dict(norm_type=norm2, act_cls=None, ndim=ndim, **kwargs)
        if expansion == 1:
            layers = [
                ConvLayer(ni, nh2, kernel_size, stride=stride, groups=ni if dw else groups, **k0),
                ConvLayer(nh2, nf, kernel_size, groups=g2, **k1),
            ]
        else:
            layers = [
                ConvLayer(ni, nh1, 1, **k0),
                ConvLayer(nh1, nh2, kernel_size, stride=stride, groups=nh1 if dw else groups, **k0),
                ConvLayer(nh2, nf, 1, groups=g2, **k1),
            ]
        self.convs = nn.Sequential(*layers)
        self.convpath = nn.Sequential(self.convs)
        idpath = []
        if ni != nf:
            idpath.append(ConvLayer(ni, nf, 1, act_cls=None, ndim=ndim, **kwargs))
        if stride != 1:
            idpath.insert((1, 0)[pool_first], pool(2, ndim=ndim, ceil_mode=True))
        self.idpath = nn.Sequential(*idpath)
        self.act = nn.ReLU(inplace=True) if act_cls is nn.ReLU else act_cls()

    def forward(self, x):
        return self.act(self.convpath(x) + self.idpath(x))


def init_cnn(m):
    if getattr(m, "bias", None) is not None:
        nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
    for l in m.children():
        init_cnn(l)


class XResNet1dBackbone(nn.Sequential):
    def __init__(
        self,
        block,
        expansion,
        layers,
        p=0.0,
        input_channels=3,
        num_classes=1000,
        stem_szs=(32, 32, 64),
        kernel_size=5,
        kernel_size_stem=5,
        widen=1.0,
        sa=False,
        act_cls=nn.ReLU,
        ps_head=0.5,
        concat_pooling=True,
        **kwargs,
    ):
        self.block = block
        self.expansion = expansion
        self.act_cls = act_cls
        stem_szs = [input_channels, *stem_szs]
        stem = [
            ConvLayer(
                stem_szs[i],
                stem_szs[i + 1],
                ks=kernel_size_stem,
                stride=2 if i == 0 else 1,
                act_cls=act_cls,
                ndim=1,
            )
            for i in range(3)
        ]
        block_szs = [int(o * widen) for o in [64, 64, 64, 64] + [32] * (len(layers) - 4)]
        block_szs = [64 // expansion] + block_szs
        blocks = [
            self._make_layer(
                ni=block_szs[i],
                nf=block_szs[i + 1],
                blocks=l,
                stride=1 if i == 0 else 2,
                kernel_size=kernel_size,
                sa=sa and i == len(layers) - 4,
                ndim=1,
                **kwargs,
            )
            for i, l in enumerate(layers)
        ]
        head = create_head1d(
            block_szs[-1] * expansion,
            nc=num_classes,
            ps=ps_head,
            concat_pooling=concat_pooling,
        )
        super().__init__(*stem, nn.MaxPool1d(kernel_size=3, stride=2, padding=1), *blocks, head)
        init_cnn(self)

    def _make_layer(self, ni, nf, blocks, stride, kernel_size, sa, **kwargs):
        return nn.Sequential(
            *[
                self.block(
                    self.expansion,
                    ni if i == 0 else nf,
                    nf,
                    stride=stride if i == 0 else 1,
                    kernel_size=kernel_size,
                    sa=sa and i == (blocks - 1),
                    act_cls=self.act_cls,
                    **kwargs,
                )
                for i in range(blocks)
            ]
        )


def xresnet1d18(**kwargs):
    return XResNet1dBackbone(ResBlock, 1, [2, 2, 2, 2], **kwargs)
