# -*- coding: utf-8 -*-
"""
#!/usr/bin/env python3
Created on Tue Aug 20 11:04:00 2024

@author: parteeksj
"""
import sys

sys.path.append("../")

import torch
from torch import nn
from typing import Tuple
import torch.nn.functional as F
import math
from models.mae import MaskedAutoencoderViT

# copy from detectron2
class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


# copy from detectron2
class Conv_LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class FPN(nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        scale_factors: Tuple[float, ...] = (4.0, 2.0, 1.0),
        decoder_embed_dim: int = 1024,
        FPN_output_dim: Tuple[int, ...] = (256, 512, 1024),
    ):
        super().__init__()

        # for scale = 4, 2, 1
        strides = [int(patch_size / scale) for scale in scale_factors]  # [4, 8, 16]
        self.stages = []
        use_bias = False

        for idx, scale in enumerate(scale_factors):
            out_dim = decoder_embed_dim
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(
                        decoder_embed_dim,
                        decoder_embed_dim // 2,
                        kernel_size=2,
                        stride=2,
                    ),
                    Conv_LayerNorm(decoder_embed_dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(
                        decoder_embed_dim // 2,
                        decoder_embed_dim // 4,
                        kernel_size=2,
                        stride=2,
                    ),
                ]
                out_dim = decoder_embed_dim // 4
            elif scale == 2.0:
                layers = [
                    nn.ConvTranspose2d(
                        decoder_embed_dim,
                        decoder_embed_dim // 2,
                        kernel_size=2,
                        stride=2,
                    )
                ]
                out_dim = decoder_embed_dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            layers.extend(
                [
                    Conv2d(
                        out_dim,
                        FPN_output_dim[idx],
                        kernel_size=1,
                        bias=use_bias,
                        norm=Conv_LayerNorm(FPN_output_dim[idx]),
                    ),
                    Conv2d(
                        FPN_output_dim[idx],
                        FPN_output_dim[idx],
                        kernel_size=3,
                        padding=1,
                        bias=use_bias,
                        norm=Conv_LayerNorm(FPN_output_dim[idx]),
                    ),
                ]
            )
            layers = nn.Sequential(*layers)

            stage = int(math.log2(strides[idx]))
            self.add_module(f"simfp_{stage}", layers)
            self.stages.append(layers)

    def forward(self, x_: torch.Tensor, ids_restore: torch.Tensor):
        h = w = int(x_.shape[1] ** 0.5)  # x.shape = [B,N,decoder_embed_dim]
        decoder_dim = x_.shape[2]

        # [2, 196, 768] -> [2, 768, 14, 14]
        x = x_.permute(0, 2, 1).view(-1, decoder_dim, h, w)  # (B, channel, h, w)
        results = []

        for idx, stage in enumerate(self.stages):
            stage_feature_map = stage(x)
            results.append(stage_feature_map)

        return {
            layer: feature
            for layer, feature in zip(
                ["layer1", "layer2", "layer3"],
                results,
            )
        }


if __name__ == "__main__":
    mae = MaskedAutoencoderViT()
    fpn = FPN()

    test_input = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        pred, ids_restore = mae(test_input)
        op_dict = fpn(pred, ids_restore)

    print("END")
