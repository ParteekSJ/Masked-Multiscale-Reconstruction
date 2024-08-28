import sys

sys.path.append("../")

import torch
from torch import nn
from fvcore.common.config import CfgNode
from models.mae import MaskedAutoencoderViT
from models.fpn import FPN
from config.default import get_cfg


class MMR(nn.Module):
    def __init__(self, cfg: CfgNode):
        super().__init__()

        self.mae = MaskedAutoencoderViT(
            img_size=cfg.MODEL.image_size,
            patch_size=cfg.MODEL.patch_size,
            in_channels=cfg.MODEL.in_channels,
            embed_dim=cfg.MODEL.embed_dim,
            depth=cfg.MODEL.depth,
            num_heads=cfg.MODEL.num_heads,
            mlp_ratio=cfg.MODEL.mlp_ratio,
            norm_layer=nn.LayerNorm,
        )
        self.fpn = FPN(
            patch_size=cfg.MODEL.patch_size,
            scale_factors=cfg.MODEL.scale_factors,
            decoder_embed_dim=cfg.MODEL.embed_dim,
            FPN_output_dim=cfg.MODEL.fpn_output_dim,
        )

        """
        self.apply(self._init_weights): This won't be applied to the standalone models we've
        defined unless we call the method explicitly on those models.
        
        `apply` method would initialize weights of the MMR model itself.
        """

        # initialize nn.Linear and nn.LayerNorm
        self.mae.apply(self._init_weights)
        self.fpn.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask_ratio):
        pred, ids_restore = self.mae(x, mask_ratio)
        op_dict = self.fpn(pred, ids_restore)
        return op_dict


if __name__ == "__main__":
    cfg = get_cfg()
    mmr_model = MMR(cfg)
    test_input = torch.randn(8, 3, 224, 224)

    with torch.no_grad():
        op_dict = mmr_model(test_input, 0.4)

    for x in op_dict.keys():
        print(f"{op_dict[x].shape=}")


"""

===============================================================================================
Layer (type:depth-idx)                        Input Shape               Output Shape
===============================================================================================
MMR                                           [1, 3, 224, 224]          [1, 1024, 14, 14]
├─MaskedAutoencoderViT: 1-1                   [1, 3, 224, 224]          [1, 196, 1024]
│    └─PatchEmbed: 2-1                        [1, 3, 224, 224]          [1, 196, 1024]
│    │    └─Conv2d: 3-1                       [1, 3, 224, 224]          [1, 1024, 14, 14]
│    │    └─Identity: 3-2                     [1, 196, 1024]            [1, 196, 1024]
│    └─ModuleList: 2-2                        --                        --
│    │    └─Block: 3-3                        [1, 50, 1024]             [1, 50, 1024]
│    │    └─Block: 3-4                        [1, 50, 1024]             [1, 50, 1024]
│    │    └─Block: 3-5                        [1, 50, 1024]             [1, 50, 1024]
│    │    └─Block: 3-6                        [1, 50, 1024]             [1, 50, 1024]
│    │    └─Block: 3-7                        [1, 50, 1024]             [1, 50, 1024]
│    │    └─Block: 3-8                        [1, 50, 1024]             [1, 50, 1024]
│    │    └─Block: 3-9                        [1, 50, 1024]             [1, 50, 1024]
│    │    └─Block: 3-10                       [1, 50, 1024]             [1, 50, 1024]
│    │    └─Block: 3-11                       [1, 50, 1024]             [1, 50, 1024]
│    │    └─Block: 3-12                       [1, 50, 1024]             [1, 50, 1024]
│    │    └─Block: 3-13                       [1, 50, 1024]             [1, 50, 1024]
│    │    └─Block: 3-14                       [1, 50, 1024]             [1, 50, 1024]
│    │    └─Block: 3-15                       [1, 50, 1024]             [1, 50, 1024]
│    │    └─Block: 3-16                       [1, 50, 1024]             [1, 50, 1024]
│    │    └─Block: 3-17                       [1, 50, 1024]             [1, 50, 1024]
│    │    └─Block: 3-18                       [1, 50, 1024]             [1, 50, 1024]
│    │    └─Block: 3-19                       [1, 50, 1024]             [1, 50, 1024]
│    │    └─Block: 3-20                       [1, 50, 1024]             [1, 50, 1024]
│    │    └─Block: 3-21                       [1, 50, 1024]             [1, 50, 1024]
│    │    └─Block: 3-22                       [1, 50, 1024]             [1, 50, 1024]
│    │    └─Block: 3-23                       [1, 50, 1024]             [1, 50, 1024]
│    │    └─Block: 3-24                       [1, 50, 1024]             [1, 50, 1024]
│    │    └─Block: 3-25                       [1, 50, 1024]             [1, 50, 1024]
│    │    └─Block: 3-26                       [1, 50, 1024]             [1, 50, 1024]
│    └─LayerNorm: 2-3                         [1, 50, 1024]             [1, 50, 1024]
├─FPN: 1-2                                    [1, 196, 1024]            [1, 1024, 14, 14]
│    └─Sequential: 2-4                        [1, 1024, 14, 14]         [1, 256, 56, 56]
│    │    └─ConvTranspose2d: 3-27             [1, 1024, 14, 14]         [1, 512, 28, 28]
│    │    └─Conv_LayerNorm: 3-28              [1, 512, 28, 28]          [1, 512, 28, 28]
│    │    └─GELU: 3-29                        [1, 512, 28, 28]          [1, 512, 28, 28]
│    │    └─ConvTranspose2d: 3-30             [1, 512, 28, 28]          [1, 256, 56, 56]
│    │    └─Conv2d: 3-31                      [1, 256, 56, 56]          [1, 256, 56, 56]
│    │    └─Conv2d: 3-32                      [1, 256, 56, 56]          [1, 256, 56, 56]
│    └─Sequential: 2-5                        [1, 1024, 14, 14]         [1, 512, 28, 28]
│    │    └─ConvTranspose2d: 3-33             [1, 1024, 14, 14]         [1, 512, 28, 28]
│    │    └─Conv2d: 3-34                      [1, 512, 28, 28]          [1, 512, 28, 28]
│    │    └─Conv2d: 3-35                      [1, 512, 28, 28]          [1, 512, 28, 28]
│    └─Sequential: 2-6                        [1, 1024, 14, 14]         [1, 1024, 14, 14]
│    │    └─Conv2d: 3-36                      [1, 1024, 14, 14]         [1, 1024, 14, 14]
│    │    └─Conv2d: 3-37                      [1, 1024, 14, 14]         [1, 1024, 14, 14]
===============================================================================================
Total params: 321,995,008
Trainable params: 321,591,552
Non-trainable params: 403,456
Total mult-adds (G): 5.40
===============================================================================================
Input size (MB): 0.60
Forward/backward pass size (MB): 148.68
Params size (MB): 1231.31
Estimated Total Size (MB): 1380.59
===============================================================================================


"""
