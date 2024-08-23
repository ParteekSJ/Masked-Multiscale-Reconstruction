#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 12:23:40 2024

@author: parteeksj
"""
import sys

sys.path.append("../")

import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block
from utils.pos_embed import get_2d_sincos_pos_embed, get_abs_pos
from utils.random_masking import random_masking
from functools import partial
from config.default import get_cfg
import math


class MaskedAutoencoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 1024,  # encoder/decoder embedding dimension
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim

        pretrain_image_size = 224
        self.pretrain_num_patches = (pretrain_image_size // patch_size) * (
            pretrain_image_size // patch_size
        )

        ## MAE Encoder
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # Creating the CLS token. Shape: [1, 1, embed_dim]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)

        # Creating the Positional Encoding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim),
            requires_grad=False,
        )  # fixed sin-cos embedding. Shape: [1, num_patches + 1, embed_dim]

        # Creating the Encoder Blocks
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )

        self.norm = norm_layer(normalized_shape=embed_dim)

        ## MAE Decoder
        decoder_embed_dim = embed_dim  # no linear layer is used.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Creating Decoder Positional Embedding
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim),
            requires_grad=False,
        )  # fixed sin-cos embedding. Shape: [1, num_patches + 1, embed_dim]

        self.initialize_weights()  # cls token, mask token, pos embedding

    def initialize_weights(self):
        # Initializing Encoder Positional Embedding. Shape: [num_patches + 1, embed_dim]
        pos_embed = get_2d_sincos_pos_embed(
            embed_dim=self.embed_dim,
            grid_size=int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0),
        )  # [num_patches + 1, embed_dim] -> [1, num_patches + 1, embed_dim]

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0),
        )

        # Intializing patch_embed CONV layer's kernel weights
        w = self.patch_embed.proj.weight.data  # convolution layer. Shape: [1024, 1, 4, 4]
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02)
        # as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

    def forward_encoder(
        self,
        x: torch.Tensor,
        mask_ratio: float,
        ids_shuffle: torch.Tensor = None,
    ):
        # embed patches
        x = self.patch_embed(x)  # [B, 3, 224, 224] -> [B, 196, 1024]

        # add pos embed w/o cls token
        if self.patch_embed.num_patches != self.pretrain_num_patches:
            hw = (int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1])))
            x = x + get_abs_pos(self.pos_embed[:, 1:, :], hw)
        else:
            x = x + self.pos_embed[:, 1:, :]  # [B, 196, 1024] (no cls token)

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = random_masking(x, mask_ratio, ids_shuffle)
        # x.shape=[B, 49, 1024], mask.shape=[B, 196], ids_restore.shape=[B, 196]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]  # [1, 1, 1024]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)  # [B, 1, 1024]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 49 + 1, 1024] => [B, 50, 1024]

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)  # [B, 50, 1024]

        return x, mask, ids_restore

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor):
        # x.shape = [B, 50, 1024] (Embeddings of the Preserved Patches)

        # append mask tokens to sequence. Shape: [B, 147, 1024]
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )

        # ids_restore.shape[1] + 1 => TOTAL PATCHES + CLS TOKEN = 196 + 1 = 197
        # x.shape[1]: AMOUNT OF MASKED PATCHES = 50
        # ids_restore.shape[1] + 1 - x.shape[1] = 197 - 50 = 147 (NUM_MASKED_PATCHES)

        # Shape: [B, N, decoder_embed_dim]: [B, 196, 1024]
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token

        # unshuffling using `torch.gather`
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        # add pos embed
        x_ = x_ + self.decoder_pos_embed[:, 1:, :]  # [B, N, decoder_embed_dim]

        return x_, ids_restore

    def forward(self, imgs, mask_ratio=0.4):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred, ids_restore = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        return pred, ids_restore


if __name__ == "__main__":
    test_input = torch.randn(1, 3, 224, 224)
    mae = MaskedAutoencoderViT()
    cfg = get_cfg()

    with torch.no_grad():
        op = mae(test_input)

    # print(f"{op.shape=}")


"""
This gives us a good idea of what the pt model checkpoint dimensions are.
for x in ckpt['model'].keys():
    print(ckpt['model'][x].shape)


"""
