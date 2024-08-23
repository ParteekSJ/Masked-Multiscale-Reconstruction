#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 09:29:04 2024

@author: parteeksj
"""

import torch
from config.default import get_cfg
from models.pretrained_feat_extractor import get_pretrained_extractor
from models.mmr import MMR
from utils.loss import each_patch_loss_function
from utils.optim_scheduler import mmr_lr_custom_scheduler
from dataset.aebad_S import get_aebads
from datetime import datetime
from statistics import fmean
import os


def freeze_params(backbone):
    for para in backbone.parameters():
        para.requires_grad = False


def scratch_MAE_decoder(checkpoint):
    for key_indv in list(checkpoint["model"].keys()):
        if "decoder" in key_indv or key_indv == "mask_token":
            checkpoint["model"].pop(key_indv)
    return checkpoint


if __name__ == "__main__":
    cur_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    cfg = get_cfg()
    # device = (
    #     "mps"
    #     if torch.backends.mps.is_available()
    #     else "cuda" if torch.cuda.is_available() else "cpu"
    # )
    device = "cpu"
    print(f"Using device: {device}")

    # Initialize Models
    pretrained_feat_extractor = get_pretrained_extractor(return_nodes=cfg.MODEL.return_nodes)
    pretrained_feat_extractor.to(device)
    freeze_params(pretrained_feat_extractor)  # freezing pt model weights

    mmr_model = MMR(cfg=cfg)  # MAE + FPN
    # Loading MAE-VIT-b Checkpoints
    ckpt = torch.load(cfg.MODEL.mae_pt_ckpt)
    # Remove all decoder-related parameters and the mask token values from the MAE-VIT ckpt
    ckpt = scratch_MAE_decoder(ckpt)
    mmr_model.mae.load_state_dict(ckpt["model"], strict=False)  # Load encoder weights
    mmr_model.to(device)

    # Load Dataset & Dataloaders
    dataset, dataloader = get_aebads(cfg)
    print(f"Total Samples: {len(dataset)}")

    # Check if 'checkpoints' folder is created. If not, create one.
    if not os.path.exists(cfg.MODEL.save_ckpt):
        os.makedirs(cfg.MODEL.save_ckpt)

    # Create Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        mmr_model.parameters(),
        lr=cfg.TRAIN_SETUPS.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=cfg.TRAIN_SETUPS.weight_decay,
    )
    scheduler = mmr_lr_custom_scheduler(optimizer, cfg)

    best_loss = 100.0
    total_steps = int(len(dataloader.dataset) / cfg.TRAIN_SETUPS.batch_size)
    running_loss = []

    for epoch in range(cfg.TRAIN_SETUPS.epochs):
        pretrained_feat_extractor.eval()
        mmr_model.train()

        for idx, image in enumerate(dataloader):
            image = image.to(device)
            with torch.no_grad():
                pretrained_op_dict = pretrained_feat_extractor(image)

            multi_scale_features = [pretrained_op_dict[key] for key in cfg.MODEL.return_nodes]
            reverse_features = mmr_model(image)
            multi_scale_reverse_features = [
                reverse_features[key] for key in cfg.MODEL.return_nodes
            ]

            loss = each_patch_loss_function(multi_scale_features, multi_scale_reverse_features)
            # print(f"E[{epoch}] - S[{idx}] - {loss.item()}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())

            if idx % cfg.MODEL.display_step == 0 and idx != 0:
                print(
                    f"[Epoch {epoch}, Step {idx}/{total_steps}]: {sum(running_loss[-cfg.MODEL.display_step:]) / cfg.MODEL.display_step}"
                )

        avg_epoch_loss = fmean(running_loss[-total_steps:])
        scheduler.step()

        if avg_epoch_loss <= best_loss:
            # Save a checkpoint
            checkpoint = {"model": mmr_model.state_dict(), "epoch": epoch}
            torch.save(checkpoint, f"{cfg.MODEL.save_ckpt}/MMR_{cur_time}.pth")
            print(f"[{epoch}] MODEL SAVED with LOSS: {avg_epoch_loss}.")
            best_loss = avg_epoch_loss


"""

Checkpoint Keys Dictionary

odict_keys(['cls_token', 'pos_embed', 'mask_token', 'decoder_pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias', 'blocks.0.norm1.weight', 'blocks.0.norm1.bias', 'blocks.0.attn.qkv.weight', 'blocks.0.attn.qkv.bias', 'blocks.0.attn.proj.weight', 'blocks.0.attn.proj.bias', 'blocks.0.norm2.weight', 'blocks.0.norm2.bias', 'blocks.0.mlp.fc1.weight', 'blocks.0.mlp.fc1.bias', 'blocks.0.mlp.fc2.weight', 'blocks.0.mlp.fc2.bias', 'blocks.1.norm1.weight', 'blocks.1.norm1.bias', 'blocks.1.attn.qkv.weight', 'blocks.1.attn.qkv.bias', 'blocks.1.attn.proj.weight', 'blocks.1.attn.proj.bias', 'blocks.1.norm2.weight', 'blocks.1.norm2.bias', 'blocks.1.mlp.fc1.weight', 'blocks.1.mlp.fc1.bias', 'blocks.1.mlp.fc2.weight', 'blocks.1.mlp.fc2.bias', 'blocks.2.norm1.weight', 'blocks.2.norm1.bias', 'blocks.2.attn.qkv.weight', 'blocks.2.attn.qkv.bias', 'blocks.2.attn.proj.weight', 'blocks.2.attn.proj.bias', 'blocks.2.norm2.weight', 'blocks.2.norm2.bias', 'blocks.2.mlp.fc1.weight', 'blocks.2.mlp.fc1.bias', 'blocks.2.mlp.fc2.weight', 'blocks.2.mlp.fc2.bias', 'blocks.3.norm1.weight', 'blocks.3.norm1.bias', 'blocks.3.attn.qkv.weight', 'blocks.3.attn.qkv.bias', 'blocks.3.attn.proj.weight', 'blocks.3.attn.proj.bias', 'blocks.3.norm2.weight', 'blocks.3.norm2.bias', 'blocks.3.mlp.fc1.weight', 'blocks.3.mlp.fc1.bias', 'blocks.3.mlp.fc2.weight', 'blocks.3.mlp.fc2.bias', 'blocks.4.norm1.weight', 'blocks.4.norm1.bias', 'blocks.4.attn.qkv.weight', 'blocks.4.attn.qkv.bias', 'blocks.4.attn.proj.weight', 'blocks.4.attn.proj.bias', 'blocks.4.norm2.weight', 'blocks.4.norm2.bias', 'blocks.4.mlp.fc1.weight', 'blocks.4.mlp.fc1.bias', 'blocks.4.mlp.fc2.weight', 'blocks.4.mlp.fc2.bias', 'blocks.5.norm1.weight', 'blocks.5.norm1.bias', 'blocks.5.attn.qkv.weight', 'blocks.5.attn.qkv.bias', 'blocks.5.attn.proj.weight', 'blocks.5.attn.proj.bias', 'blocks.5.norm2.weight', 'blocks.5.norm2.bias', 'blocks.5.mlp.fc1.weight', 'blocks.5.mlp.fc1.bias', 'blocks.5.mlp.fc2.weight', 'blocks.5.mlp.fc2.bias', 'blocks.6.norm1.weight', 'blocks.6.norm1.bias', 'blocks.6.attn.qkv.weight', 'blocks.6.attn.qkv.bias', 'blocks.6.attn.proj.weight', 'blocks.6.attn.proj.bias', 'blocks.6.norm2.weight', 'blocks.6.norm2.bias', 'blocks.6.mlp.fc1.weight', 'blocks.6.mlp.fc1.bias', 'blocks.6.mlp.fc2.weight', 'blocks.6.mlp.fc2.bias', 'blocks.7.norm1.weight', 'blocks.7.norm1.bias', 'blocks.7.attn.qkv.weight', 'blocks.7.attn.qkv.bias', 'blocks.7.attn.proj.weight', 'blocks.7.attn.proj.bias', 'blocks.7.norm2.weight', 'blocks.7.norm2.bias', 'blocks.7.mlp.fc1.weight', 'blocks.7.mlp.fc1.bias', 'blocks.7.mlp.fc2.weight', 'blocks.7.mlp.fc2.bias', 'blocks.8.norm1.weight', 'blocks.8.norm1.bias', 'blocks.8.attn.qkv.weight', 'blocks.8.attn.qkv.bias', 'blocks.8.attn.proj.weight', 'blocks.8.attn.proj.bias', 'blocks.8.norm2.weight', 'blocks.8.norm2.bias', 'blocks.8.mlp.fc1.weight', 'blocks.8.mlp.fc1.bias', 'blocks.8.mlp.fc2.weight', 'blocks.8.mlp.fc2.bias', 'blocks.9.norm1.weight', 'blocks.9.norm1.bias', 'blocks.9.attn.qkv.weight', 'blocks.9.attn.qkv.bias', 'blocks.9.attn.proj.weight', 'blocks.9.attn.proj.bias', 'blocks.9.norm2.weight', 'blocks.9.norm2.bias', 'blocks.9.mlp.fc1.weight', 'blocks.9.mlp.fc1.bias', 'blocks.9.mlp.fc2.weight', 'blocks.9.mlp.fc2.bias', 'blocks.10.norm1.weight', 'blocks.10.norm1.bias', 'blocks.10.attn.qkv.weight', 'blocks.10.attn.qkv.bias', 'blocks.10.attn.proj.weight', 'blocks.10.attn.proj.bias', 'blocks.10.norm2.weight', 'blocks.10.norm2.bias', 'blocks.10.mlp.fc1.weight', 'blocks.10.mlp.fc1.bias', 'blocks.10.mlp.fc2.weight', 'blocks.10.mlp.fc2.bias', 'blocks.11.norm1.weight', 'blocks.11.norm1.bias', 'blocks.11.attn.qkv.weight', 'blocks.11.attn.qkv.bias', 'blocks.11.attn.proj.weight', 'blocks.11.attn.proj.bias', 'blocks.11.norm2.weight', 'blocks.11.norm2.bias', 'blocks.11.mlp.fc1.weight', 'blocks.11.mlp.fc1.bias', 'blocks.11.mlp.fc2.weight', 'blocks.11.mlp.fc2.bias', 'norm.weight', 'norm.bias', 'decoder_embed.weight', 'decoder_embed.bias', 'decoder_blocks.0.norm1.weight', 'decoder_blocks.0.norm1.bias', 'decoder_blocks.0.attn.qkv.weight', 'decoder_blocks.0.attn.qkv.bias', 'decoder_blocks.0.attn.proj.weight', 'decoder_blocks.0.attn.proj.bias', 'decoder_blocks.0.norm2.weight', 'decoder_blocks.0.norm2.bias', 'decoder_blocks.0.mlp.fc1.weight', 'decoder_blocks.0.mlp.fc1.bias', 'decoder_blocks.0.mlp.fc2.weight', 'decoder_blocks.0.mlp.fc2.bias', 'decoder_blocks.1.norm1.weight', 'decoder_blocks.1.norm1.bias', 'decoder_blocks.1.attn.qkv.weight', 'decoder_blocks.1.attn.qkv.bias', 'decoder_blocks.1.attn.proj.weight', 'decoder_blocks.1.attn.proj.bias', 'decoder_blocks.1.norm2.weight', 'decoder_blocks.1.norm2.bias', 'decoder_blocks.1.mlp.fc1.weight', 'decoder_blocks.1.mlp.fc1.bias', 'decoder_blocks.1.mlp.fc2.weight', 'decoder_blocks.1.mlp.fc2.bias', 'decoder_blocks.2.norm1.weight', 'decoder_blocks.2.norm1.bias', 'decoder_blocks.2.attn.qkv.weight', 'decoder_blocks.2.attn.qkv.bias', 'decoder_blocks.2.attn.proj.weight', 'decoder_blocks.2.attn.proj.bias', 'decoder_blocks.2.norm2.weight', 'decoder_blocks.2.norm2.bias', 'decoder_blocks.2.mlp.fc1.weight', 'decoder_blocks.2.mlp.fc1.bias', 'decoder_blocks.2.mlp.fc2.weight', 'decoder_blocks.2.mlp.fc2.bias', 'decoder_blocks.3.norm1.weight', 'decoder_blocks.3.norm1.bias', 'decoder_blocks.3.attn.qkv.weight', 'decoder_blocks.3.attn.qkv.bias', 'decoder_blocks.3.attn.proj.weight', 'decoder_blocks.3.attn.proj.bias', 'decoder_blocks.3.norm2.weight', 'decoder_blocks.3.norm2.bias', 'decoder_blocks.3.mlp.fc1.weight', 'decoder_blocks.3.mlp.fc1.bias', 'decoder_blocks.3.mlp.fc2.weight', 'decoder_blocks.3.mlp.fc2.bias', 'decoder_blocks.4.norm1.weight', 'decoder_blocks.4.norm1.bias', 'decoder_blocks.4.attn.qkv.weight', 'decoder_blocks.4.attn.qkv.bias', 'decoder_blocks.4.attn.proj.weight', 'decoder_blocks.4.attn.proj.bias', 'decoder_blocks.4.norm2.weight', 'decoder_blocks.4.norm2.bias', 'decoder_blocks.4.mlp.fc1.weight', 'decoder_blocks.4.mlp.fc1.bias', 'decoder_blocks.4.mlp.fc2.weight', 'decoder_blocks.4.mlp.fc2.bias', 'decoder_blocks.5.norm1.weight', 'decoder_blocks.5.norm1.bias', 'decoder_blocks.5.attn.qkv.weight', 'decoder_blocks.5.attn.qkv.bias', 'decoder_blocks.5.attn.proj.weight', 'decoder_blocks.5.attn.proj.bias', 'decoder_blocks.5.norm2.weight', 'decoder_blocks.5.norm2.bias', 'decoder_blocks.5.mlp.fc1.weight', 'decoder_blocks.5.mlp.fc1.bias', 'decoder_blocks.5.mlp.fc2.weight', 'decoder_blocks.5.mlp.fc2.bias', 'decoder_blocks.6.norm1.weight', 'decoder_blocks.6.norm1.bias', 'decoder_blocks.6.attn.qkv.weight', 'decoder_blocks.6.attn.qkv.bias', 'decoder_blocks.6.attn.proj.weight', 'decoder_blocks.6.attn.proj.bias', 'decoder_blocks.6.norm2.weight', 'decoder_blocks.6.norm2.bias', 'decoder_blocks.6.mlp.fc1.weight', 'decoder_blocks.6.mlp.fc1.bias', 'decoder_blocks.6.mlp.fc2.weight', 'decoder_blocks.6.mlp.fc2.bias', 'decoder_blocks.7.norm1.weight', 'decoder_blocks.7.norm1.bias', 'decoder_blocks.7.attn.qkv.weight', 'decoder_blocks.7.attn.qkv.bias', 'decoder_blocks.7.attn.proj.weight', 'decoder_blocks.7.attn.proj.bias', 'decoder_blocks.7.norm2.weight', 'decoder_blocks.7.norm2.bias', 'decoder_blocks.7.mlp.fc1.weight', 'decoder_blocks.7.mlp.fc1.bias', 'decoder_blocks.7.mlp.fc2.weight', 'decoder_blocks.7.mlp.fc2.bias', 'decoder_norm.weight', 'decoder_norm.bias', 'decoder_pred.weight', 'decoder_pred.bias'])

"""
