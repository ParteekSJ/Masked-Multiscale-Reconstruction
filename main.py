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
from dataset.aebad_S import get_aebads
from datetime import datetime
from statistics import fmean


def freeze_params(backbone):
    for para in backbone.parameters():
        para.requires_grad = False


if __name__ == "__main__":
    cur_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    cfg = get_cfg()
    num_batches = cfg.TRAIN_SETUPS.batch_size
    # Initialize Models
    pretrained_feat_extractor = get_pretrained_extractor(return_nodes=cfg.MODEL.return_nodes)
    freeze_params(pretrained_feat_extractor)  # freezing pt model weights

    mmr_model = MMR(cfg=cfg)  # MAE + FPN

    # Load Dataset & Dataloaders
    dataset, dataloader = get_aebads(cfg)

    # Create Optimizer
    optimizer = torch.optim.AdamW(
        mmr_model.parameters(),
        lr=cfg.TRAIN_SETUPS.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=cfg.TRAIN_SETUPS.weight_decay,
    )

    best_loss = 100.0
    total_steps = int(len(dataloader.dataset) / cfg.TRAIN_SETUPS.batch_size)
    running_loss = []

    for epoch in range(10):
        pretrained_feat_extractor.eval()
        mmr_model.train()

        for idx, image in enumerate(dataloader):
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
                    f"[{idx}/{total_steps}]: {sum(running_loss[-cfg.MODEL.display_step:]) / cfg.MODEL.display_step}"
                )

        avg_epoch_loss = fmean(running_loss[-total_steps:])

        if avg_epoch_loss <= best_loss:
            # Save a checkpoint
            checkpoint = {"model": mmr_model.state_dict(), "epoch": epoch}
            torch.save(checkpoint, f"{cfg.MODEL.save_ckpt}/MMR_{cur_time}.pth")
            print(f"MODEL SAVED with LOSS: {avg_epoch_loss}.")
            best_loss = avg_epoch_loss