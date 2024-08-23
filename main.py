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
    device = torch.device(cfg.MODEL.device if torch.cuda.is_available() else "cpu")
    num_batches = cfg.TRAIN_SETUPS.batch_size
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
            print(f"MODEL SAVED with LOSS: {avg_epoch_loss}.")
            best_loss = avg_epoch_loss



"""
