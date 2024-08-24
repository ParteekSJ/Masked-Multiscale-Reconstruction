#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 09:29:04 2024

@author: parteeksj
"""

import torch
from config.default import get_cfg
from models.pretrained_feat_extractor import get_pretrained_extractor, freeze_params
from models.mmr import MMR
from utils.loss import each_patch_loss_function
from utils.optim_scheduler import mmr_lr_custom_scheduler
from dataset.aebad_S import get_aebads
from datetime import datetime
from statistics import fmean
import os
from inference import cal_anomaly_map
from sklearn.metrics import roc_auc_score
from utils.plot_predictions import plot_predictions
from logger.logger import setup_logging, get_logger
from utils.directory_fns import create_folders


def scratch_MAE_decoder(checkpoint):
    for key_indv in list(checkpoint["model"].keys()):
        if "decoder" in key_indv or key_indv == "mask_token":
            checkpoint["model"].pop(key_indv)
    return checkpoint


if __name__ == "__main__":
    cur_time = datetime.now().strftime("%Y-%m%d_%H%M%S")
    cfg = get_cfg()

    experiment_dir, checkpoints_dir, config_dir, log_dir = create_folders(cfg)

    setup_logging(save_dir=log_dir, log_config=cfg.TRAIN_SETUPS.logger_json_dir)
    logger = get_logger(name="train")  # log message printing

    # device = (
    #     "mps"
    #     if torch.backends.mps.is_available()
    #     else "cuda" if torch.cuda.is_available() else "cpu"
    # )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Initialize Models
    pretrained_feat_extractor = get_pretrained_extractor(return_nodes=cfg.MODEL.return_nodes)
    pretrained_feat_extractor.to(device)
    freeze_params(pretrained_feat_extractor)  # freezing pt model weights
    logger.info("Pretrained Model Loaded.")

    mmr_model = MMR(cfg=cfg)  # MAE + FPN
    # Loading MAE-VIT-b Checkpoints
    ckpt = torch.load(cfg.MODEL.mae_pt_ckpt)
    # Remove all decoder-related parameters and the mask token values from the MAE-VIT ckpt
    ckpt = scratch_MAE_decoder(ckpt)
    mmr_model.mae.load_state_dict(ckpt["model"], strict=False)  # Load encoder-only weights
    mmr_model.to(device)
    logger.info("MMR Model Loaded.")

    p = torch.load(
        f="/Users/parteeksj/Desktop/MMR_2024-08-22_19_30_07.pth", map_location="cpu"
    )
    mmr_model.load_state_dict(p["model"])

    # Load Dataset & Dataloaders
    train_loader, test_loader = get_aebads(cfg)
    logger.info(f"Total Training Samples: {len(train_loader.dataset)}")
    logger.info(f"Total Testing Samples: {len(test_loader.dataset)}")

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

    logger.info("Optimizer & Scheduler Defined.")

    best_loss = 100.0
    best_auroc = 0.0
    running_loss = []
    total_train_steps = int(len(train_loader.dataset) / cfg.TRAIN_SETUPS.train_batch_size)
    total_test_steps = int(len(test_loader.dataset) / cfg.TRAIN_SETUPS.test_batch_size)

    # PERFORM TRAINING.
    for epoch in range(cfg.TRAIN_SETUPS.epochs):
        pretrained_feat_extractor.eval()
        mmr_model.train()

        for idx, image in enumerate(train_loader):
            image = image.to(device)
            with torch.no_grad():
                pretrained_op_dict = pretrained_feat_extractor(image)

            multi_scale_features = [pretrained_op_dict[key] for key in cfg.MODEL.return_nodes]
            reverse_features = mmr_model(image)
            multi_scale_reverse_features = [
                reverse_features[key] for key in cfg.MODEL.return_nodes
            ]

            loss = each_patch_loss_function(
                multi_scale_features,
                multi_scale_reverse_features,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())

            if idx % cfg.MODEL.display_step == 0 and idx != 0:
                logger.info(
                    f"[Epoch {epoch}, Step {idx}/{total_train_steps}]: {fmean(running_loss[-cfg.MODEL.display_step:])}"
                )

        avg_epoch_loss = fmean(running_loss[-total_train_steps:])
        scheduler.step()

        if avg_epoch_loss <= best_loss:
            # Save a checkpoint
            checkpoint = {
                "model": mmr_model.state_dict(),
                "epoch": epoch,
                "loss": avg_epoch_loss,
            }
            torch.save(checkpoint, f"{checkpoints_dir}/MMR_{cur_time}_LOSS.pth")
            logger.info(f"[{epoch}] MODEL SAVED with LOSS: {avg_epoch_loss}.")
            best_loss = avg_epoch_loss

        # PERFORM VALIDATION.
        if (epoch + 1) % cfg.TRAIN_SETUPS.validation_every_epoch == 0:
            mmr_model.eval()

            auroc_arr = []  # array to hold auroc scores.

            for idx, (image, mask, is_anom) in enumerate(test_loader):
                image = image.to(device)

                # Ignore "good" test samples since no ground-truth masks are available.
                if is_anom == 0:
                    continue

                with torch.no_grad():
                    pretrained_op_dict = pretrained_feat_extractor(image)

                multi_scale_features = [
                    pretrained_op_dict[key] for key in cfg.MODEL.return_nodes
                ]
                reverse_features = mmr_model(image)
                multi_scale_reverse_features = [
                    reverse_features[key] for key in cfg.MODEL.return_nodes
                ]

                anomaly_map, amap_list = cal_anomaly_map(
                    multi_scale_features, multi_scale_reverse_features
                )
                # Thresholding the Mask
                mask[mask > 0.0] = 1.0

                # Calculating the AUROC Score
                auroc_score = roc_auc_score(mask.flatten(), anomaly_map.flatten())
                auroc_arr.append(auroc_score)

                if idx % cfg.MODEL.display_step == 0 and idx != 0:
                    logger.info(
                        f"[Epoch {epoch}, Step {idx}/{total_test_steps}], AUROC: {fmean(auroc_arr[-cfg.MODEL.display_step:])}"
                    )

                # plot_predictions(
                #     cfg=cfg,
                #     test_image=image,
                #     mask=mask,
                #     anom_map=anomaly_map,
                #     mode="2_2",
                #     auroc_score=auroc_score,
                #     save_path="./plotss",
                #     image_name=str(idx),
                # )

            # Compute mean of the AUROC scores.
            avg_auroc = fmean(auroc_arr)

            if avg_auroc > best_auroc:
                checkpoint = {
                    "model": mmr_model.state_dict(),
                    "epoch": epoch,
                    "auroc": avg_auroc,
                    "loss": avg_epoch_loss,
                }
                torch.save(checkpoint, f"{checkpoints_dir}/MMR_{cur_time}_AUROC.pth")
                logger.info(f"[{epoch}] MODEL SAVED with AUROC: {avg_auroc}.")
                best_auroc = avg_auroc
