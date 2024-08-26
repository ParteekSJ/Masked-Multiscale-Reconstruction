#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 09:52:26 2024

@author: parteeksj
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.plot_predictions import get_inverse_imagenet_transforms
from dataset.aebad_V import AeBAD_VDataset
from config.aebad_V_config import get_cfg
from models.pretrained_feat_extractor import get_pretrained_extractor, freeze_params
from models.mmr import MMR
from utils.plot_predictions import plot_predictions
from sklearn.metrics import roc_auc_score
from statistics import fmean


def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode="mul"):
    if amap_mode == "mul":
        anomaly_map = np.ones([fs_list[0].shape[0], out_size, out_size])
    else:
        anomaly_map = np.zeros([fs_list[0].shape[0], out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)  # cosine similarity alongside the dimension 1
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode="bilinear", align_corners=True)
        a_map = a_map.squeeze(1).cpu().detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == "mul":
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list


if __name__ == "__main__":
    cfg = get_cfg()
    inv_transforms = get_inverse_imagenet_transforms(cfg)

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
    freeze_params(pretrained_feat_extractor)

    model_name = "bestMMR"
    mmr_model = MMR(cfg=cfg)  # MAE + FPN
    ckpt = torch.load(
        f"/Users/parteeksj/Desktop/{model_name}.pth",
        map_location=device,
    )

    new_state_dict = {
        (
            "mask_token"
            if k == "decoder_FPN_mask_token"
            else "decoder_embed_dim" if k == "decoder_FPN_pos_embed" else k
        ): v
        for k, v in ckpt["model"].items()
    }

    mmr_model.mae.load_state_dict(new_state_dict, strict=False)
    mmr_model.fpn.load_state_dict(new_state_dict, strict=False)
    mmr_model.eval()

    # mmr_model.load_state_dict(ck)
    # mmr_model.load_state_dict(ckpt["model"])

    print("MODELS LOADED.")

    # Load test data
    dataset = AeBAD_VDataset(
        source="/Users/parteeksj/Desktop/DATASETS/AeBAD",
        classname="AeBAD_V",
        cfg=cfg,
        split="test",
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.TRAIN_SETUPS.test_batch_size,
        shuffle=False,
        # num_workers=cfg.TRAIN_SETUPS.num_workers,
    )
    print(f"LENGTH: {dataset.__len__()}")
    print(f"Testing Domain Shift Category: {cfg.DATASET.domain_shift_category}")

    # auroc_arr = []

    for idx, item in enumerate(dataloader):
        image = item["image"].to(device)
        # mask = item["mask"]

        # Ignore "good" test samples since no ground-truth masks are available.
        if item["is_anomaly"].item() == 0:
            continue

        with torch.no_grad():
            pretrained_op_dict = pretrained_feat_extractor(image)

        multi_scale_features = [pretrained_op_dict[key] for key in cfg.MODEL.return_nodes]
        reverse_features = mmr_model(image)
        multi_scale_reverse_features = [
            reverse_features[key] for key in cfg.MODEL.return_nodes
        ]

        anomaly_map, amap_list = cal_anomaly_map(
            multi_scale_features, multi_scale_reverse_features
        )

        # # Thresholding the Mask
        # mask[mask > 0.0] = 1.0

        # # Calculating the AUROC Score
        # auroc_score = roc_auc_score(mask.flatten(), anomaly_map.flatten())
        # print(f"{idx} - {auroc_score}")
        # auroc_arr.append(auroc_score)

        plot_predictions(
            cfg=cfg,
            data_dict=item,
            anom_map=anomaly_map,
            mode="1_1_OVERLAY",
            model_name=model_name,
            save_path="/Users/parteeksj/Desktop/saab",
        )

    # avg_auroc_score = fmean(auroc_arr)
    # print(f"average AUROC score: {avg_auroc_score}")


# # return anomaly_map np.array (batch_size, imagesize, imagesize)
#             anomaly_map, _ = cal_anomaly_map(multi_scale_features, multi_scale_reverse_features, image.shape[-1],
#                                              amap_mode='a')
#             for item in range(len(anomaly_map)):
#                 anomaly_map[item] = gaussian_filter(anomaly_map[item], sigma=4)


"""

odict_keys(['cls_token', 'pos_embed', 'decoder_FPN_mask_token', 'decoder_FPN_pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias', 'blocks.0.norm1.weight', 'blocks.0.norm1.bias', 'blocks.0.attn.qkv.weight', 'blocks.0.attn.qkv.bias', 'blocks.0.attn.proj.weight', 'blocks.0.attn.proj.bias', 'blocks.0.norm2.weight', 'blocks.0.norm2.bias', 'blocks.0.mlp.fc1.weight', 'blocks.0.mlp.fc1.bias', 'blocks.0.mlp.fc2.weight', 'blocks.0.mlp.fc2.bias', 'blocks.1.norm1.weight', 'blocks.1.norm1.bias', 'blocks.1.attn.qkv.weight', 'blocks.1.attn.qkv.bias', 'blocks.1.attn.proj.weight', 'blocks.1.attn.proj.bias', 'blocks.1.norm2.weight', 'blocks.1.norm2.bias', 'blocks.1.mlp.fc1.weight', 'blocks.1.mlp.fc1.bias', 'blocks.1.mlp.fc2.weight', 'blocks.1.mlp.fc2.bias', 'blocks.2.norm1.weight', 'blocks.2.norm1.bias', 'blocks.2.attn.qkv.weight', 'blocks.2.attn.qkv.bias', 'blocks.2.attn.proj.weight', 'blocks.2.attn.proj.bias', 'blocks.2.norm2.weight', 'blocks.2.norm2.bias', 'blocks.2.mlp.fc1.weight', 'blocks.2.mlp.fc1.bias', 'blocks.2.mlp.fc2.weight', 'blocks.2.mlp.fc2.bias', 'blocks.3.norm1.weight', 'blocks.3.norm1.bias', 'blocks.3.attn.qkv.weight', 'blocks.3.attn.qkv.bias', 'blocks.3.attn.proj.weight', 'blocks.3.attn.proj.bias', 'blocks.3.norm2.weight', 'blocks.3.norm2.bias', 'blocks.3.mlp.fc1.weight', 'blocks.3.mlp.fc1.bias', 'blocks.3.mlp.fc2.weight', 'blocks.3.mlp.fc2.bias', 'blocks.4.norm1.weight', 'blocks.4.norm1.bias', 'blocks.4.attn.qkv.weight', 'blocks.4.attn.qkv.bias', 'blocks.4.attn.proj.weight', 'blocks.4.attn.proj.bias', 'blocks.4.norm2.weight', 'blocks.4.norm2.bias', 'blocks.4.mlp.fc1.weight', 'blocks.4.mlp.fc1.bias', 'blocks.4.mlp.fc2.weight', 'blocks.4.mlp.fc2.bias', 'blocks.5.norm1.weight', 'blocks.5.norm1.bias', 'blocks.5.attn.qkv.weight', 'blocks.5.attn.qkv.bias', 'blocks.5.attn.proj.weight', 'blocks.5.attn.proj.bias', 'blocks.5.norm2.weight', 'blocks.5.norm2.bias', 'blocks.5.mlp.fc1.weight', 'blocks.5.mlp.fc1.bias', 'blocks.5.mlp.fc2.weight', 'blocks.5.mlp.fc2.bias', 'blocks.6.norm1.weight', 'blocks.6.norm1.bias', 'blocks.6.attn.qkv.weight', 'blocks.6.attn.qkv.bias', 'blocks.6.attn.proj.weight', 'blocks.6.attn.proj.bias', 'blocks.6.norm2.weight', 'blocks.6.norm2.bias', 'blocks.6.mlp.fc1.weight', 'blocks.6.mlp.fc1.bias', 'blocks.6.mlp.fc2.weight', 'blocks.6.mlp.fc2.bias', 'blocks.7.norm1.weight', 'blocks.7.norm1.bias', 'blocks.7.attn.qkv.weight', 'blocks.7.attn.qkv.bias', 'blocks.7.attn.proj.weight', 'blocks.7.attn.proj.bias', 'blocks.7.norm2.weight', 'blocks.7.norm2.bias', 'blocks.7.mlp.fc1.weight', 'blocks.7.mlp.fc1.bias', 'blocks.7.mlp.fc2.weight', 'blocks.7.mlp.fc2.bias', 'blocks.8.norm1.weight', 'blocks.8.norm1.bias', 'blocks.8.attn.qkv.weight', 'blocks.8.attn.qkv.bias', 'blocks.8.attn.proj.weight', 'blocks.8.attn.proj.bias', 'blocks.8.norm2.weight', 'blocks.8.norm2.bias', 'blocks.8.mlp.fc1.weight', 'blocks.8.mlp.fc1.bias', 'blocks.8.mlp.fc2.weight', 'blocks.8.mlp.fc2.bias', 'blocks.9.norm1.weight', 'blocks.9.norm1.bias', 'blocks.9.attn.qkv.weight', 'blocks.9.attn.qkv.bias', 'blocks.9.attn.proj.weight', 'blocks.9.attn.proj.bias', 'blocks.9.norm2.weight', 'blocks.9.norm2.bias', 'blocks.9.mlp.fc1.weight', 'blocks.9.mlp.fc1.bias', 'blocks.9.mlp.fc2.weight', 'blocks.9.mlp.fc2.bias', 'blocks.10.norm1.weight', 'blocks.10.norm1.bias', 'blocks.10.attn.qkv.weight', 'blocks.10.attn.qkv.bias', 'blocks.10.attn.proj.weight', 'blocks.10.attn.proj.bias', 'blocks.10.norm2.weight', 'blocks.10.norm2.bias', 'blocks.10.mlp.fc1.weight', 'blocks.10.mlp.fc1.bias', 'blocks.10.mlp.fc2.weight', 'blocks.10.mlp.fc2.bias', 'blocks.11.norm1.weight', 'blocks.11.norm1.bias', 'blocks.11.attn.qkv.weight', 'blocks.11.attn.qkv.bias', 'blocks.11.attn.proj.weight', 'blocks.11.attn.proj.bias', 'blocks.11.norm2.weight', 'blocks.11.norm2.bias', 'blocks.11.mlp.fc1.weight', 'blocks.11.mlp.fc1.bias', 'blocks.11.mlp.fc2.weight', 'blocks.11.mlp.fc2.bias', 'norm.weight', 'norm.bias', 'simfp_2.0.weight', 'simfp_2.0.bias', 'simfp_2.1.weight', 'simfp_2.1.bias', 'simfp_2.3.weight', 'simfp_2.3.bias', 'simfp_2.4.weight', 'simfp_2.4.norm.weight', 'simfp_2.4.norm.bias', 'simfp_2.5.weight', 'simfp_2.5.norm.weight', 'simfp_2.5.norm.bias', 'simfp_3.0.weight', 'simfp_3.0.bias', 'simfp_3.1.weight', 'simfp_3.1.norm.weight', 'simfp_3.1.norm.bias', 'simfp_3.2.weight', 'simfp_3.2.norm.weight', 'simfp_3.2.norm.bias', 'simfp_4.0.weight', 'simfp_4.0.norm.weight', 'simfp_4.0.norm.bias', 'simfp_4.1.weight', 'simfp_4.1.norm.weight', 'simfp_4.1.norm.bias'])

"""
