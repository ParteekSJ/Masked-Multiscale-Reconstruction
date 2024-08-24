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

from config.default import get_cfg
from dataset.aebad_S import AeBAD_S_dataset, get_inverse_imagenet_transforms
from models.pretrained_feat_extractor import get_pretrained_extractor, freeze_params
from models.mmr import MMR
from utils.plot_predictions import plot_predictions


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

    mmr_model = MMR(cfg=cfg)  # MAE + FPN
    ckpt = torch.load(
        "/Users/parteeksj/Desktop/MMR_2024-08-22_19_30_07.pth", map_location=device
    )
    # mmr_model.load_state_dict(ck)
    mmr_model.load_state_dict(ckpt["model"])

    print("MODELS LOADED.")

    # Load test data
    dataset = AeBAD_S_dataset(cfg, "test")
    dataloader = DataLoader(
        dataset,
        batch_size=1,
    )

    print("TEST DATA LOADED.")

    for idx, (image, mask) in enumerate(dataloader):
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

        plot_predictions(image.squeeze(0), mask, anomaly_map, cfg, mode="2_2")


# # return anomaly_map np.array (batch_size, imagesize, imagesize)
#             anomaly_map, _ = cal_anomaly_map(multi_scale_features, multi_scale_reverse_features, image.shape[-1],
#                                              amap_mode='a')
#             for item in range(len(anomaly_map)):
#                 anomaly_map[item] = gaussian_filter(anomaly_map[item], sigma=4)
