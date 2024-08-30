import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# from dataset.aebad_V import AeBAD_VDataset
from dataset.aebad_V import AeBAD_VDataset

# from config.aebad_V_config import get_cfg
from config.aebad_V_config import get_cfg
from models.pretrained_feat_extractor import get_pretrained_extractor, freeze_params
from models.mmr import MMR
from utils.plot_predictions import plot_predictions
from utils.compute_metrics import (
    compute_pixelwise_retrieval_metrics,
    compute_imagewise_retrieval_metrics,
)
from scipy.ndimage import gaussian_filter


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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize Models
    pretrained_feat_extractor = get_pretrained_extractor(
        cfg=cfg, return_nodes=cfg.MODEL.return_nodes
    )
    pretrained_feat_extractor.to(device)
    freeze_params(pretrained_feat_extractor)

    model_name = "MMR_2024-0830_082945_LOSS"
    mmr_model = MMR(cfg=cfg)  # MAE + FPN
    ckpt = torch.load(
        f"/Users/parteeksj/Desktop/{model_name}.pth",
        map_location=device,
    )
    mmr_model.load_state_dict(ckpt["model"])

    pretrained_feat_extractor.eval()
    mmr_model.eval()

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

    label_gt_arr, label_pred_arr = [], []

    for idx, item in enumerate(dataloader):
        # Ignore "good" test samples since no ground-truth masks are available.
        if item["is_anomaly"].item() == 0:
            continue

        label_gt = item["is_anomaly"].numpy()
        label_gt_arr.append(label_gt)

        image = item["image"].to(device)

        with torch.no_grad():
            pretrained_op_dict = pretrained_feat_extractor(image)

        multi_scale_features = [pretrained_op_dict[key] for key in cfg.MODEL.return_nodes]
        reverse_features = mmr_model(image, mask_ratio=cfg.MODEL.test_mask_ratio)
        multi_scale_reverse_features = [
            reverse_features[key] for key in cfg.MODEL.return_nodes
        ]

        anomaly_map, amap_list = cal_anomaly_map(
            multi_scale_features,
            multi_scale_reverse_features,
            amap_mode="a",
        )
        anomaly_map = gaussian_filter(anomaly_map, sigma=4)
        label_pred = np.max(anomaly_map.reshape(anomaly_map.shape[0], -1), axis=1)

        label_pred_arr.append(label_pred)

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
            save_path="/Users/parteeksj/Desktop/sss",
        )

    # avg_auroc_score = fmean(auroc_arr)
    # print(f"average AUROC score: {avg_auroc_score}")

    # results = compute_imagewise_retrieval_metrics(label_pred_arr, label_gt_arr)
    # print(results)
