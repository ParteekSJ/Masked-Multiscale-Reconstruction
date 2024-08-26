import sys

sys.path.append("../")

from fvcore.common.config import CfgNode
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import torch
from PIL import Image

def get_inverse_imagenet_transforms(cfg: CfgNode):
    return transforms.Compose(
        [
            transforms.Normalize(
                mean=cfg.DATASET.INV_IMAGENET_MEAN,
                std=cfg.DATASET.INV_IMAGENET_STD,
            ),
        ]
    )


def plot_predictions(
    cfg: CfgNode,
    data_dict: dict,
    anom_map: np.ndarray,
    mode: str = "1_3",
    auroc_score: float = None,
    save_path: str = "",
    model_name: str = "",
):
    
    image_path = data_dict['image_path'][0]
    image_name = data_dict['image_name'][0].split("/")[-1]
    mask = data_dict['mask']

    if mode == "1_3":
        fig, ax = plt.subplots(1, 3, figsize=(12, 12))
        
        # Set the suptitle
        fig.suptitle(f"{model_name} predictions.", fontsize=16)

        ax[0].set_title("Test Image")
        ax[0].imshow(Image.open(image_path))

        ax[1].set_title("Mask")
        ax[1].imshow(mask.squeeze(), cmap="gray")

        if auroc_score != None:
            ax[2].set_title(f"Prediction\nAUROC:{auroc_score}")
        else:
            ax[2].set_title("Prediction")
        im = ax[2].imshow(torch.from_numpy(anom_map).squeeze(), cmap="viridis")
        ax[2].axis("off")

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
        cbar.set_label("Anomaly Score", rotation=270, labelpad=15)

        plt.tight_layout()
        plt.show()
        if save_path != "":
            plt.savefig(f"{save_path}/{image_name}", bbox_inches="tight", dpi=300) 
        plt.close()

    elif mode == "2_2":
        thresholded_map = torch.from_numpy(anom_map)
        thresholded_map = (thresholded_map >= cfg.MODEL.inference_threshold).float()

        fig, ax = plt.subplots(2, 2, figsize=(12, 12))
        
        # Set the suptitle
        fig.suptitle(f"{model_name} predictions.", fontsize=16)

        # Test Image
        ax[0, 0].set_title("Test Image")
        ax[0, 0].imshow(Image.open(image_path))

        # Mask
        ax[0, 1].set_title("Ground Truth Mask")
        ax[0, 1].imshow(mask.squeeze(), cmap="gray")

        # Prediction (Anomaly Map)
        if auroc_score != None:
            ax[1, 0].set_title(f"Prediction\nAUROC:{auroc_score:.4f}")
        else:
            ax[1, 0].set_title("Prediction")
        im = ax[1, 0].imshow(anom_map.squeeze(), cmap="viridis")
        cbar = fig.colorbar(im, ax=ax[1, 0], fraction=0.046, pad=0.04)
        cbar.set_label("Anomaly Score", rotation=270, labelpad=15)

        # Thresholded Map
        ax[1, 1].set_title(f"Thresholded Map\nthreshold={cfg.MODEL.inference_threshold}")
        im = ax[1, 1].imshow(thresholded_map.squeeze(), cmap="gray")
        cbar = fig.colorbar(im, ax=ax[1, 1], fraction=0.046, pad=0.04)
        cbar.set_label("Binary Mask", rotation=270, labelpad=15)

        plt.tight_layout()

        if save_path != "":
            plt.savefig(f"{save_path}/{image_name}", bbox_inches="tight", dpi=300)

        plt.show()
        plt.close()
        
        
    elif mode == "1_1_OVERLAY":
        fig, ax = plt.subplots(1,2,figsize=(9,9))
            
        ax[0].set_title("Test Image")
        ax[0].imshow(Image.open(image_path))


        ax[1].title.set_text(f'{model_name}')
        ax[1].imshow(Image.open(image_path))
        ax[1].imshow(anom_map.squeeze(), cmap="turbo", alpha=0.65)
        plt.tight_layout()

        if save_path != "":
            plt.savefig(f"{save_path}/{image_name}", bbox_inches="tight", dpi=300)

        plt.show()
        plt.close()


if __name__ == "__main__":
    _x = torch.randn(3, 224, 224)
    _m = torch.randn(1, 224, 224)
    plot_predictions(_x, _m, _x)
