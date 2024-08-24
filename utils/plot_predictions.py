import sys

sys.path.append("../")

from fvcore.common.config import CfgNode
from dataset.aebad_S import get_inverse_imagenet_transforms
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_predictions(
    cfg: CfgNode,
    test_image: torch.Tensor,
    mask: torch.Tensor,
    anom_map: np.ndarray,
    mode: str = "1_3",
    auroc_score: float = None,
    save_path: str = "",
    image_name: str = "",
):
    inv_transforms = get_inverse_imagenet_transforms(cfg)

    if mode == "1_3":
        fig, ax = plt.subplots(1, 3, figsize=(12, 12))

        ax[0].set_title("Test Image")
        ax[0].imshow(inv_transforms(test_image.squeeze(0)).permute(1, 2, 0))

        ax[1].set_title("Mask")
        ax[1].imshow(mask.squeeze(), cmap="gray")

        ax[2].set_title(f"Prediction\nAUROC:{auroc_score}")
        im = ax[2].imshow(torch.from_numpy(anom_map).squeeze(), cmap="viridis")
        ax[2].axis("off")

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
        cbar.set_label("Anomaly Score", rotation=270, labelpad=15)

        plt.tight_layout()
        plt.show()
        if save_path != "":
            plt.savefig(f"{save_path}/{image_name}.png", bbox_inches="tight", dpi=300)
        plt.close()

    elif mode == "2_2":
        thresholded_map = torch.from_numpy(anom_map)
        thresholded_map = (thresholded_map >= 0.056).float()

        fig, ax = plt.subplots(2, 2, figsize=(12, 12))

        # Test Image
        ax[0, 0].set_title("Test Image")
        ax[0, 0].imshow(inv_transforms(test_image.squeeze(0)).permute(1, 2, 0))

        # Mask
        ax[0, 1].set_title("Ground Truth Mask")
        ax[0, 1].imshow(mask.squeeze(), cmap="gray")

        # Prediction (Anomaly Map)
        ax[1, 0].set_title(f"Prediction\nAUROC:{auroc_score:.4f}")
        im = ax[1, 0].imshow(anom_map.squeeze(), cmap="viridis")
        cbar = fig.colorbar(im, ax=ax[1, 0], fraction=0.046, pad=0.04)
        cbar.set_label("Anomaly Score", rotation=270, labelpad=15)

        # Thresholded Map
        ax[1, 1].set_title(f"Thresholded Map\nthreshold={0.056:.4f}")
        im = ax[1, 1].imshow(thresholded_map.squeeze(), cmap="gray")
        cbar = fig.colorbar(im, ax=ax[1, 1], fraction=0.046, pad=0.04)
        cbar.set_label("Binary Mask", rotation=270, labelpad=15)

        plt.tight_layout()

        if save_path != "":
            plt.savefig(f"{save_path}/{image_name}.png", bbox_inches="tight", dpi=300)

        plt.show()
        plt.close()


if __name__ == "__main__":
    _x = torch.randn(3, 224, 224)
    _m = torch.randn(1, 224, 224)
    plot_predictions(_x, _m, _x)
