import sys

sys.path.append("../")

from fvcore.common.config import CfgNode
from dataset.aebad_S import get_inverse_imagenet_transforms
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_predictions(
    test_image: torch.Tensor,
    mask: torch.Tensor,
    anom_map: np.ndarray,
    cfg: CfgNode,
):
    inv_transforms = get_inverse_imagenet_transforms(cfg)
    fig, ax = plt.subplots(1, 3, figsize=(12, 12))

    ax[0].title.set_text("Test Image")
    ax[0].imshow(inv_transforms(test_image).permute(1, 2, 0))
    ax[0].plot()

    ax[1].title.set_text("Mask")
    ax[1].imshow(mask.squeeze())
    ax[1].plot()

    ax[2].title.set_text("Prediction")
    ax[2].imshow(torch.from_numpy(anom_map).permute(1, 2, 0))
    ax[2].plot()

    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == "__main__":
    _x = torch.randn(3, 224, 224)
    _m = torch.randn(1, 224, 224)
    plot_predictions(_x, _m, _x)
