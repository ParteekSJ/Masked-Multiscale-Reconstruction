#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 06:31:12 2024

# from torchvision.datasets.folder import default_loader
1, `default_loader` loads grayscale image as RGB. Hence, we do not use it.
2, Normalizing masks, i.e., mask/max(mask)??

@author: parteeksj
"""
import sys

sys.path.append("../")

import os
import torch
from fvcore.common.config import CfgNode
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


from config.default import get_cfg
from PIL import Image


"""
AeBAD_S Folder Structure

- ground_truth: Consists of directories containing the ground_truth masks.
    - ablation
        - background
        - illumination
        - same
        - view
    - breakdown
        - background
        - illumination
        - same
        - view
    - fracture
        - background
        - illumination
        - same
        - view
    - groove
        - background
        - illumination
        - same
        - view
- test: Consists of directories containing the test images.
    - ablation
        - background
        - illumination
        - same
        - view
    - breakdown
        - background
        - illumination
        - same
        - view
    - fracture
        - background
        - illumination
        - same
        - view
    - good
        - background
        - illumination
        - same
        - view
    - groove
        - background
        - illumination
        - same
        - view
- train: Consists of directories containing the training images.
    - good
        - background
        - illumination
        - same
        - view 


"""


def get_inverse_imagenet_transforms(cfg: CfgNode):
    return transforms.Compose(
        [
            transforms.Normalize(
                mean=cfg.DATASET.INV_IMAGENET_MEAN,
                std=cfg.DATASET.INV_IMAGENET_STD,
            ),
        ]
    )


class AeBAD_S_dataset(Dataset):
    def __init__(
        self,
        cfg: CfgNode,
        split: str,
        resize: int = 256,
        imagesize: int = 224,
    ):
        # Contains the train, test, and ground_truth files.
        self.parent_dir = cfg.DATASET.aebad_s_dir
        self.ds_split = split  # "train" or "test"
        self.cfg = cfg

        # Defining a list of transforms.
        self.train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    imagesize,
                    scale=(cfg.DATASET.DA_low_limit, cfg.DATASET.DA_up_limit),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=cfg.DATASET.IMAGENET_MEAN,
                    std=cfg.DATASET.IMAGENET_STD,
                ),
            ]
        )

        self.test_transforms = transforms.Compose(
            [
                transforms.Resize((resize, resize)),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=cfg.DATASET.IMAGENET_MEAN,
                    std=cfg.DATASET.IMAGENET_STD,
                ),
            ]
        )

        self.mask_transforms = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
            ]
        )

        # Initializing dictionarieis to store train, test image & corresponding masks paths.
        self.train_image_paths_dict = {}
        self.test_image_paths_dict, self.masks_path_dict = {}, {}

        if split == "train":
            # Constructing a path to the "AeBAD_S/train" folder
            train_folder_path = os.path.join(self.parent_dir, "train")

            # Constructing a path to the "AeBAD_S/train/good" folder
            good_folder_path = os.path.join(train_folder_path, "good")

            # Retrieving the folder names inside of `good_folder_path`
            train_directory_names = [
                name
                for name in os.listdir(good_folder_path)
                if os.path.isdir(os.path.join(good_folder_path, name)) and name != ".DS_Store"
            ]

            ## FOR TRAIN IMAGES
            for train_directory in train_directory_names:
                directory_path = os.path.join(good_folder_path, train_directory)

                train_imgs_path_dir = []

                # Retrieving all the imagges
                directory_path = os.path.join(good_folder_path, train_directory)
                for filename in os.listdir(directory_path):
                    if filename.endswith(".jpg") or filename.endswith(".png"):
                        image_path = os.path.join(directory_path, filename)
                        train_imgs_path_dir.append(image_path)

                self.train_image_paths_dict[train_directory] = train_imgs_path_dir

            # Flattening the dictionary.
            self.train_images_arr = [
                subdirectory_image
                for directory_name, subdirectory_images in self.train_image_paths_dict.items()
                for subdirectory_image in subdirectory_images
            ]

        else:
            # Constructing a path to the "AeBAD_S/test" folder
            test_folder_path = os.path.join(self.parent_dir, "test")

            # Constructing a path to the "AeBAD_S/ground_truth" folder
            gt_folder_path = os.path.join(self.parent_dir, "ground_truth")

            # Retrieving the folder names inside of `test_folder_path`/`gt_folder_path`
            test_folder_directory_names = [
                name
                for name in os.listdir(test_folder_path)
                if os.path.isdir(os.path.join(test_folder_path, name)) and name != ".DS_Store"
            ]

            ## FOR TEST IMAGES
            for test_directory in test_folder_directory_names:
                directory_path = os.path.join(test_folder_path, test_directory)

                subdirectory_paths = [
                    os.path.join(directory_path, subdir)
                    for subdir in os.listdir(directory_path)
                    if os.path.isdir(os.path.join(directory_path, subdir))
                    and subdir != ".DS_Store"
                ]
                subdirectory_images = {}

                for subdirectory_path in subdirectory_paths:
                    # Get subdirectory name
                    subdirectory_images[subdirectory_path.split("/")[-1]] = []
                    for filename in os.listdir(subdirectory_path):
                        if filename.endswith(".jpg") or filename.endswith(".png"):
                            image_path = os.path.join(subdirectory_path, filename)
                            subdirectory_images[subdirectory_path.split("/")[-1]].append(
                                image_path
                            )
                    self.test_image_paths_dict[test_directory] = subdirectory_images

            # Retrieving the number of 'good' images
            num_good_images = 0
            for x in self.test_image_paths_dict["good"]:
                num_good_images += len(self.test_image_paths_dict["good"][x])

            ## FOR TEST-MASKS IMAGES
            for directory_name in test_folder_directory_names:
                directory_path = os.path.join(gt_folder_path, directory_name)

                if directory_name == "good":
                    self.masks_path_dict[directory_name] = {"good": ["0"] * num_good_images}
                    continue

                subdirectory_paths = [
                    os.path.join(directory_path, subdir)
                    for subdir in os.listdir(directory_path)
                    if os.path.isdir(os.path.join(directory_path, subdir))
                    and subdir != ".DS_Store"
                ]
                subdirectory_images = {}

                for subdirectory_path in subdirectory_paths:
                    # Get subdirectory name
                    subdirectory_images[subdirectory_path.split("/")[-1]] = []
                    for filename in os.listdir(subdirectory_path):
                        if filename.endswith(".jpg") or filename.endswith(".png"):
                            image_path = os.path.join(subdirectory_path, filename)
                            subdirectory_images[subdirectory_path.split("/")[-1]].append(
                                image_path
                            )
                    self.masks_path_dict[directory_name] = subdirectory_images

            # Flattening the dictionaries.
            self.test_images_arr = [
                image_path
                for directory_name, subdirectory_images in self.test_image_paths_dict.items()
                for subdirectory_name, image_paths in subdirectory_images.items()
                for image_path in image_paths
            ]

            self.masks_images_arr = [
                image_path
                for directory_name, subdirectory_images in self.masks_path_dict.items()
                for subdirectory_name, image_paths in subdirectory_images.items()
                for image_path in image_paths
            ]

    def __len__(self):
        if self.ds_split == "train":
            # return len(self.train_images_arr)
            return 3
        else:
            # return len(self.test_images_arr)
            return 100
            # len_images = 0
            # for category in self.image_paths_dict.keys():
            #     for sub_category in self.image_paths_dict[category].keys():
            #         len_images += len(self.image_paths_dict[category][sub_category])
            # return len_images

    def __getitem__(self, index):
        if self.ds_split == "train":
            image_path = self.train_images_arr[index]
            image = Image.open(image_path)

            transformed_image = self.train_transforms(image)

            return transformed_image
        else:
            is_anom = 0  # indicating non-anomalous
            image_path = self.test_images_arr[index]
            image = Image.open(image_path)
            image = self.test_transforms(image)

            if image_path.split("/")[-3] != "good":
                mask_path = self.masks_images_arr[index]
                mask = Image.open(mask_path)
                mask = self.mask_transforms(mask)
                is_anom = 1

            else:
                mask = torch.zeros([1, *image.size()[1:]])

            return image, mask, is_anom


def get_aebads(cfg: CfgNode):
    train_dataset = AeBAD_S_dataset(cfg=cfg, split="train")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN_SETUPS.train_batch_size,
        shuffle=True,
    )

    test_dataset = AeBAD_S_dataset(cfg=cfg, split="test")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.TRAIN_SETUPS.test_batch_size,
        shuffle=True,
    )

    return train_dataloader, test_dataloader


if __name__ == "__main__":
    cfg = get_cfg()
    aebad_s_ds = AeBAD_S_dataset(cfg, "train")
    print(len(aebad_s_ds))

    inverse_transforms = transforms.Compose(
        [
            transforms.Normalize(
                mean=cfg.DATASET.INV_IMAGENET_MEAN,
                std=cfg.DATASET.INV_IMAGENET_STD,
            ),
        ]
    )

    _image, _mask = aebad_s_ds.__getitem__(0)
    _image = inverse_transforms(_image)
    print("END.")


"""

TRAIN_IMAGE_TRANSFORMS
Compose(
    RandomResizedCrop(size=(224, 224), scale=(0.2, 1.0), ratio=(0.75, 1.3333), interpolation=bicubic, antialias=True)
    RandomHorizontalFlip(p=0.5)
    ToTensor()
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
)


TEST_IMAGE_TRANSFORMS
Compose(
    Resize(size=(256, 256), interpolation=bilinear, max_size=None, antialias=True)
    CenterCrop(size=(224, 224))
    ToTensor()
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
)

MASKS_TRANSFORMS
Compose(
    Resize(size=(256, 256), interpolation=bilinear, max_size=None, antialias=True)
    CenterCrop(size=(224, 224))
    ToTensor()
)




V2 Transforms:
    
    self.train_transforms = v2.Compose(
        [
            v2.RandomResizedCrop(
                imagesize,
                # scale=(cfg.TRAIN.MMR.DA_low_limit, cfg.TRAIN.MMR.DA_up_limit),
                interpolation=InterpolationMode.BICUBIC,
            ),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=cfg.DATASET.IMAGENET_MEAN,
                std=cfg.DATASET.IMAGENET_STD,
            ),
        ]
    )

    self.test_transforms = v2.Compose(
        [
            v2.Resize((resize, resize)),
            v2.CenterCrop(imagesize),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=cfg.DATASET.IMAGENET_MEAN,
                std=cfg.DATASET.IMAGENET_STD,
            ),
        ]
    )

    self.mask_transforms = v2.Compose(
        [
            v2.Resize(resize),
            v2.CenterCrop(imagesize),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

"""
