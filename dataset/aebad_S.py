import sys

sys.path.append("../")

import os
from glob import glob
from dataset.mvtec import MVTecDataset
from torch.utils.data import DataLoader
from config.default import get_cfg


class AeBAD_SDataset(MVTecDataset):
    """
    Demonstration for domain shift setups in AeBAD dataset:
    1. same is without domain shift setups.
    2. The categories include background, illumination and view.
    For more details, please read [Industrial Anomaly Detection with Domain Shift: A Real-world Dataset and Masked Multi-scale Reconstruction](https://arxiv.org/abs/2304.02216).
    """

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, self.split)
            maskpath = os.path.join(self.source, classname, "ground_truth")
            anomaly_types = [
                i for i in os.listdir(classpath) if os.path.isdir(os.path.join(classpath, i))
            ]

            imgpaths_per_class[classname] = {}
            maskpaths_per_class[classname] = {}

            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                imgpaths_per_class[classname][anomaly] = []

                if self.split == "train" and anomaly == "good":
                    sub_types = [
                        i
                        for i in os.listdir(anomaly_path)
                        if os.path.isdir(os.path.join(anomaly_path, i))
                    ]
                    for sub_good_train in sub_types:
                        imgpaths_per_class = png_load(
                            anomaly_path=anomaly_path,
                            sub_good_train=sub_good_train,
                            imgpaths_per_class=imgpaths_per_class,
                            classname=classname,
                            anomaly=anomaly,
                        )
                else:
                    # for test mode
                    imgpaths_per_class = png_load(
                        anomaly_path=anomaly_path,
                        sub_good_train=self.cfg.DATASET.domain_shift_category,
                        imgpaths_per_class=imgpaths_per_class,
                        classname=classname,
                        anomaly=anomaly,
                    )

                if self.split == "test" and anomaly != "good":
                    anomaly_mask_path = os.path.join(maskpath, anomaly)
                    temp_imgpaths_per_class = imgpaths_per_class[classname][anomaly]
                    maskpaths_per_class[classname][anomaly] = [
                        os.path.join(
                            anomaly_mask_path,
                            self.cfg.DATASET.domain_shift_category,
                            x.split("/")[-1],
                        )
                        for x in temp_imgpaths_per_class
                    ]
                else:
                    maskpaths_per_class[classname]["good"] = None

        # Unrolls the data dictionary to an easy-to-iterate list.
        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == "test" and anomaly != "good":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate


def png_load(anomaly_path, sub_good_train, imgpaths_per_class, classname, anomaly):
    specific_anomaly_path = os.path.join(anomaly_path, sub_good_train)
    anomaly_files = glob(os.path.join(specific_anomaly_path, "*.png"))
    imgpaths_per_class[classname][anomaly].extend(anomaly_files)
    return imgpaths_per_class


def get_aebadS_data(cfg):
    train_dataset = AeBAD_SDataset(
        source="/Users/parteeksj/Desktop/DATASETS/AeBAD",
        classname="AeBAD_S",
        cfg=cfg,
        split="train",
    )

    test_dataset = AeBAD_SDataset(
        source="/Users/parteeksj/Desktop/DATASETS/AeBAD",
        classname="AeBAD_S",
        cfg=cfg,
        split="test",
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.TRAIN_SETUPS.train_batch_size,
        shuffle=True,
        # num_workers=cfg.TRAIN_SETUPS.num_workers,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg.TRAIN_SETUPS.test_batch_size,
        shuffle=False,
        # num_workers=cfg.TRAIN_SETUPS.num_workers,
    )

    return train_loader, test_loader


if __name__ == "__main__":
    cfg = get_cfg()
    trl, tsl = get_aebadS_data(cfg)
    
    print('KINGS.')
