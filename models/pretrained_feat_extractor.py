#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 10:35:21 2024

@author: parteeksj
"""
import sys

sys.path.append("../")

import torch
from torchvision.models.feature_extraction import (
    get_graph_node_names,
    create_feature_extractor,
)
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from torchinfo import summary
import matplotlib.pyplot as plt
import torch.fx as fx
from PIL import Image


def freeze_params(backbone):
    for para in backbone.parameters():
        para.requires_grad = False


def get_pretrained_extractor(return_nodes):
    resnet_50_model = resnet50(weights=ResNet50_Weights.DEFAULT)
    feat_extractor = create_feature_extractor(model=resnet_50_model, return_nodes=return_nodes)
    return feat_extractor


if __name__ == "__main__":
    image = Image.open(fp="/Users/parteeksj/Desktop/Personal/PICS/GGatk6IbEAADxXj.jpg")
    T = transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.Resize(224),
            transforms.ToTensor(),
        ]
    )
    resnet_50_model = resnet50(weights=ResNet50_Weights.DEFAULT)

    # return node names in order of execution
    # train_nodes - correspond to nodes in model.train(), and modele,vak
    train_nodes, eval_nodes = get_graph_node_names(model=resnet_50_model)
    # check if train and eval nodes are the same.
    assert [t == e for t, e in zip(train_nodes, eval_nodes)]

    return_nodes = ["layer1", "layer2", "layer3"]
    feat_extractor = create_feature_extractor(model=resnet_50_model, return_nodes=return_nodes)

    with torch.no_grad():
        out = feat_extractor(T(image).unsqueeze(0))

    print("DONE.")


"""

['x',
 'conv1',
 'bn1',
 'relu',
 'maxpool',
 'layer1.0.conv1',
 'layer1.0.bn1',
 'layer1.0.relu',
 'layer1.0.conv2',
 'layer1.0.bn2',
 'layer1.0.relu_1',
 'layer1.0.conv3',
 'layer1.0.bn3',
 'layer1.0.downsample.0',
 'layer1.0.downsample.1',
 'layer1.0.add',
 'layer1.0.relu_2',
 'layer1.1.conv1',
 'layer1.1.bn1',
 'layer1.1.relu',
 'layer1.1.conv2',
 'layer1.1.bn2',
 'layer1.1.relu_1',
 'layer1.1.conv3',
 'layer1.1.bn3',
 'layer1.1.add',
 'layer1.1.relu_2',
 'layer1.2.conv1',
 'layer1.2.bn1',
 'layer1.2.relu',
 'layer1.2.conv2',
 'layer1.2.bn2',
 'layer1.2.relu_1',
 'layer1.2.conv3',
 'layer1.2.bn3',
 'layer1.2.add',
 'layer1.2.relu_2',
 'layer2.0.conv1',
 'layer2.0.bn1',
 'layer2.0.relu',
 'layer2.0.conv2',
 'layer2.0.bn2',
 'layer2.0.relu_1',
 'layer2.0.conv3',
 'layer2.0.bn3',
 'layer2.0.downsample.0',
 'layer2.0.downsample.1',
 'layer2.0.add',
 'layer2.0.relu_2',
 'layer2.1.conv1',
 'layer2.1.bn1',
 'layer2.1.relu',
 'layer2.1.conv2',
 'layer2.1.bn2',
 'layer2.1.relu_1',
 'layer2.1.conv3',
 'layer2.1.bn3',
 'layer2.1.add',
 'layer2.1.relu_2',
 'layer2.2.conv1',
 'layer2.2.bn1',
 'layer2.2.relu',
 'layer2.2.conv2',
 'layer2.2.bn2',
 'layer2.2.relu_1',
 'layer2.2.conv3',
 'layer2.2.bn3',
 'layer2.2.add',
 'layer2.2.relu_2',
 'layer2.3.conv1',
 'layer2.3.bn1',
 'layer2.3.relu',
 'layer2.3.conv2',
 'layer2.3.bn2',
 'layer2.3.relu_1',
 'layer2.3.conv3',
 'layer2.3.bn3',
 'layer2.3.add',
 'layer2.3.relu_2',
 'layer3.0.conv1',
 'layer3.0.bn1',
 'layer3.0.relu',
 'layer3.0.conv2',
 'layer3.0.bn2',
 'layer3.0.relu_1',
 'layer3.0.conv3',
 'layer3.0.bn3',
 'layer3.0.downsample.0',
 'layer3.0.downsample.1',
 'layer3.0.add',
 'layer3.0.relu_2',
 'layer3.1.conv1',
 'layer3.1.bn1',
 'layer3.1.relu',
 'layer3.1.conv2',
 'layer3.1.bn2',
 'layer3.1.relu_1',
 'layer3.1.conv3',
 'layer3.1.bn3',
 'layer3.1.add',
 'layer3.1.relu_2',
 'layer3.2.conv1',
 'layer3.2.bn1',
 'layer3.2.relu',
 'layer3.2.conv2',
 'layer3.2.bn2',
 'layer3.2.relu_1',
 'layer3.2.conv3',
 'layer3.2.bn3',
 'layer3.2.add',
 'layer3.2.relu_2',
 'layer3.3.conv1',
 'layer3.3.bn1',
 'layer3.3.relu',
 'layer3.3.conv2',
 'layer3.3.bn2',
 'layer3.3.relu_1',
 'layer3.3.conv3',
 'layer3.3.bn3',
 'layer3.3.add',
 'layer3.3.relu_2',
 'layer3.4.conv1',
 'layer3.4.bn1',
 'layer3.4.relu',
 'layer3.4.conv2',
 'layer3.4.bn2',
 'layer3.4.relu_1',
 'layer3.4.conv3',
 'layer3.4.bn3',
 'layer3.4.add',
 'layer3.4.relu_2',
 'layer3.5.conv1',
 'layer3.5.bn1',
 'layer3.5.relu',
 'layer3.5.conv2',
 'layer3.5.bn2',
 'layer3.5.relu_1',
 'layer3.5.conv3',
 'layer3.5.bn3',
 'layer3.5.add',
 'layer3.5.relu_2',
 'layer4.0.conv1',
 'layer4.0.bn1',
 'layer4.0.relu',
 'layer4.0.conv2',
 'layer4.0.bn2',
 'layer4.0.relu_1',
 'layer4.0.conv3',
 'layer4.0.bn3',
 'layer4.0.downsample.0',
 'layer4.0.downsample.1',
 'layer4.0.add',
 'layer4.0.relu_2',
 'layer4.1.conv1',
 'layer4.1.bn1',
 'layer4.1.relu',
 'layer4.1.conv2',
 'layer4.1.bn2',
 'layer4.1.relu_1',
 'layer4.1.conv3',
 'layer4.1.bn3',
 'layer4.1.add',
 'layer4.1.relu_2',
 'layer4.2.conv1',
 'layer4.2.bn1',
 'layer4.2.relu',
 'layer4.2.conv2',
 'layer4.2.bn2',
 'layer4.2.relu_1',
 'layer4.2.conv3',
 'layer4.2.bn3',
 'layer4.2.add',
 'layer4.2.relu_2',
 'avgpool',
 'flatten',
 'fc']


"""
