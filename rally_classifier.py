# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 13:11:16 2026

@author: Wyatt Jones
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.v2 as transforms
import cv2
from PIL import Image
from sklearn.cluster import KMeans
import os
from collections import OrderedDict

# %% Build DINOv2 classifier

def build_model(num_classes=2,fine_tune=False):
    backbone_model = torch.hub.load('facebookresearch/dinov2','dinov2_vits14')
    
    model = torch.nn.Sequential(OrderedDict([
        ('backbone', backbone_model),
        ('head', torch.nn.Linear (
            in_features=384, out_features=num_classes, bias=True
        ))
    ]))
    
    if not fine_tune:
        for params in model.backbone.parameters():
            params.requires_grad = False
            
    return model

if __name__ == '__main__':
    model = build_model()
    print(model)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    
# %% Transform Data

# Training transforms
def get_train_transform(image_size):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return train_transform

# Validation transforms
def get_valid_transform(image_size):
    valid_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return valid_transform
