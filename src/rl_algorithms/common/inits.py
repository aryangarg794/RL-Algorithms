import torch
import torch.nn as nn
import numpy as np


@torch.no_grad()
def kaiming_init(m):
    if hasattr(m, "weight"):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if hasattr(m, "bias"):
            nn.init.constant_(m.bias, 0.0)


@torch.no_grad()
def orthogonal_init(layer, std=np.sqrt(2), bias_const=0.0):
    if hasattr(layer, "weight"):
        nn.init.orthogonal_(layer.weight, std)
        if hasattr(layer, "bias") and layer.bias is not None:
            nn.init.constant_(layer.bias, bias_const)

@torch.no_grad()
def init_trpo(model):
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)

    nn.init.orthogonal_(model.ffn[-1].weight, gain=0.01)
    nn.init.constant_(model.ffn[-1].bias, 0.0)


