"""
Author: Alok Pandey
Git username: wildoctopus
License: MIT License
"""

import torch
import torch.nn as nn

def fine_tune_model(model, fine_tune_at):
    """
    Freezes the layers in a PyTorch model up to a specified index.

    Args:
        model (nn.Module): The PyTorch model.
        fine_tune_at (int): The index of the last layer to be frozen.

    Returns:
        nn.Module: The modified PyTorch model with frozen layers.
    """
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()

    for idx, param in enumerate(model.parameters()):
        if idx < fine_tune_at:
            param.requires_grad = False
        else:
            break
    
    return model

def num_trainable_parameters(model):
    """
    Calculates the number of trainable parameters in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        int: The number of trainable parameters.
    """
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_trainable_params
