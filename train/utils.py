# utils.py
"""
Utility functions for training and reproducibility.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): Seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(model, path):
    """
    Save a PyTorch model to the specified path.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        path (str): Destination file path to save the model.
    """
    torch.save(model.state_dict(), path)


def load_model(model_class, path, device):
    """
    Load a PyTorch model from file.

    Args:
        model_class (type): The model class to instantiate.
        path (str): Path to saved model file.
        device (str): Device to load the model onto.

    Returns:
        torch.nn.Module: Loaded model in eval mode.
    """
    model = model_class()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model
