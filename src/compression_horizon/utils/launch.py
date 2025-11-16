import torch
from transformers import set_seed


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set the tensors to be allocated on a specified device
    torch.set_default_device(device)
    return device


def set_launch_seed(seed):
    if seed is not None:
        set_seed(int(seed))


def freeze_model_parameters(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
