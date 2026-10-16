import os
import warnings
import torch
from torch import nn

def pick_checkpoint(checkpoint_dir):

    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    available_checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]

    if not available_checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    if len(available_checkpoints) > 1:

        latest = max(available_checkpoints, key=lambda x: int(x.replace('.pt', '').split('_')[-1]))
        warnings.warn(f"{len(available_checkpoints)} checkpoints found in {checkpoint_dir}. Using the latest checkpoint {latest}.")
        return os.path.join(checkpoint_dir, latest)
    else:
        return os.path.join(checkpoint_dir, available_checkpoints[0])


def save_checkpoint(model: nn.Module, method: str, save_path: str):


    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)


    if method in ['best-val-loss', 'best-smooth-rank']:
        existing_checkpoints = [os.path.join(save_dir, checkpoint) for checkpoint in os.listdir(save_dir)]
        assert len(
            existing_checkpoints) <= 1, f"More than one checkpoint found in {save_dir}, even though only one is expected due to save method {method}."
        if len(existing_checkpoints) > 0:
            os.remove(existing_checkpoints[0])

    torch.save(model.state_dict(), save_path)

def load_checkpoint(model: nn.Module, load_from: str, verbose = True):


    assert os.path.exists(load_from), f'Checkpoint {load_from} not found'

    ckpt_clean = torch.load(load_from, map_location='cpu', weights_only=True)

    model.load_state_dict(ckpt_clean, strict=True)

    if verbose:
        print(f'\033[92mLoaded checkpoint from {load_from}\033[0m')

    return model

def freeze(model, layers = None):


    assert layers is None or isinstance(layers, list), f'layers must be a list or None, received {layers} instead'

    if layers is None:

        for param in model.parameters():
            param.requires_grad = False

        model.eval()

        model.frozen = 'all'

    else:

        for layer in layers:

            for name, param in model.named_parameters():
                if name.startswith(layer):
                    param.requires_grad = False

            for name, module in model.named_modules():
                if name.startswith(layer):
                    module.eval()

        model.frozen = layers

def save_model_architecture(model: nn.Module, save_path: str):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(str(model))
        f.write("\n")
        f.write(f"Total number of parameters: {sum(param.numel() for param in model.parameters())} \n")

        f.write(
            f"Total number of trainable parameters: {sum(param.numel() for param in model.parameters() if param.requires_grad)} \n")