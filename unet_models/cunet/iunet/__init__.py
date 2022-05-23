import torch
from torch import nn

from .networks import iUNet

# from .layers import InvertibleDownsampling2D, InvertibleUpsampling2D


class RemoveChannelDim(nn.Module):
    def forward(self, x):
        return x[:, 0, ...]


def make_iunet(
    depth,
    train=True,
    lr=1e-3,
    weight_decay=2e-4,
    disable_custom_gradient=True,
    device="cpu",
):
    """
    Helper function to easily build an
    invertible iUNet + BCELoss + ADAM
    """
    iunet = iUNet(
        in_channels=4,
        dim=2,
        architecture=[2] * depth,
        disable_custom_gradient=disable_custom_gradient,
    )

    # blowup_layer = InvertibleDownsampling2D(1)
    # collapse_layer = InvertibleUpsampling2D(1)

    blowup_layer = nn.Conv2d(1, 4, 3)
    collapse_layer = nn.Conv2d(4, 1, 3)

    iunet_full = nn.Sequential(
        blowup_layer, iunet, collapse_layer, RemoveChannelDim()
    ).to(device)

    if not train:
        return iunet_full

    criterion = nn.BCEWithLogitsLoss(reduction="sum").to(device)
    optim = torch.optim.Adam(iunet_full.parameters(), lr=lr, weight_decay=weight_decay)
    return iunet_full, criterion, optim
