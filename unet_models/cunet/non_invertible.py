""" Non-invertible modules """

import torch
from torch import nn


class StandardBlock(nn.Module):
    """
    Standard Conv2d -> LeakyReLU -> BatchNorm (aka GroupNorm) block
    Used in both invertible and non-invertible layers.
    """

    def __init__(self, channels, k_size, out_channels=None):
        super(StandardBlock, self).__init__()

        self.channels = channels
        if out_channels is not None:
            self.out_channels = out_channels
        else:
            self.out_channels = channels

        self.conv = nn.Conv2d(
            channels, self.out_channels, k_size, padding=int(k_size // 2), bias=False
        )
        self.relu = nn.LeakyReLU(inplace=True)
        norm_val = self.out_channels // 2 if self.out_channels % 2 == 0 else 1
        self.norm = nn.GroupNorm(max(1, (norm_val)), self.out_channels, eps=1e-3)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.norm(x)
        return x


class NonInvBlock(nn.Module):
    """
    Non-invertible convolution block, mimicking the invertible additive coupling layer
    from iunets. The input is split in two, but - in contrast to the invertible layer -
    both parts are passed through the convolution.
    The advantage of that is that the parameters and outputs have exactly the same shape,
    making this a drop-in replacement for the coupling layer.
    """

    def __init__(self, channels, k_size, out_channels=None):
        """
        Non-invertible convolutional block.
        :param channels: in/out channels, must be multiple of 2
        """
        super(NonInvBlock, self).__init__()

        assert channels % 2 == 0, "channels must be multiple of 2"
        self.channels = channels

        self.F2 = StandardBlock(channels, k_size, out_channels)

    def forward(self, x):
        return self.F2(x)


class NonInvDown(nn.Conv2d):
    """Non-invertible, learnable downsampling, implemented as convolution"""

    def __init__(self, channels):
        super(NonInvDown, self).__init__(
            channels, channels * 4, kernel_size=2, stride=2, bias=False
        )


class NonInvUp(nn.ConvTranspose2d):
    """Non-invertible, learnable upsampling, implemented as transposed convolution"""

    def __init__(self, channels):
        super(NonInvUp, self).__init__(
            channels, channels // 4, kernel_size=2, stride=2, bias=False
        )


class MaxPool2d(nn.MaxPool2d):
    def __init__(self, channels=None):
        super(MaxPool2d, self).__init__(kernel_size=2, stride=2)
