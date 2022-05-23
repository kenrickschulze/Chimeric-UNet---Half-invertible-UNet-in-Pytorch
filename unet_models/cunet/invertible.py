""" Invertible modules """

from torch import nn
import torch

from .non_invertible import StandardBlock
from .iunet.layers import StandardAdditiveCoupling

# import and re-export the resampling layers for clarity
# pylint: disable=unused-import
from .iunet.layers import InvertibleDownsampling2D, InvertibleUpsampling2D


class InvBlock(nn.Module):
    """
    Invertible convolution block wrapping additive coupling
    from iunets
    """

    def __init__(self, channels, k_size):
        """
        Invertible convolutional block.
        :param channels: in/out channels, must be multiple of 2

        """
        super(InvBlock, self).__init__()

        assert channels % 2 == 0, "channels must be multiple of 2"
        self.channels = channels
        self.split_pos = channels // 2
        block = StandardBlock(self.split_pos, k_size)
        self.coupling = StandardAdditiveCoupling(
            F=block, channel_split_pos=self.split_pos
        )

    def forward(self, x):
        return self.coupling(x)

    def inverse(self, x):
        return self.coupling.inverse(x)


class ZeroConcat(nn.Module):
    """
    Adds channels filled with 0 to invertibly increase dimensions
    """

    def __init__(self, in_channels, out_channels, k_size, padding):
        super(ZeroConcat, self).__init__()

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.in_channels = in_channels
        self.ch_dif = out_channels - in_channels

    def forward(self, x):
        return torch.cat((x, torch.zeros((x.shape[0], self.ch_dif, *x.shape[-2:])).to(self.device)), dim=1)

    def inverse(self, x):
        return x[:self.in_channels]
