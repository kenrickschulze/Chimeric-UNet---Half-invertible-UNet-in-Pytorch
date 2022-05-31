""" U-Net with adaptable convolutional-/up-/down-sampling operations """

from torch import nn
import numpy as np

from .non_invertible import MaxPool2d
from .util import stacked_block
from .iunet.layers import SplitChannels, ConcatenateChannels
from itertools import zip_longest
import torch


class DownLayer(nn.Module):
    """
    Layer in the encoding path

        convolutions -> split off skipped channels -> downsample
    """

    def __init__(self, in_channels, block_op, down_op, bidx):
        """
        Build encoding/downsampling layer
        :param block_op: Callable that takes channels and returns pytorch module
                         implementing a convolutional block
        :param down_op: Callable returning downsample operation
        """
        super(DownLayer, self).__init__()

        assert in_channels % 2 == 0, "in_channels must be multiple of 2"
        self.send_all = False
        self.in_channels: int = in_channels
        self.skip_channels: int = in_channels // 2

        if down_op == MaxPool2d and bidx == 0:
            self.out_channels = in_channels // 2
        else:
            self.out_channels: int = in_channels * 2

        if down_op == MaxPool2d and bidx > 0:
            self.skip_channels: int = in_channels * 2

        else:
            self.skip_channels: int = in_channels // 2

        self.block = block_op(in_channels)
        self.split = SplitChannels(self.skip_channels)
        self.down = down_op(self.skip_channels)

    def forward(self, x):
        x = self.block(x)
        x_split, z = self.split(x)
        if not self.send_all:
            x = x_split
        x = self.down(x)
        return x, z

    def inverse(self, x, z):
        x = self.down.inverse(x)
        x = self.split.inverse(x, z)
        x = self.block.inverse(x)
        return x

    def iter_activations(self, x):
        for F in self.block:
            x = F(x)
            yield x

        x, z = self.split(x)
        x = self.down(x)
        yield x

        return x, z


class UpLayerWoConcat(nn.Module):
    def __init__(self, in_channels, block_op, up_op):
        """
        Build decoding/upsampling layer BUT without concatneating skip connection... used to exhange idwt as last step of decoder
        :param in_channels: must be multiple of 4
        :param block_op: Callable that takes channels and returns pytorch module
                         implementing a convolutional block
        :param up_op: Callable returning upsample operation
        """
        super(UpLayerWoConcat, self).__init__()

        assert in_channels % 4 == 0, "in_channels must be multiple of 4"

        self.in_channels = in_channels
        self.out_channels = in_channels // 4

        self.up = up_op(in_channels)
        self.block = block_op(self.out_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.block(x)
        return x

    def inverse2(self, x):
        print(".. called inverse before block")
        x = self.block.inverse(x)
        print(".. called inverse AFTER block")
        x = self.up.inverse(x)
        return x

    def iter_activations(self, x, z):
        x = self.up(x)
        yield x

        for F in self.block:
            x = F(x)
            yield x

        return x

    def inverse(self, x):
        for F in self.block[::-1]:
            x = F.inverse(x)
        x = self.up.inverse(x)

        return x


class UpLayer(nn.Module):
    """
    Layer in the decoding path.

        upsample -> concat skipped channels -> convolutions
    """

    def __init__(self, in_channels, block_op, up_op):
        """
        Build decoding/upsampling layer
        :param in_channels: must be multiple of 4
        :param block_op: Callable that takes channels and returns pytorch module
                         implementing a convolutional block
        :param up_op: Callable returning upsample operation
        """
        super(UpLayer, self).__init__()

        assert in_channels % 4 == 0, "in_channels must be multiple of 4"

        self.in_channels = in_channels
        self.skip_channels = in_channels // 4
        self.out_channels = in_channels // 2

        self.up = up_op(in_channels)
        self.concat = ConcatenateChannels(self.skip_channels)
        self.block = block_op(self.out_channels)

    def forward(self, x, z):
        x = self.up(x)
        x = self.concat(x, z)
        x = self.block(x)
        return x

    def inverse(self, x):
        x = self.block.inverse(x)
        x, z = self.concat.inverse(x)
        x = self.up.inverse(x)
        return x, z

    def iter_activations(self, x, z):
        x = self.up(x)
        yield x

        x = self.concat(x, z)
        for F in self.block:
            x = F(x)
            yield x

        return x

    def iter_inv_activations(self, x):
        for F in self.block[::-1]:
            x = F.inverse(x)

        x, z = self.concat.inverse(x)
        x = self.up.inverse(x)

        return x, z


class Encoder(nn.Module):
    """
    Encoder part of an U-Net.
    Produces list of Tensors in format always compatible to the Decoder
    using the same architecture.
    """

    def __init__(self, channels: int, architecture, block_op, down_op, k_size: int):
        """
        Build U-Net encoder

        :param architecture: Conv. depths of each layer, passed as list of ints
                             E.g. [2, 2, 2] will have 3 layers with 2 convolutions each.
                             Must be same as corresponding Decoder.
        :param block_op: Callable that takes channels and returns pytorch module
                         implementing a convolutional block
        :param down_op: Callable returning downsample operation
        """
        super().__init__()

        self.architecture = architecture
        self.depth = len(architecture) - 1

        # if boost is set boost channels and change channels to boosted size

        # number of channels
        self.channels = channels
        self.down_channels = [channels]

        # store out channels
        self.out_channels = []
        self.skip_channels = []

        # for colvolving out channels --> skip size == channels // 2 --> true channels* i // 2
        self.skip_layers = nn.ModuleList()

        self.layers = nn.ModuleList()
        # for all but last
        for bidx, n_blocks in enumerate(architecture[:-1]):
            if bidx > 0 and down_op == MaxPool2d:
                layer = DownLayer(
                    self.down_channels[-1],
                    block_op=stacked_block(
                        block_op, n_blocks, k_size, increase_channels=True
                    ),
                    down_op=down_op,
                    bidx=bidx,
                )

            else:
                layer = DownLayer(
                    self.down_channels[-1],
                    block_op=stacked_block(block_op, n_blocks, k_size),
                    down_op=down_op,
                    bidx=bidx,
                )

            self.down_channels.append(layer.out_channels)
            self.out_channels.append(layer.skip_channels)

            self.layers.append(layer)

        incr_ch = True if down_op == MaxPool2d else False

        self.last = stacked_block(
            block_op, architecture[-1], k_size, increase_channels=incr_ch
        )(self.down_channels[-1])

    def forward(self, x):
        codes = []

        for layer in self.layers:
            x, z = layer(x)
            codes.append(z)

        x = self.last(x)
        codes.append(x)

        return codes

    def inverse(self, codes):
        x, *codes = codes[::-1]
        x = self.last.inverse(x)

        for layer, z in zip(self.layers[::-1], codes):
            x = layer.inverse(x, z)

        return x

    def iter_fw_activations(self, x):
        Z = []
        for layer in self.layers:
            x, z = layer(x)
            Z.append(z)

        return Z

    def iter_activations(self, x):
        codes = []

        for layer in self.layers:
            x, z = yield from layer.iter_activations(x)
            codes.append(z)

        x = self.last(x)
        codes.append(x)
        yield x

        return codes


class Decoder(nn.Module):
    """
    Decoder part of an U-Net.
    Accepts codes from the Encoder and transforms back to image.
    """

    def __init__(self, channels, architecture, block_op, up_op, k_size):
        """
        Build U-Net decoder

        :param architecture: Conv. depths of each layer (in reverse), passed as list of ints
                             E.g. [2, 2, 2] will have 3 layers with 2 convolutions each.
                             Must be same for corresponding Encoder.
        :param block_op: Callable that takes channels and returns pytorch module
                         implementing a convolutional block
        :param up_op: Callable returning upsample operation
        """
        super(Decoder, self).__init__()

        self.architecture = architecture
        self.depth = len(architecture) - 1

        # reverse architecture and remove bottom layer
        self._reverse_arch = architecture[-2::-1]

        self.channels = channels
        in_channels = channels * 2**self.depth
        self._rev_in_channels = [in_channels]
        self.up_channels = [in_channels]

        self.layers = nn.ModuleList()
        for n_blocks in self._reverse_arch:
            layer = UpLayer(
                self.up_channels[-1],
                block_op=stacked_block(block_op, n_blocks, k_size),
                up_op=up_op,
            )

            self.up_channels.append(layer.out_channels)
            self._rev_in_channels.append(layer.skip_channels)
            self.layers.append(layer)

        # reverse in channels to get input channel signature
        # that matches encoder
        self.in_channels = self._rev_in_channels[::-1]

    def forward(self, codes):
        x, *codes = codes[::-1]

        for layer, z in zip(self.layers, codes):
            x = layer(x, z)

        return x

    def inverse(self, x):
        codes = []

        for layer in self.layers[::-1]:
            x, z = layer.inverse(x)
            codes.append(z)

        codes.append(x)
        return codes

    def iter_activations(self, codes):
        x, *codes = codes[::-1]

        for layer, z in zip(self.layers, codes):
            x = layer.iter_activations(x, z)

        return x

    def iter_inv_activations(self, x):
        codes = []

        for layer in self.layers[::-1]:
            x, z = layer.iter_inv_activations(x)
            codes.append(z)

        return codes


class StandardUnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_in_channels,
        num_out_channels,
        depth=1,
        k_size=3,
        zero_init=False,
        normalization="group",
        **kwargs
    ):
        super(StandardUnetBlock, self).__init__()

        conv_op = [nn.Conv1d, nn.Conv2d, nn.Conv3d][dim - 1]

        self.seq = nn.ModuleList()
        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels

        for i in range(depth):

            current_in_channels = max(num_in_channels, num_out_channels)
            current_out_channels = max(num_in_channels, num_out_channels)
            if i == 0:
                current_in_channels = num_in_channels
            if i == depth - 1:
                current_out_channels = num_out_channels

            self.seq.append(
                conv_op(
                    current_in_channels,
                    current_out_channels,
                    kernel_size=k_size,
                    padding=k_size // 2,
                    bias=True,
                )
            )

            if normalization == "instance":
                norm_op = [nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d][
                    dim - 1
                ]
                self.seq.append(norm_op(current_out_channels, affine=True))

            elif normalization == "group":
                norm_val = (
                    current_out_channels // 2 if current_out_channels % 2 == 0 else 1
                )
                self.seq.append(
                    nn.GroupNorm(max(1, (norm_val)), current_out_channels, eps=1e-3)
                )

            elif normalization == "batch":
                norm_op = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d][dim - 1]
                self.seq.append(norm_op(current_out_channels, eps=1e-3))

            else:
                print("No normalization specified.")

            self.seq.append(nn.LeakyReLU(inplace=True))

        # Initialize the block as the zero transform, such that the coupling
        # becomes the coupling becomes an identity transform (up to permutation
        # of channels)
        if zero_init:
            nn.init.zeros_(self.seq[-1].weight)
            nn.init.zeros_(self.seq[-1].bias)

        self.F = nn.Sequential(*self.seq)
        del self.seq

    def forward(self, x):
        x = self.F(x)
        return x
