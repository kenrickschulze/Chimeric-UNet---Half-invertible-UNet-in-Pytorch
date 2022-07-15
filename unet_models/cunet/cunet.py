"""
U-Net build from separate encoder and decoder, enabling us to
combine invertible and non-invertible layers
"""

from dataclasses import dataclass
from typing import List, Type
from torch import nn
import torch
from torch.nn.modules.linear import Identity
from .invertible import (
    InvBlock,
    InvertibleDownsampling2D,
    InvertibleUpsampling2D,
    ZeroConcat,
)
from .non_invertible import NonInvBlock, NonInvDown, NonInvUp, MaxPool2d
from .layers import UpLayerWoConcat, UpLayer, Decoder, Encoder
from .util import DWTMultiDownsample, DWTMultiUpsample, stacked_block
from functools import partial


@dataclass(frozen=True)
class UNetConfig:
    """
    Configuration for UNet, also provides typical architectures
    "chimMax" was used within the paper.

    """

    architecture: List[int]
    name: str

    encoder_block: Type
    decoder_block: Type

    down: Type
    up: Type

    expand: Type

    @classmethod
    def non_invertible(cls, architecture: List[int]):
        """Returns config for non-invertible U-Net (standard Unet)"""
        return cls(
            name="standard",
            architecture=architecture,
            encoder_block=NonInvBlock,
            decoder_block=NonInvBlock,
            down=MaxPool2d,
            up=NonInvUp,
            expand=nn.Conv2d,
        )

    @classmethod
    def invertible(cls, architecture: List[int]):
        """Returns config for invertible U-Net"""
        return cls(
            name="invertible",
            architecture=architecture,
            encoder_block=InvBlock,
            decoder_block=InvBlock,
            down=InvertibleDownsampling2D,
            up=InvertibleUpsampling2D,
            expand=ZeroConcat,
        )

    @classmethod
    def chimeric(cls, architecture: List[int]):
        """Returns config for U-Net with non-invertible encoder and invertible decoder"""
        return cls(
            name="chimeric",
            architecture=architecture,
            encoder_block=NonInvBlock,
            decoder_block=InvBlock,
            down=NonInvDown,
            up=InvertibleUpsampling2D,
            expand=nn.Conv2d,
        )

    @classmethod
    def chimMax(cls, architecture: List[int]):
        """Returns config for U-Net with non-invertible encoder and invertible decoder.
        Uses Maxpoolings instead of convolutions for downsampling within the encoder.

        """

        return cls(
            name="chimax",
            architecture=architecture,
            encoder_block=NonInvBlock,
            decoder_block=InvBlock,
            down=MaxPool2d,
            up=InvertibleUpsampling2D,
            expand=nn.Conv2d,
        )


class cUNet(nn.Module):
    """
    U-Net with adaptable architecture.
    """

    def __init__(
        self,
        model_config: UNetConfig,
        out_classes: int,
        in_channels: int,
        k_size: int = 3,
        idwt: bool = False,
        *args,
        **kwargs
    ):
        super(cUNet, self).__init__()

        self.architecture = model_config.architecture
        self.label = model_config.name

        # in_channels must match out channels for architectures with inv components
        if in_channels != out_classes:
            self.expand = model_config.expand(
                in_channels, out_classes, k_size, padding=k_size // 2
            )
        else:
            self.expand = nn.Identity()

        self.up = nn.Conv2d(
            out_classes, 4 * out_classes, k_size, padding=k_size // 2, stride=2
        )

        if idwt:
            print("...used IDWT")
            self.down = DWTMultiDownsample(reorder=False)

        else:
            print("... used invertible upsampling and additive couplings")
            assert out_classes % 2 == 0, "... only implemented for out_classes divisible by 2"
            self.down = UpLayerWoConcat(
                out_classes * 4,
                stacked_block(model_config.decoder_block, 2, k_size),
                model_config.up,
            )

        start_channels: int = 4 * out_classes

        self.encoder = Encoder(
            channels=start_channels,
            architecture=self.architecture,
            block_op=model_config.encoder_block,
            down_op=model_config.down,
            k_size=k_size
        )

        self.decoder = Decoder(
            start_channels,
            self.architecture,
            block_op=model_config.decoder_block,
            up_op=model_config.up,
            k_size=k_size,
        )

    def forward(self, x):
        x = self.expand(x)
        x = self.up(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.down(x)
        return x

    def fw_skipconnections(self, x):
        x = self.expand(x)
        x = self.up(x)
        x = self.encoder.iter_fw_activations(x)
        return x

    
    def pullback(self, x):
        if not isinstance(self.down, nn.Identity):
            x = self.down.inverse(x)
        return self.decoder.iter_inv_activations(x)
