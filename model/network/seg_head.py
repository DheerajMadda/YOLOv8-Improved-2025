from typing import List
import torch
import torch.nn as nn
from .modules import Conv

class Proto(nn.Module):
    """Models mask Proto module for segmentation models."""

    def __init__(self, c1: int, c_: int = 256, c2: int = 32):
        """
        Initialize the models mask Proto module with specified number of protos and masks.

        Args:
            c1 (int): Input channels.
            c_ (int): Intermediate channels.
            c2 (int): Output channels (number of protos).
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))

class SegHead(nn.Module):
    """
    Each of the 3 heads are decoupled (i.e. seperate bounding box block and classification block)
    """
    def __init__(
        self,
        num_masks: int=32,
        head_channels: List=[64, 128, 256],
    ):
        """
        Parameters:
        ----------
        num_masks: int
            The number of masks
        head_channels: tuple
            A tuple of input-channels for the 3 heads of the detecion head

        """
        
        super().__init__()

        self.num_masks = num_masks # number of masks
        self.rep = len(head_channels)

        self.proto = Proto(head_channels[0], head_channels[0], self.num_masks)  
        
        c4 = max(head_channels[0] // 4, self.num_masks)

        self.conv_blocks = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.num_masks, 1))
            for x in head_channels
        )

    def forward(self, x):
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([
            self.conv_blocks[i](x[i]).view(bs, self.num_masks, -1)
            for i in range(self.rep)
        ], dim=2)  # mask coefficients

        return mc, p