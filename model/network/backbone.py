from typing import List
import torch
import torch.nn as nn
from .modules import CBS, C2f

class Backbone(nn.Module):
    def __init__(
        self,
        in_channels: int=3,
        out_channels: List=[32, 64, 128, 256],
        rep: List=[1, 2, 2, 1],
        projection_layer: bool=False,
        projection_out_dim: int=768
    ):
        super().__init__()

        self.projection_layer = projection_layer

        self.stem = CBS(in_channels, out_channels[0] // 2, 3, 2)
        self.block1 = nn.Sequential(
            CBS(out_channels[0] // 2, out_channels[0], 3, 2),
            C2f(out_channels[0], out_channels[0], n=rep[0])
        )
        self.block2 = nn.Sequential(
            CBS(out_channels[0], out_channels[1], 3, 2),
            C2f(out_channels[1], out_channels[1], n=rep[1])
        )
        self.block3 = nn.Sequential(
            CBS(out_channels[1], out_channels[2], 3, 2),
            C2f(out_channels[2], out_channels[2], n=rep[2])
        )
        self.block4 = nn.Sequential(
            CBS(out_channels[2], out_channels[3], 3, 2),
            C2f(out_channels[3], out_channels[3], n=rep[3])
        )

        if self.projection_layer:
            self.projection = nn.Conv2d(
                in_channels=out_channels[-1],
                out_channels=projection_out_dim,
                kernel_size=1
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x2 = self.block2(x)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        
        if self.projection_layer:
            proj_out = self.projection(x4)
            return proj_out

        return x2, x3, x4
