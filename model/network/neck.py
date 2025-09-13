from typing import List
import torch
import torch.nn as nn
from .modules import (
    SPPF,
    C2f_LDG,
    LDGConv,
    Conv,
    CoordinateAttention
)

class UpFlowNeck(nn.Module):
    def __init__(self, channels: List=[32, 64, 128, 256]):
        super().__init__()
                 
        self.sppf = SPPF(channels[3], channels[3])
        self.ca1 = CoordinateAttention(channels[3])
        self.ca2 = CoordinateAttention(channels[2])
        self.c2f_ldg1 = C2f_LDG(channels[2] + channels[3], channels[2], n=2)
        self.c2f_ldg2 = C2f_LDG(channels[1] + channels[2], channels[1], n=2)
        self.up = nn.Upsample(scale_factor=2)
    
    def forward(self, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor) -> torch.Tensor:
        y4 = self.sppf(x4)
    
        y3 = self.ca1(y4)
        y3 = self.up(y3)
        y3 = self.c2f_ldg1(torch.cat([y3, x3], dim=1))
        
        y2 = self.ca2(y3)
        y2 = self.up(y2)
        y2 = self.c2f_ldg2(torch.cat([y2, x2], dim=1))
        
        return y2, y3, y4
    
class DownFlowNeck(nn.Module):
    def __init__(self, channels: List=[32, 64, 128, 256]):
        super().__init__()
                 
        self.ca1 = CoordinateAttention(channels[1])
        self.ca2 = CoordinateAttention(channels[2])
        self.c2f_ldg1 = C2f_LDG(channels[1] + channels[2], channels[2], n=2)
        self.c2f_ldg2 = C2f_LDG(channels[2] + channels[3], channels[3], n=2)
        self.ldg_conv1 = LDGConv(channels[1], channels[1])
        self.ldg_conv2 = LDGConv(channels[2], channels[2])
        self.down1 = Conv(channels[1], channels[1], k=3, s=2)
        self.down2 = Conv(channels[2], channels[2], k=3, s=2)
    
    def forward(self, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor) -> torch.Tensor:

        y3 = self.down1(x2)
        y3 = self.ca1(y3)
        y3 = self.ldg_conv1(y3)
        y3 = self.c2f_ldg1(torch.cat([y3, x3], dim=1))
        
        y4 = self.down2(y3)
        y4 = self.ca2(y4)
        y4 = self.ldg_conv2(y4)
        y4 = self.c2f_ldg2(torch.cat([y4, x4], dim=1))
        
        return y3, y4
    
class Neck(nn.Module):
    def __init__(self, channels: List=[32, 64, 128, 256]):
        super().__init__()
                 
        self.upflow = UpFlowNeck(channels)
        self.downflow = DownFlowNeck(channels)
                 
    def forward(self, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor) -> torch.Tensor:
                 
        x2, x3, x4 = self.upflow(x2, x3, x4)
        x3, x4 = self.downflow(x2, x3, x4)
        return x2, x3, x4
