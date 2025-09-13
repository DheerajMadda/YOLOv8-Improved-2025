from typing import List, Tuple, Literal
import torch
import torch.nn as nn
from .network import Backbone, Neck, DetectionHead, SegHead

class YoloV8I(nn.Module):
    def __init__(self, config, task: Literal["det", "seg"]="det", export: bool=False):
        super().__init__()

        assert task.lower() in ("det", "seg"), '`task` must be one of ("det", "seg")'

        self.task = task.lower()
        self.export = export

        image_height: int=config.image_height
        image_width: int=config.image_width
        in_channels: int=config.in_channels
        out_channels: List=config.out_channels
        rep: List=config.rep
        stride: Tuple=config.stride
        num_classes: int=config.num_classes
        num_masks: int=config.num_masks

        feats_shape = [
            (image_height// stride[0], image_width// stride[0]),
            (image_height// stride[1], image_width// stride[1]),
            (image_height// stride[2], image_width// stride[2])
        ]
        self.backbone = Backbone(
            in_channels=in_channels,
            out_channels=out_channels,
            rep=rep
        )
        self.neck = Neck(channels=out_channels)
        self.detect_head = DetectionHead(
            feats_shape=feats_shape,
            num_classes=num_classes,
            head_channels=out_channels[1:],
            stride=stride,
            reg_max=16
        )

        if self.task == "seg":
            self.seg_head = SegHead(
                num_masks=num_masks,
                head_channels=out_channels[1:],
            )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2, x3, x4 = self.backbone(x)
        x2, x3, x4 = self.neck(x2, x3, x4)
        
        if self.training:
            x = self.detect_head([x2, x3, x4])
            if self.task == "seg":
                mc, p = self.seg_head([x2, x3, x4])
                return x, mc, p
            return x
        
        y, x = self.detect_head([x2, x3, x4])
        if self.task == "seg":
            mc, p = self.seg_head([x2, x3, x4])
            if self.export:
                return torch.cat([y, mc], dim=1), p
            return y, (x, mc, p)
        
        if self.export:
            return y
        return y, x
    