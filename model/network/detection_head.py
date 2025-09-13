from typing import List, Tuple
import torch
import torch.nn as nn
from .modules import Conv, DFL

class DetectionHead(nn.Module):
    """
    Each of the 3 heads are decoupled (i.e. seperate bounding box block and classification block)
    """
    def __init__(
        self,
        feats_shape: List[Tuple],
        num_classes: int=80,
        head_channels: List=[64, 128, 256],
        stride: Tuple=(8, 16, 32),
        reg_max: int=16
    ):
        """
        Parameters:
        ----------
        
        num_classes: int
            The number of classes
        head_channels: tuple
            A tuple of input-channels for the 3 heads of the detecion head
        stride: tuple
            A tuple of strides at different scales
        reg_max: int
            Anchor scale factor
        """
        
        super().__init__()

        self.num_classes = num_classes  # number of classes
        self.reg_max = reg_max # DFL channels (head_channels[0] // 16 to scale: 4)
        
        self.num_heads = len(head_channels)  # num of detection heads
        self.no = self.num_classes + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.tensor(stride)  # strides will be computed during build using this (only once)
        
        c2 = max((16, head_channels[0] // 4, self.reg_max * 4))
        c3 = max(head_channels[0], self.num_classes)

        self.bbox_layers = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) 
            for x in head_channels
        )
        
        self.class_layers = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.num_classes, 1)) 
            for x in head_channels
        ) 
        
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        
        # shape: anchors = (2, Sum of h*w for all feats_shapes); strides = (1, Sum of h*w for all feats_shapes)
        anchors, strides = self.make_anchors(feats_shape, self.stride, 0.5)
        self.register_buffer("anchors", anchors)
        self.register_buffer("strides", strides)
    
    def make_anchors(self, feats_shape, strides, grid_cell_offset=0.5):
        """Generate anchors from features."""
        
        anchor_points, stride_tensor = [], []
        for i, stride in enumerate(strides):
            h, w = feats_shape[i]
            sx = torch.arange(end=w) + grid_cell_offset  # shift x
            sy = torch.arange(end=h) + grid_cell_offset  # shift y
            # sy, sx = torch.meshgrid(sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sy, sx)
            sy, sx = torch.meshgrid(sy, sx, indexing='ij')
            anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride))
        return torch.cat(anchor_points).transpose(0, 1), torch.cat(stride_tensor).transpose(0, 1)

    def dist2bbox(self, distance, anchor_points, xywh=True, dim=-1):
        """Transform distance(ltrb) to box(xywh or xyxy)."""
        lt, rb = distance.chunk(2, dim)
        x1y1 = anchor_points - lt
        x2y2 = anchor_points + rb
        if xywh:
            c_xy = (x1y1 + x2y2) / 2
            wh = x2y2 - x1y1
            return torch.cat((c_xy, wh), dim)  # xywh bbox
        return torch.cat((x1y1, x2y2), dim)  # xyxy bbox

    def forward(self, x):

        for i in range(self.num_heads):
            box_out = self.bbox_layers[i](x[i])  # box_out_channels = 4 * self.reg_max = 16
            cls_out = self.class_layers[i](x[i]) # cls_out_channels = no of classes 
            x[i] = torch.cat((box_out, cls_out), 1) # N_CHANNELS = self.no = box_out_channels + cls_out_channels
        
        if self.training:
            # For input image height = 192, width = 640
            # x[0] shape = (batch_size, self.no, 24, 80)
            # x[1] shape = (batch_size, self.no, 12, 40)
            # x[2] shape = (batch_size, self.no, 6, 20)
            return x
        
        # N_OUT = (6 * 20) + (12 * 40) + (24 * 80) = 2520
        x_cat = torch.cat([xi.view(x[0].shape[0], self.no, -1) for xi in x], 2) # shape: (batch_size, self.no, N_OUT)

        box, cls = x_cat.split((self.reg_max * 4, self.num_classes), 1) 
        # shape: box = (batch_size, 4 * self.reg_max, N_OUT); cls = (batch_size, no. of classes, N_OUT)
        
        dbox = self.dist2bbox(
            self.dfl(box),
            self.anchors.unsqueeze(0),
            xywh=True,
            dim=1
        ) * self.strides
        
        # shape: dbox = (batch_size, 4, N_OUT)

        y = torch.cat((dbox, cls.sigmoid()), 1)  # shape: (batch_size, 4 + no. of classes, N_OUT). e.g (2, 11, 2520)

        return y, x
