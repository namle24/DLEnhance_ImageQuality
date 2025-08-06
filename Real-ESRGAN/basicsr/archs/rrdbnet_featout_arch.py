import torch
import torch.nn as nn
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class RRDBNetFeatOut(RRDBNet):
    """RRDBNet mở rộng để xuất đặc trưng trung gian (feature maps) cho Knowledge Distillation."""

    def __init__(self, return_intermediate=False, **kwargs):
        self.return_intermediate = return_intermediate
        super().__init__(**kwargs)

    def forward(self, x):
        if self.return_intermediate:
            # Lấy feature đầu ra trước khi pixel shuffle
            fea = self.conv_first(x)
            trunk = self.rrdb_trunk(fea)
            fea = fea + self.trunk_conv(trunk)
            out = self.upconv1(fea)
            out = self.upconv2(out)
            out = self.conv_hr(self.lrelu(self.conv_last(out)))
            return out, fea  # output + intermediate features
        else:
            return super().forward(x)
