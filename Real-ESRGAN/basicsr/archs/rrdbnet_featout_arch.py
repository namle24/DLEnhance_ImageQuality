import torch
import torch.nn as nn
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class RRDBNetFeatOut(RRDBNet):
    """RRDBNet mở rộng để xuất đặc trưng trung gian (feature maps) cho Knowledge Distillation."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.return_intermediate = kwargs.get('return_intermediate', False)

    def forward(self, x):
        if self.return_intermediate:
            # Lấy feature đầu ra trước khi lên pixel
            fea = self.conv_first(x)
            trunk = self.rrdb_trunk(fea)
            fea = fea + self.trunk_conv(trunk)
            out = self.upconv1(fea)
            out = self.upconv2(out)
            out = self.conv_hr(self.lrelu(self.conv_last(out)))
            return out, fea  # trả về SR + đặc trưng trung gian
        else:
            return super().forward(x)
