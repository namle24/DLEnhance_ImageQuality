import torch
import torch.nn as nn
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class RRDBNetFeatOut(RRDBNet):
    """RRDBNet mở rộng để xuất đặc trưng trung gian (feature maps) cho Knowledge Distillation."""

    def __init__(self, return_intermediate=False, **kwargs):
        super().__init__(**kwargs)
        self.return_intermediate = return_intermediate  # Di chuyển sau super().__init__()

    def forward(self, x, return_feats=None):
        # Thêm tham số return_feats để tương thích với cả 2 cách gọi
        return_intermediate = return_feats if return_feats is not None else self.return_intermediate
        
        if return_intermediate:
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