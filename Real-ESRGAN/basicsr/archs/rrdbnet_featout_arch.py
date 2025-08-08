import torch
import torch.nn as nn
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.registry import ARCH_REGISTRY
from torch.nn import functional as F
from basicsr.archs.arch_util import pixel_unshuffle

@ARCH_REGISTRY.register()
class RRDBNetFeatOut(RRDBNet):
    """RRDBNet mở rộng để xuất đặc trưng trung gian (feature maps) cho Knowledge Distillation."""

    def __init__(self, return_intermediate=False, **kwargs):
        super().__init__(**kwargs)
        self.return_intermediate = return_intermediate

    def forward(self, x, return_feats=None):
        # Hỗ trợ cả 2 cách gọi tham số
        return_intermediate = return_feats if return_feats is not None else self.return_intermediate
        
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat  # Đây là intermediate feature chúng ta muốn lấy
        
        if return_intermediate:
            # Lưu intermediate feature trước khi upscale
            intermediate_feat = feat
            
            # Tiếp tục quá trình upscale
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
            out = self.conv_last(self.lrelu(self.conv_hr(feat)))
            
            return out, intermediate_feat
        else:
            # Chỉ return output cuối cùng
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
            out = self.conv_last(self.lrelu(self.conv_hr(feat)))
            return out