import torch
import torch.nn as nn
from basicsr.archs.arch_util import make_layer, pixel_unshuffle
from basicsr.archs.rrdbnet_arch import RRDBNet

class RRDBNetFeatOut(RRDBNet):
    def forward(self, x, return_feats=False):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.rrdb_trunk(fea))
        fea = fea + trunk

        if return_feats:
            feat_out = fea.clone()

        out = self.upsample(fea)
        out = self.conv_last(out)

        if return_feats:
            return out, feat_out
        return out
