import torch.nn as nn
from archs.rrdbnet_arch import RRDBNet


class ESRGANGenerator(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, pretrained_path=None):
        super().__init__()
        self.rrdb = RRDBNet(in_nc, out_nc, nf, nb)

        if pretrained_path:
            self.load_pretrained(pretrained_path)

    def load_pretrained(self, path):
        state_dict = torch.load(path)
        if 'params_ema' in state_dict:
            self.rrdb.load_state_dict(state_dict['params_ema'])
        else:
            self.rrdb.load_state_dict(state_dict)

    def forward(self, x):
        return self.rrdb(x)