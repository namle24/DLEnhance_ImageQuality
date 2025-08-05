import torch.nn as nn
from archs.discriminator_arch import UNetDiscriminatorSN


class ESRGANDiscriminator(nn.Module):
    def __init__(self, in_nc=3, nf=64, pretrained_path=None):
        super().__init__()
        self.disc = nn.utils.spectral_norm(UNetDiscriminatorSN(in_nc, nf))

        if pretrained_path:
            self.load_pretrained(pretrained_path)

    def load_pretrained(self, path):
        state_dict = torch.load(path)
        self.disc.load_state_dict(state_dict)

    def forward(self, x):
        return self.disc(x)