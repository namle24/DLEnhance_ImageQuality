import torch
import torch.nn as nn
from basicsr.models.realesrgan_model import RealESRGANModel

class SiameseESRGAN(RealESRGANModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.teacher = self._init_generator(opt['network_g_teacher'])
        self.student = self._init_generator(opt['network_g_student'])
        self.discriminator = self._init_discriminator(opt['network_d'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _init_generator(self, opt):
        from models.generators import ESRGANGenerator
        generator = ESRGANGenerator(
            in_nc=opt['num_in_ch'],
            out_nc=opt['num_out_ch'],
            nf=opt['num_feat'],
            nb=opt['num_block']
        )
        if 'pretrained_path' in opt and opt['pretrained_path']:
            generator.load_pretrained(opt['pretrained_path'])
        return generator

    def _init_discriminator(self, opt):
        from models.discriminators import ESRGANDiscriminator
        discriminator = ESRGANDiscriminator(
            in_nc=opt['num_in_ch'],
            nf=opt['num_feat']
        )
        if 'pretrained_path' in opt and opt['pretrained_path']:
            discriminator.load_pretrained(opt['pretrained_path'])
        return discriminator

    def forward(self, low, very_low):
        out_teacher = self.teacher(low)
        out_student = self.student(very_low)
        return out_teacher, out_student