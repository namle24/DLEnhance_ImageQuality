from basicsr.models.realesrgan_model import RealESRGANModel

class SiameseESRGANModel(RealESRGANModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.teacher = self._init_generator(opt['network_g_teacher'])
        self.student = self._init_generator(opt['network_g_student'])
        self.discriminator = self._init_discriminator(opt['network_d'])

    def forward(self, low, very_low):
        out_teacher = self.teacher(low)
        out_student = self.student(very_low)
        return out_teacher, out_student