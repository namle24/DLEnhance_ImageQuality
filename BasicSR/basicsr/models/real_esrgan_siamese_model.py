import torch
import torch.nn as nn
from basicsr.models.real_esrgan_model import RealESRGANModel
from basicsr.utils.registry import MODEL_REGISTRY
import copy


@MODEL_REGISTRY.register()
class REALESRGANModelSiamese(RealESRGANModel):
    def __init__(self, opt):
        super().__init__(opt)

        # Clone generator làm 2 nhánh
        self.net_g_teacher = self.net_g  # dùng nguyên bản cho Teacher
        self.net_g_student = copy.deepcopy(self.net_g)  # nhân bản cho Student

        # Loss KD: dùng loss L1 nếu không khai báo riêng
        if 'kd_loss' in opt:
            self.cri_kd = self.build_loss(opt['kd_loss']).to(self.device)
        else:
            self.cri_kd = nn.L1Loss()

    def feed_data(self, data):
        self.lq_a = data['lq_a'].to(self.device)  # vào cho teacher
        self.lq_b = data['lq_b'].to(self.device)  # vào cho student
        self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        # Forward
        self.output_teacher = self.net_g_teacher(self.lq_a)
        self.output_student = self.net_g_student(self.lq_b)

        # Loss từng nhánh
        loss_teacher = self.cri_pix(self.output_teacher, self.gt)
        loss_student = self.cri_pix(self.output_student, self.gt)

        # Loss KD: so student và teacher
        loss_kd = self.cri_kd(self.output_student, self.output_teacher.detach())

        # Tổng loss
        loss = loss_teacher + loss_student + self.opt.get('lambda_kd', 1.0) * loss_kd

        self.optimizer_g.zero_grad()
        loss.backward()
        self.optimizer_g.step()

        # Log loss
        self.log_dict = {
            'loss_teacher': loss_teacher.item(),
            'loss_student': loss_student.item(),
            'loss_kd': loss_kd.item()
        }

    def test(self):
        self.net_g_student.eval()
        with torch.no_grad():
            self.output = self.net_g_student(self.lq_b)
        self.net_g_student.train()
