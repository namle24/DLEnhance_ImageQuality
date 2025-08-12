import torch
from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.utils.registry import MODEL_REGISTRY
from realesrgan.models.realesrgan_model import RealESRGANModel
import os.path as osp
from basicsr.utils import imwrite, tensor2img

@MODEL_REGISTRY.register()
class RealESRGANSiameseModel(RealESRGANModel):
    def feed_data(self, data, opt):
        self.lq_a = data['lq_a'].to(self.device)
        self.lq_b = data['lq_b'].to(self.device)
        self.gt = data['gt'].to(self.device)
        super().__init__(opt)
        self.kd_warmup_iters = opt['train'].get('kd_warmup_iters', 5000)
        self.current_iter = 0
    def optimize_parameters(self, current_iter):
        # Teacher forward
        with torch.no_grad():
            self.output_a, self.feat_a = self.net_g(self.lq_a, return_feats=True)

        # Student forward
        self.output_b, self.feat_b = self.net_g(self.lq_b, return_feats=True)

        # Pixel loss
        l_pix = self.cri_pix(self.output_b, self.gt)

        # KD output loss
        l_kd_out = torch.nn.functional.l1_loss(self.output_b, self.output_a.detach())

        # KD feature loss
        l_kd_feat = torch.nn.functional.l1_loss(self.feat_b, self.feat_a.detach())

        # Perceptual loss
        l_percep, l_style = 0, 0
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output_b, self.gt)

        # gan loss
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan
            
        # Total
        loss = l_pix + \
               self.opt['train']['lambda_kd_out'] * l_kd_out + \
               self.opt['train']['lambda_kd_feat'] * l_kd_feat + \
               l_percep + l_style

        self.optimizer_g.zero_grad()
        loss.backward()
        self.optimizer_g.step()

        self.log_dict = {
            'l_pix': l_pix.item(),
            'l_kd_out': l_kd_out.item(),
            'l_kd_feat': l_kd_feat.item(),
            'l_percep': l_percep.item() if isinstance(l_percep, torch.Tensor) else 0,
            'l_style': l_style.item() if isinstance(l_style, torch.Tensor) else 0
        }

    
    def validation(self, dataloader, current_iter, tb_logger, save_img=False):
        """Chỉ validate student network"""
        with torch.no_grad():
            return super().validation(dataloader, current_iter, tb_logger, save_img)