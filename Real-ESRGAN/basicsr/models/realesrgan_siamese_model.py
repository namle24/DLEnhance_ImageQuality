import torch
from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.utils.registry import MODEL_REGISTRY
from realesrgan.models.realesrgan_model import RealESRGANModel
import os.path as osp
from basicsr.utils import imwrite, tensor2img
from torch.cuda.amp import autocast
from collections import OrderedDict

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
        self.current_iter = current_iter
        
        l1_gt = self.gt_usm if self.opt['l1_gt_usm'] else self.gt
        percep_gt = self.gt_usm if self.opt['percep_gt_usm'] else self.gt
        gan_gt = self.gt_usm if self.opt['gan_gt_usm'] else self.gt

        if current_iter > self.kd_warmup_iters:
            with torch.no_grad(), autocast(enabled=self.opt['train'].get('use_amp', False)):
                self.output_a, self.feat_a = self.net_g(self.lq_a, return_feats=True)

        with autocast(enabled=self.opt['train'].get('use_amp', False)):
            self.output_b, self.feat_b = self.net_g(self.lq_b, return_feats=True)

        loss_dict = OrderedDict()
        l_g_total = 0

        if self.cri_pix:
            l_g_pix = self.cri_pix(self.output_b, l1_gt)
            l_g_total += l_g_pix * self.opt['train']['pixel_opt']['loss_weight']
            loss_dict['l_g_pix'] = l_g_pix

        if self.cri_perceptual:
            l_g_percep, l_g_style = self.cri_perceptual(self.output_b, percep_gt)
            if l_g_percep is not None:
                l_g_total += l_g_percep * self.opt['train']['perceptual_opt']['perceptual_weight']
                loss_dict['l_g_percep'] = l_g_percep
            if l_g_style is not None:
                l_g_total += l_g_style * self.opt['train']['perceptual_opt']['style_weight']
                loss_dict['l_g_style'] = l_g_style

        if current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters:
            fake_g_pred = self.net_d(self.output_b)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan * self.opt['train']['gan_opt']['loss_weight']
            loss_dict['l_g_gan'] = l_g_gan

        if current_iter > self.kd_warmup_iters:
            # KD output loss
            l_kd_out = F.l1_loss(self.output_b, self.output_a.detach())
            kd_out_weight = self.opt['train'].get('lambda_kd_out', 0.05)
            l_g_total += l_kd_out * kd_out_weight
            loss_dict['l_kd_out'] = l_kd_out

            if hasattr(self, 'feat_a'):
                l_kd_feat = F.l1_loss(self.feat_b, self.feat_a.detach())
                kd_feat_weight = self.opt['train'].get('lambda_kd_feat', 0.03)
                l_g_total += l_kd_feat * kd_feat_weight
                loss_dict['l_kd_feat'] = l_kd_feat

        if current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters:
            self.optimizer_g.zero_grad()
            if self.opt['train'].get('use_amp'):
                self.scaler.scale(l_g_total).backward()
                self.scaler.step(self.optimizer_g)
                self.scaler.update()
            else:
                l_g_total.backward()
                self.optimizer_g.step()

            self._update_discriminator(gan_gt)

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def _update_discriminator(self, gan_gt):
        loss_dict = OrderedDict()
        
        self.optimizer_d.zero_grad()
        # Real
        real_d_pred = self.net_d(gan_gt)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        l_d_real.backward()
        loss_dict['l_d_real'] = l_d_real
        
        # Fake
        fake_d_pred = self.net_d(self.output_b.detach())
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        l_d_fake.backward()
        loss_dict['l_d_fake'] = l_d_fake
        
        self.optimizer_d.step()
        return loss_dict
    def validation(self, dataloader, current_iter, tb_logger, save_img=False):
        with torch.no_grad():
            return super().validation(dataloader, current_iter, tb_logger, save_img)
    
    