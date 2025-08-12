import torch
from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.utils.registry import MODEL_REGISTRY
from realesrgan.models.realesrgan_model import RealESRGANModel
import os.path as osp
from basicsr.utils import imwrite, tensor2img
from torch.cuda.amp import autocast, GradScaler

@MODEL_REGISTRY.register()
class RealESRGANSiameseModel(RealESRGANModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.scaler = GradScaler(enabled=opt['train'].get('use_amp', False))
        
    def feed_data(self, data):
        self.lq_a = data['lq_a'].to(self.device, non_blocking=True)
        self.lq_b = data['lq_b'].to(self.device, non_blocking=True)
        self.gt = data['gt'].to(self.device, non_blocking=True)

    def optimize_parameters(self, current_iter):
        use_amp = self.opt['train'].get('use_amp', False)
        
        with autocast(enabled=use_amp):
            # Teacher forward
            with torch.no_grad():
                self.output_a, self.feat_a = self.net_g(self.lq_a, return_feats=True)

            # Student forward
            self.output_b, self.feat_b = self.net_g(self.lq_b, return_feats=True)

            # Calculate all losses
            l_pix = self.cri_pix(self.output_b, self.gt)
            l_kd_out = torch.nn.functional.l1_loss(self.output_b, self.output_a.detach())
            l_kd_feat = torch.nn.functional.l1_loss(self.feat_b, self.feat_a.detach())

            # Perceptual and style losses
            l_percep, l_style = 0, 0
            if self.cri_perceptual:
                l_percep, l_style = self.cri_perceptual(self.output_b, self.gt)

            # Total loss with weighted components
            loss = l_pix + \
                   self.opt['train']['lambda_kd_out'] * l_kd_out + \
                   self.opt['train']['lambda_kd_feat'] * l_kd_feat + \
                   l_percep + l_style

        # Optimization
        self.optimizer_g.zero_grad(set_to_none=True)
        if use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer_g)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer_g.step()

        # Logging
        self.log_dict = {
            'l_pix': l_pix.item(),
            'l_kd_out': l_kd_out.item(),
            'l_kd_feat': l_kd_feat.item(),
            'l_percep': l_percep.item() if isinstance(l_percep, torch.Tensor) else 0,
            'l_style': l_style.item() if isinstance(l_style, torch.Tensor) else 0
        }