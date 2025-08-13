import torch
import torch.nn.functional as F
from collections import OrderedDict
from basicsr.utils.registry import MODEL_REGISTRY
from realesrgan.models.realesrgan_model import RealESRGANModel
from torch.cuda.amp import autocast

@MODEL_REGISTRY.register()
class RealESRGANSiameseModel(RealESRGANModel):
    def __init__(self, opt):
        super().__init__(opt)
        
        self.kd_warmup_iters = opt['train'].get('kd_warmup_iters', 5000)
        
        if opt['path'].get('pretrain_network_teacher'):
            from basicsr.archs.rrdbnet_arch import RRDBNet
            self.net_g_teacher = RRDBNet(
                num_in_ch=opt['network_g'].get('num_in_ch', 3),
                num_out_ch=opt['network_g'].get('num_out_ch', 3),
                num_feat=opt['network_g'].get('num_feat', 64),
                num_block=opt['network_g'].get('num_block', 23),
                num_grow_ch=opt['network_g'].get('num_grow_ch', 32)
            )
            
            # Load weights với xử lý linh hoạt key
            load_path = opt['path']['pretrain_network_teacher']
            param_key = opt['path'].get('param_key_teacher', 'params_ema')  # Mặc định là 'params_ema'
            strict_load = opt['path'].get('strict_load_g', False)
            
            load_net = torch.load(load_path)
            if param_key is not None and param_key in load_net:
                load_net = load_net[param_key]
            self.net_g_teacher.load_state_dict(load_net, strict=strict_load)
            
            self.net_g_teacher.eval()
            for param in self.net_g_teacher.parameters():
                param.requires_grad = False

    def feed_data(self, data):
        if self.is_train:
            self.lq_a = data['lq_a'].to(self.device)
            self.lq_b = data['lq_b'].to(self.device)
            self.gt = data['gt'].to(self.device)
            
            # Apply USM sharpening to GT (critical for RealESRGAN)
            self.gt_usm = self.usm_sharpener(self.gt)
        else:
            # For validation: use standard pipeline
            super().feed_data(data)

    def optimize_parameters(self, current_iter):
        self.current_iter = current_iter
        
        loss_dict = OrderedDict()
        l_g_total = 0

        with autocast(enabled=self.opt['train'].get('use_amp', False)):
            self.output, self.feat_b = self.net_g(self.lq_b, return_feats=True)

        if current_iter > self.kd_warmup_iters and hasattr(self, 'net_g_teacher'):
            with torch.no_grad(), autocast(enabled=self.opt['train'].get('use_amp', False)):
                self.output_a, self.feat_a = self.net_g_teacher(self.lq_a, return_feats=True)

        l1_gt = self.gt_usm if self.opt['l1_gt_usm'] else self.gt
        percep_gt = self.gt_usm if self.opt['percep_gt_usm'] else self.gt
        gan_gt = self.gt_usm if self.opt['gan_gt_usm'] else self.gt


        l_g_pix = self.cri_pix(self.output, l1_gt)
        l_g_total += l_g_pix * self.opt['train']['pixel_opt']['loss_weight']
        loss_dict['l_g_pix'] = l_g_pix

        # Perceptual Loss
        if self.cri_perceptual:
            l_g_percep, l_g_style = self.cri_perceptual(self.output, percep_gt)
            l_g_total += l_g_percep * self.opt['train']['perceptual_opt']['perceptual_weight']
            loss_dict['l_g_percep'] = l_g_percep
            if l_g_style is not None:
                l_g_total += l_g_style * self.opt['train']['perceptual_opt']['style_weight']
                loss_dict['l_g_style'] = l_g_style

        # GAN Loss (critical for artifact reduction)
        if current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters:
            fake_g_pred = self.net_d(self.output)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan * self.opt['train']['gan_opt']['loss_weight']
            loss_dict['l_g_gan'] = l_g_gan

        #KD Losses (auxiliary)
        if current_iter > self.kd_warmup_iters and hasattr(self, 'output_a'):
            # Output-level KD (L1)
            l_kd_out = F.l1_loss(self.output, self.output_a.detach())
            l_g_total += l_kd_out * self.opt['train'].get('lambda_kd_out', 0.01)
            loss_dict['l_kd_out'] = l_kd_out
            
            # Feature-level KD (L1 on intermediate features)
            if hasattr(self, 'feat_a'):
                l_kd_feat = F.l1_loss(self.feat_b, self.feat_a.detach())
                l_g_total += l_kd_feat * self.opt['train'].get('lambda_kd_feat', 0.005)
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

            # Discriminator Update (must keep original logic)
            self._update_discriminator(gan_gt, loss_dict)

        # EMA update
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = loss_dict

    def _update_discriminator(self, gan_gt, loss_dict):
        """Original discriminator update logic from RealESRGAN"""
        self.optimizer_d.zero_grad()
        
        # Real
        real_d_pred = self.net_d(gan_gt)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        l_d_real.backward()
        loss_dict['l_d_real'] = l_d_real
        
        # Fake
        fake_d_pred = self.net_d(self.output.detach())
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        l_d_fake.backward()
        loss_dict['l_d_fake'] = l_d_fake
        
        self.optimizer_d.step()

    def validation(self, dataloader, current_iter, tb_logger, save_img=False):
        self.net_g.eval()
        super().validation(dataloader, current_iter, tb_logger, save_img)
        self.net_g.train()