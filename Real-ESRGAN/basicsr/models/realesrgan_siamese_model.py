import torch
import torch.nn as nn
import os
from collections import OrderedDict
from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.utils.registry import MODEL_REGISTRY
from realesrgan.models.realesrgan_model import RealESRGANModel
import os.path as osp
from basicsr.utils import imwrite, tensor2img

@MODEL_REGISTRY.register()
class RealESRGANSiameseModel(RealESRGANModel):
    """Improved Siamese Real-ESRGAN with progressive training and better loss design."""
    
    def __init__(self, opt):
        super(RealESRGANSiameseModel, self).__init__(opt)
        self.current_iter = 0
        self.progressive_kd = opt['train'].get('progressive_kd', True)
        self.warmup_kd_iters = opt['train'].get('warmup_kd_iters', 50000)
        
        # Adaptive weights for different loss components
        self.lambda_kd_out_base = opt['train'].get('lambda_kd_out', 0.1)
        self.lambda_kd_feat_base = opt['train'].get('lambda_kd_feat', 0.1)
        self.lambda_consistency = opt['train'].get('lambda_consistency', 0.05)
        
        # Feature matching loss
        self.cri_feat_match = nn.L1Loss() if opt['train'].get('use_feat_match', True) else None
        
    def feed_data(self, data):
        """Enhanced data feeding with better error handling."""
        try:
            # For training with siamese data
            if 'lq_a' in data and 'lq_b' in data:
                self.lq_a = data['lq_a'].to(self.device)
                self.lq_b = data['lq_b'].to(self.device) 
            # For validation with standard paired data
            elif 'lq' in data:
                self.lq_b = data['lq'].to(self.device)
                # Use same input for both branches during validation
                self.lq_a = data['lq'].to(self.device)
                
            self.gt = data['gt'].to(self.device) if 'gt' in data else None
            
            # Apply USM sharpening for better training
            if self.gt is not None and hasattr(self, 'usm_sharpener'):
                self.gt_usm = self.usm_sharpener(self.gt)
            elif self.gt is not None:
                self.gt_usm = self.gt
                
        except Exception as e:
            print(f"Error in feed_data: {e}")
            raise

    def get_adaptive_weights(self, current_iter):
        """Progressive weight scheduling for knowledge distillation."""
        if not self.progressive_kd:
            return self.lambda_kd_out_base, self.lambda_kd_feat_base
        
        # Progressive increase of KD weights
        progress = min(1.0, current_iter / self.warmup_kd_iters)
        kd_out_weight = self.lambda_kd_out_base * progress
        kd_feat_weight = self.lambda_kd_feat_base * progress
        
        return kd_out_weight, kd_feat_weight

    def optimize_parameters(self, current_iter):
        """Enhanced training with progressive KD and better loss design."""
        self.current_iter = current_iter
        
        # Get adaptive weights
        lambda_kd_out, lambda_kd_feat = self.get_adaptive_weights(current_iter)
        
        # Optimize Generator
        for p in self.net_d.parameters():
            p.requires_grad = False
            
        self.optimizer_g.zero_grad()
        
        # Teacher forward (no gradient)
        with torch.no_grad():
            if hasattr(self.net_g, 'module'):
                self.output_a, self.feat_a = self.net_g.module(self.lq_a, return_feats=True)
            else:
                self.output_a, self.feat_a = self.net_g(self.lq_a, return_feats=True)

        # Student forward
        if hasattr(self.net_g, 'module'):
            self.output_b, self.feat_b = self.net_g.module(self.lq_b, return_feats=True)
        else:
            self.output_b, self.feat_b = self.net_g(self.lq_b, return_feats=True)

        l_g_total = 0
        loss_dict = OrderedDict()

        # 1. Pixel Loss (L1)
        if self.cri_pix:
            l_g_pix = self.cri_pix(self.output_b, self.gt_usm)
            l_g_total += l_g_pix
            loss_dict['l_g_pix'] = l_g_pix

        # 2. Perceptual Loss
        if self.cri_perceptual:
            l_g_percep, l_g_style = self.cri_perceptual(self.output_b, self.gt_usm)
            if l_g_percep is not None:
                l_g_total += l_g_percep
                loss_dict['l_g_percep'] = l_g_percep
            if l_g_style is not None:
                l_g_total += l_g_style
                loss_dict['l_g_style'] = l_g_style

        # 3. Knowledge Distillation - Output Level
        l_kd_out = torch.nn.functional.l1_loss(self.output_b, self.output_a.detach())
        l_g_total += lambda_kd_out * l_kd_out
        loss_dict['l_kd_out'] = l_kd_out

        # 4. Knowledge Distillation - Feature Level
        if self.feat_a is not None and self.feat_b is not None:
            l_kd_feat = torch.nn.functional.l1_loss(self.feat_b, self.feat_a.detach())
            l_g_total += lambda_kd_feat * l_kd_feat
            loss_dict['l_kd_feat'] = l_kd_feat

        # 5. Consistency Loss (Teacher should also produce good results)
        if current_iter > self.warmup_kd_iters // 2:
            l_consistency = torch.nn.functional.l1_loss(self.output_a, self.gt_usm)
            l_g_total += self.lambda_consistency * l_consistency
            loss_dict['l_consistency'] = l_consistency

        # 6. GAN Loss (after warmup)
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            fake_g_pred = self.net_d(self.output_b)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

        l_g_total.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), max_norm=1.0)
        
        self.optimizer_g.step()

        # Optimize Discriminator (after warmup)
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            for p in self.net_d.parameters():
                p.requires_grad = True

            self.optimizer_d.zero_grad()
            
            # Real
            real_d_pred = self.net_d(self.gt_usm)
            l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
            loss_dict['l_d_real'] = l_d_real
            loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
            l_d_real.backward()
            
            # Fake
            fake_d_pred = self.net_d(self.output_b.detach().clone())
            l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
            loss_dict['l_d_fake'] = l_d_fake
            loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
            l_d_fake.backward()
            
            self.optimizer_d.step()

        # EMA update
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        # Store loss values
        loss_dict['lambda_kd_out'] = lambda_kd_out
        loss_dict['lambda_kd_feat'] = lambda_kd_feat
        self.log_dict = self.reduce_loss_dict(loss_dict)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """Enhanced validation with multiple metrics."""
        # Disable synthetic degradation during validation
        self.is_train = False
        
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):
                self.metric_results = {}
            self.metric_results[dataset_name] = {}
            for metric_name in self.opt['val']['metrics'].keys():
                self.metric_results[dataset_name][metric_name] = []

        pbar = None
        if use_pbar:
            from tqdm import tqdm
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            
            # For validation, use LQ as input (simulate real-world scenario)
            val_data['lq_b'] = val_data['lq'] 
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data = dict()
            
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img

            # Calculate metrics
            if with_metrics and 'gt' in visuals:
                for name, opt_ in self.opt['val']['metrics'].items():
                    if name == 'psnr':
                        psnr_val = calculate_psnr(sr_img, gt_img, crop_border=opt_.get('crop_border', 0))
                        self.metric_results[dataset_name][name].append(psnr_val)
                    elif name == 'ssim':
                        ssim_val = calculate_ssim(sr_img, gt_img, crop_border=opt_.get('crop_border', 0))
                        self.metric_results[dataset_name][name].append(ssim_val)

            # Save images
            if save_img:
                save_img_dir = osp.join(self.opt['path']['visualization'], dataset_name)
                os.makedirs(save_img_dir, exist_ok=True)
                save_img_path = osp.join(save_img_dir, f'{img_name}_{current_iter}.png')
                imwrite(sr_img, save_img_path)

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')

        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric_name, values in self.metric_results[dataset_name].items():
                if values:  # Check if list is not empty
                    avg_val = sum(values) / len(values)
                    
                    # Log to console
                    print(f'{dataset_name}_{metric_name}: {avg_val:.4f}')
                    
                    # Log to tensorboard
                    if tb_logger:
                        tb_logger.add_scalar(f'metrics/{dataset_name}_{metric_name}', avg_val, current_iter)

        # Set back to training mode
        self.is_train = True

    def test(self):
        """Test function using student branch."""
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                if hasattr(self.net_g_ema, 'module'):
                    self.output = self.net_g_ema.module(self.lq_b)
                else:
                    self.output = self.net_g_ema(self.lq_b)
        else:
            self.net_g.eval()
            with torch.no_grad():
                if hasattr(self.net_g, 'module'):
                    self.output = self.net_g.module(self.lq_b)  
                else:
                    self.output = self.net_g(self.lq_b)
            self.net_g.train()

    def get_current_visuals(self):
        """Return current visual results."""
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq_b.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict