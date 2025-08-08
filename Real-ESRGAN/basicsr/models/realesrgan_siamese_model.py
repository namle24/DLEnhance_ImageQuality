import torch
from basicsr.utils.registry import MODEL_REGISTRY
from realesrgan.models.realesrgan_model import RealESRGANModel

@MODEL_REGISTRY.register()
class RealESRGANSiameseModel(RealESRGANModel):
    def feed_data(self, data):
        self.lq_a = data['lq_a'].to(self.device)
        self.lq_b = data['lq_b'].to(self.device)
        self.gt = data['gt'].to(self.device)

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

    @torch.no_grad()
    def validation(self, dataloader, current_iter, tb_logger, save_img=False):
        self.net_g.eval()  # Đánh giá student
        
        avg_psnr = 0.0
        avg_ssim = 0.0
        cnt = 0

        for val_data in dataloader:
            lq = val_data['lq'].to(self.device)  # LQ_B từ dataset chuẩn
            gt = val_data['gt'].to(self.device)

            output = self.net_g(lq)  # CHỈ evaluate student

            crop_border = self.opt['val'].get('crop_border', 4)
            avg_psnr += calculate_psnr(output, gt, crop_border=crop_border)
            avg_ssim += calculate_ssim(output, gt, crop_border=crop_border)
            cnt += 1

            if save_img:
                save_img_path = osp.join(
                    self.opt['path']['visualization'],
                    'val_images',
                    f'{current_iter:08d}_{cnt:03d}.png'
                )
                imwrite(tensor2img(output), save_img_path)

        # Log metrics
        self.log_dict['val/psnr'] = avg_psnr / cnt
        self.log_dict['val/ssim'] = avg_ssim / cnt

        if tb_logger:
            tb_logger.add_scalar('val/psnr', avg_psnr / cnt, current_iter)
            tb_logger.add_scalar('val/ssim', avg_ssim / cnt, current_iter)

        self.net_g.train()