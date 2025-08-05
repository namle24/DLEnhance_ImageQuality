from .models.siamese_model import SiameseESRGAN
from losses import L1Loss, PerceptualLoss, GANLoss, DistillationLoss
from datasets.siamese_paired_dataset import SiamesePairedDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from utils.ema import ExponentialMovingAverage  # Cần tạo file utils/ema.py
import os
import torch


class SiameseTrainer:
    def __init__(self, model, opt_g_teacher, opt_g_student, opt_d, config):
        self.model = model
        self.opt_g_teacher = opt_g_teacher
        self.opt_g_student = opt_g_student
        self.opt_d = opt_d
        self.config = config
        self.steps = 0  # Thiếu biến đếm steps

        # Khởi tạo loss functions từ config
        self.l1_loss = L1Loss()
        self.perceptual_loss = PerceptualLoss(
            layer_weights=config['train']['perceptual_opt']['layer_weights'],
            use_input_norm=config['train']['perceptual_opt']['use_input_norm']
        )
        self.gan_loss = GANLoss(gan_type=config['train']['gan_opt']['gan_type'])
        self.distill_loss = DistillationLoss()

        # EMA và logging
        self.ema = ExponentialMovingAverage(self.model.teacher.parameters(),
                                            decay=config['train']['ema_decay'])
        self.save_dir = config['path']['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=config['path']['log_dir'])

        if 'val' in config['datasets']:
            val_dataset = SiamesePairedDataset({
                'dataroot_gt': config['datasets']['val']['dataroot_gt'],
                'dataroot_lq': config['datasets']['val']['dataroot_low'],
                'gt_size': config['datasets']['train']['gt_size']  # Dùng size giống train
            })
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False
            )

    def validate(self):
        if not hasattr(self, 'val_loader'):
            return {}

        self.model.eval()
        val_metrics = {'psnr': 0, 'ssim': 0}
        with torch.no_grad():
            for batch in self.val_loader:
                gt = batch["gt"].to(self.model.device)
                low = batch["low"].to(self.model.device)
                out_teacher, _ = self.model(low, low)  # Dùng low làm cả hai đầu vào

                # Tính metrics
                val_metrics['psnr'] += calculate_psnr(out_teacher, gt)
                val_metrics['ssim'] += calculate_ssim(out_teacher, gt)

        # Logging
        for metric, value in val_metrics.items():
            self.writer.add_scalar(f'Val/{metric}', value / len(self.val_loader), self.steps)

        self.model.train()
        return val_metrics

    def save_checkpoint(self, iter):
        path = os.path.join(self.save_dir, f"iter_{iter}.pth")
        state = {
            'teacher': self.model.teacher.state_dict(),
            'student': self.model.student.state_dict(),
            'ema': self.ema.state_dict(),
            'optimizer_g_teacher': self.opt_g_teacher.state_dict(),
            'optimizer_g_student': self.opt_g_student.state_dict(),
            'iter': iter
        }
        torch.save(state, path)

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        for batch in dataloader:
            self.steps += 1
            loss = self.train_step(batch)

            # Logging và lưu checkpoint
            if self.steps % self.config['path']['save_checkpoint_freq'] == 0:
                self.save_checkpoint(self.steps)
                self.validate()  # Thêm validation nếu cần

    def train_step(self, batch):
        # Chuyển dữ liệu lên GPU
        gt = batch["gt"].to(self.model.device)
        low = batch["low"].to(self.model.device)
        very_low = batch["very_low"].to(self.model.device)

        # Forward pass
        out_teacher, out_student = self.model(low, very_low)

        # Tính loss
        l1_loss_teacher = self.l1_loss(out_teacher, gt)
        percep_loss_teacher = self.perceptual_loss(out_teacher, gt)
        loss_teacher = l1_loss_teacher * self.config['train']['pixel_weight'] + \
                       percep_loss_teacher * self.config['train']['perceptual_weight']

        l1_loss_student = self.l1_loss(out_student, gt)
        distill_loss = self.distill_loss(out_student, out_teacher.detach())
        loss_student = l1_loss_student * self.config['train']['pixel_weight'] + \
                       distill_loss * self.config['train']['distillation_weight']

        # GAN loss
        fake_teacher = self.model.discriminator(out_teacher)
        fake_student = self.model.discriminator(out_student)
        loss_g_gan = self.gan_loss(fake_teacher, True) * self.config['train']['gan_weight'] + \
                     self.gan_loss(fake_student, True) * self.config['train']['gan_weight']

        # Backward và optimize
        self.opt_g_teacher.zero_grad()
        loss_teacher.backward(retain_graph=True)
        self.opt_g_teacher.step()

        self.opt_g_student.zero_grad()
        (loss_student + loss_g_gan).backward()
        self.opt_g_student.step()

        # Cập nhật EMA
        self.ema.update()

        # Discriminator update
        real_logits = self.model.discriminator(gt)
        fake_logits_teacher = self.model.discriminator(out_teacher.detach())
        fake_logits_student = self.model.discriminator(out_student.detach())

        loss_d_real = self.gan_loss(real_logits, True)
        loss_d_fake = (self.gan_loss(fake_logits_teacher, False) +
                       self.gan_loss(fake_logits_student, False)) / 2
        loss_d = (loss_d_real + loss_d_fake) * 0.5

        self.opt_d.zero_grad()
        loss_d.backward()
        self.opt_d.step()

        # Logging
        if self.steps % 100 == 0:
            self.writer.add_scalar('Loss/teacher', loss_teacher.item(), self.steps)
            self.writer.add_scalar('Loss/student', loss_student.item(), self.steps)
            self.writer.add_scalar('Loss/discriminator', loss_d.item(), self.steps)

        return {'loss_teacher': loss_teacher.item(),
                'loss_student': loss_student.item(),
                'loss_d': loss_d.item()}