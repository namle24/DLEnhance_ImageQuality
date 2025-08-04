import numpy as np
import random
import torch
from collections import OrderedDict
from torch.nn import functional as F

from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.losses.loss_util import get_refined_artifact_map
from basicsr.models.srgan_model import SRGANModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.losses import build_loss
from torch.utils.data import Dataset
import os
import cv2
from basicsr.data.transforms import augment
from basicsr.data.utils import img2tensor
from torchvision.transforms.functional import normalize

@MODEL_REGISTRY.register()
class RealESRGANModel_Siamese(SRGANModel):
    def __init__(self, opt):
        super().__init__(opt)

        # Hai generator riêng biệt
        self.net_g_a = self.build_network(opt['network_g_a'])  # ảnh nhẹ suy giảm
        self.net_g_b = self.build_network(opt['network_g_b'])  # ảnh suy giảm nặng

        self.net_g_a.train()
        self.net_g_b.train()

        # Discriminator (1 cho B)
        if self.is_train:
            self.net_d = self.build_network(opt['network_d'])
            self.net_d.train()

        # Loss weights
        self.loss_weights = opt.get('loss_weights', {'l1_a': 1.0, 'l1_b': 1.0, 'distill': 0.1, 'gan': 1e-1, 'percep': 1e-2})

        # Pixel loss
        if self.opt['train'].get('pixel_opt'):
            self.cri_pix = build_loss(self.opt['train']['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        # Perceptual loss
        if self.opt['train'].get('perceptual_opt'):
            self.cri_perceptual = build_loss(self.opt['train']['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        # GAN loss
        if self.opt['train'].get('gan_opt'):
            self.cri_gan = build_loss(self.opt['train']['gan_opt']).to(self.device)
        else:
            self.cri_gan = None

        # Optimizers
        self.optimizer_g = self.build_optimizers([self.net_g_a, self.net_g_b])
        self.optimizer_d = self.build_optimizers(self.net_d) if self.is_train else None

    def feed_data(self, data):
        self.lq_a = data['lq_a'].to(self.device)
        self.lq_b = data['lq_b'].to(self.device)
        self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.output_a = self.net_g_a(self.lq_a)
        self.output_b = self.net_g_b(self.lq_b)

        loss_g_total = 0

        # Pixel losses
        loss_a = self.cri_pix(self.output_a, self.gt)
        loss_b = self.cri_pix(self.output_b, self.gt)
        loss_g_total += self.loss_weights['l1_a'] * loss_a + self.loss_weights['l1_b'] * loss_b

        # Distillation loss
        loss_distill = self.cri_pix(self.output_b, self.output_a.detach())
        loss_g_total += self.loss_weights['distill'] * loss_distill

        # Perceptual loss (only for B)
        if self.cri_perceptual:
            loss_percep = self.cri_perceptual(self.output_b, self.gt)
            loss_g_total += self.loss_weights['percep'] * loss_percep
        else:
            loss_percep = torch.tensor(0.0).to(self.device)

        # GAN loss
        if self.cri_gan:
            pred_g_fake = self.net_d(self.output_b)
            loss_g_gan = self.cri_gan(pred_g_fake, True, is_disc=False)
            loss_g_total += self.loss_weights['gan'] * loss_g_gan
        else:
            loss_g_gan = torch.tensor(0.0).to(self.device)

        # Backward G
        self.optimizer_g.zero_grad()
        loss_g_total.backward()
        self.optimizer_g.step()

        if self.cri_gan:
            self.optimizer_d.zero_grad()
            pred_real = self.net_d(self.gt)
            pred_fake = self.net_d(self.output_b.detach())
            loss_d_real = self.cri_gan(pred_real, True, is_disc=True)
            loss_d_fake = self.cri_gan(pred_fake, False, is_disc=True)
            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            self.optimizer_d.step()

        # Logging
        self.log_dict = {
            'loss_a': loss_a.item(),
            'loss_b': loss_b.item(),
            'loss_distill': loss_distill.item(),
            'loss_percep': loss_percep.item(),
            'loss_g_gan': loss_g_gan.item(),
            'loss_g_total': loss_g_total.item(),
            'output_a_mean': self.output_a.mean().item(),
            'output_b_mean': self.output_b.mean().item(),
            'grad_g_a': self.net_g_a.parameters().__next__().grad.norm().item() if self.net_g_a.parameters().__next__().grad is not None else 0.0,
            'grad_g_b': self.net_g_b.parameters().__next__().grad.norm().item() if self.net_g_b.parameters().__next__().grad is not None else 0.0
        }