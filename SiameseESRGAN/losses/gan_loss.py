import torch
import torch.nn as nn


class GANLoss(nn.Module):
    def __init__(self, gan_type='ralsgan', real_label_val=1.0, fake_label_val=0.0):
        super().__init__()
        self.gan_type = gan_type.lower()
        valid_types = ['ralsgan', 'vanilla', 'lsgan']
        if self.gan_type not in valid_types:
            raise ValueError(f"GAN type {gan_type} not supported. Choose from {valid_types}")

        self.register_buffer('real_label', torch.tensor(real_label_val))
        self.register_buffer('fake_label', torch.tensor(fake_label_val))

    def forward(self, pred, target_is_real):
        """
        Args:
            pred (Tensor): Discriminator output
            target_is_real (bool): Whether the target is real (True) or fake (False)
        """
        if not isinstance(target_is_real, bool):
            raise TypeError("target_is_real must be boolean")

        if self.gan_type == 'ralsgan':
            target = self.real_label if target_is_real else self.fake_label
            loss = torch.mean((pred - target) ** 2)

        elif self.gan_type == 'lsgan':
            target = self.real_label if target_is_real else self.fake_label
            loss = 0.5 * torch.mean((pred - target) ** 2)

        elif self.gan_type == 'vanilla':
            loss = torch.nn.functional.softplus(-pred if target_is_real else pred).mean()

        return loss

    def get_target_label(self, pred, target_is_real):
        """Get target label with same shape as input"""
        if target_is_real:
            return self.real_label.expand_as(pred)
        return self.fake_label.expand_as(pred)