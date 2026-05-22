import torch
from torch import nn as nn
from torch.nn import functional as F
import importlib.util
import os

# Load PDMAE by absolute file path (avoids `archs` name clashes on sys.path).
_proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
_pd_mae_mae_arch = os.path.join(_proj_root, 'PD_MAE_SR', 'archs', 'mae_arch.py')

_pdmae_import_err = None
PDMAE = None
if os.path.isfile(_pd_mae_mae_arch):
    try:
        _spec = importlib.util.spec_from_file_location('pd_mae_sr_mae_arch', _pd_mae_mae_arch)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        PDMAE = _mod.PDMAE
    except Exception as e:
        _pdmae_import_err = e
else:
    _pdmae_import_err = FileNotFoundError(_pd_mae_mae_arch)

if PDMAE is None:
    print(f'Warning: Could not import PDMAE from {_pd_mae_mae_arch}: {_pdmae_import_err}')

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import default_init_weights, make_layer, pixel_unshuffle


class SFT_Layer(nn.Module):
    def __init__(self, rrdb_channels=64, mae_channels=384):
        super().__init__()
        self.scale_conv = nn.Conv2d(mae_channels, rrdb_channels, 1)
        self.shift_conv = nn.Conv2d(mae_channels, rrdb_channels, 1)
    
    def forward(self, rrdb_feat, mae_feat):
        mae_resized = F.interpolate(mae_feat,
                                    size=rrdb_feat.shape[-2:],
                                    mode='bilinear',
                                    align_corners=False)
        scale = self.scale_conv(mae_resized)
        shift = self.shift_conv(mae_resized)
        return rrdb_feat * (1 + scale) + shift


class PD_MAE_Encoder_Wrapper(nn.Module):
    """Load Stage 2 checkpoint, extract spatial features cho SFT"""
    def __init__(self, checkpoint_path=None, img_size=256, patch_size=8):
        super().__init__()
        if PDMAE is None:
            raise ImportError(
                f'PDMAE class is not available. Expected mae_arch at: {_pd_mae_mae_arch}. '
                f'Import error: {_pdmae_import_err}'
            ) from _pdmae_import_err
        self.mae = PDMAE(img_size=img_size, patch_size=patch_size)
        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            if 'model_state_dict' in ckpt:
                self.mae.load_state_dict(ckpt['model_state_dict'])
            else:
                self.mae.load_state_dict(ckpt)
        else:
            print(f"Warning: MAE checkpoint not found at {checkpoint_path}. Using uninitialized weights.")
        
        # Frozen hoàn toàn — không train lại
        for param in self.mae.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # Resize LQ input to HR size (256x256) for MAE Encoder if needed
        # since MAE was pretrained on HR patches (256x256)
        # Expected input shape: [B, C, H, W]
        img_size = int((self.mae.encoder.num_patches) ** 0.5 * self.mae.encoder.patch_size)
        if x.shape[-2] != img_size or x.shape[-1] != img_size:
            x_resized = F.interpolate(x, size=(img_size, img_size), mode='bicubic', align_corners=False)
        else:
            x_resized = x
            
        # Extract encoder features (không qua decoder)
        # Output: [B, num_patches+1, 384] → reshape về spatial map
        feat = self.mae.encoder(x_resized)  # [B, L+1, 384]
        feat = feat[:, 1:, :]       # Bỏ cls token → [B, L, 384]
        
        # Reshape về spatial: [B, 384, H/p, W/p]
        h = w = int(feat.shape[1] ** 0.5)
        feat = feat.permute(0, 2, 1).reshape(
            feat.shape[0], 384, h, w
        )
        return feat



class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Empirically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Empirically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x


@ARCH_REGISTRY.register()
class RRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32, use_sft=False, mae_checkpoint=None, img_size=256):
        super(RRDBNet, self).__init__()
        self.scale = scale
        self.use_sft = use_sft
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        
        self.body = nn.ModuleList([RRDB(num_feat, num_grow_ch=num_grow_ch) for _ in range(num_block)])
        
        if self.use_sft:
            self.mae_wrapper = PD_MAE_Encoder_Wrapper(checkpoint_path=mae_checkpoint, img_size=img_size)
            self.sft_layers = nn.ModuleDict({
                str(i): SFT_Layer(rrdb_channels=num_feat, mae_channels=384) 
                for i in [4, 9, 14, 19]
            })
            
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
            
        mae_feat = None
        if self.use_sft:
            mae_feat = self.mae_wrapper(x)
            
        feat = self.conv_first(feat)
        
        body_feat = feat
        for i, block in enumerate(self.body):
            body_feat = block(body_feat)
            if self.use_sft and str(i) in self.sft_layers:
                body_feat = self.sft_layers[str(i)](body_feat, mae_feat)
                
        body_feat = self.conv_body(body_feat)
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out