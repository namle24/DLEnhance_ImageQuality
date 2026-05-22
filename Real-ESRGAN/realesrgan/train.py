import sys
import os

# Dynamically add Real-ESRGAN root to path (works on any machine)
_real_esrgan_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _real_esrgan_root not in sys.path:
    sys.path.insert(0, _real_esrgan_root)

import os.path as osp
from basicsr.train import train_pipeline

import realesrgan.archs
import realesrgan.data
import realesrgan.models

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)