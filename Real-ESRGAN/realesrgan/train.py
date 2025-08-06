import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sys
sys.path.insert(0, '/home/namlh/DLEnhance_ImageQuality/Real-ESRGAN')
import os.path as osp
from basicsr.train import train_pipeline

import realesrgan.archs
import realesrgan.data
import realesrgan.models

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)