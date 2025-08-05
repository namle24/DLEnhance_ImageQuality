from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

import os
import cv2
from torch.utils.data import Dataset

@DATASET_REGISTRY.register()
class PairedImageDatasetSiamese(Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.gt_folder = opt['dataroot_gt']
        self.lq_a_folder = opt['dataroot_lq_a']
        self.lq_b_folder = opt['dataroot_lq_b']

        self.paths_gt = sorted([os.path.join(self.gt_folder, v) for v in os.listdir(self.gt_folder)])
        self.paths_lq_a = sorted([os.path.join(self.lq_a_folder, v) for v in os.listdir(self.lq_a_folder)])
        self.paths_lq_b = sorted([os.path.join(self.lq_b_folder, v) for v in os.listdir(self.lq_b_folder)])

        assert len(self.paths_gt) == len(self.paths_lq_a) == len(self.paths_lq_b), 'Số lượng ảnh không khớp!'

    def __getitem__(self, index):
        gt_path = self.paths_gt[index]
        lq_a_path = self.paths_lq_a[index]
        lq_b_path = self.paths_lq_b[index]

        gt_img = cv2.imread(gt_path, cv2.IMREAD_COLOR)
        lq_a_img = cv2.imread(lq_a_path, cv2.IMREAD_COLOR)
        lq_b_img = cv2.imread(lq_b_path, cv2.IMREAD_COLOR)

        # augmentation
        if self.opt.get('phase') == 'train':
            gt_img, lq_a_img = augment([gt_img, lq_a_img], use_flip=self.opt.get('use_hflip', True), use_rot=self.opt.get('use_rot', True))
            _, lq_b_img = augment([gt_img, lq_b_img], use_flip=self.opt.get('use_hflip', True), use_rot=self.opt.get('use_rot', True))

        # To tensor
        gt_tensor = img2tensor(gt_img, bgr2rgb=True, float32=True)
        lq_a_tensor = img2tensor(lq_a_img, bgr2rgb=True, float32=True)
        lq_b_tensor = img2tensor(lq_b_img, bgr2rgb=True, float32=True)

        # Normalize
        normalize(gt_tensor, [0.5], [0.5], inplace=True)
        normalize(lq_a_tensor, [0.5], [0.5], inplace=True)
        normalize(lq_b_tensor, [0.5], [0.5], inplace=True)

        return {
            'gt': gt_tensor,
            'lq_a': lq_a_tensor,
            'lq_b': lq_b_tensor
        }

    def __len__(self):
        return len(self.paths_gt)
