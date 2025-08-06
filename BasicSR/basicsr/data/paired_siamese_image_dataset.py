import random
import cv2
import numpy as np
import torch
import torch.utils.data as data
from basicsr.data.paired_image_dataset import PairedImageDataset
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import img2tensor, scandir
from basicsr.data.util import imfrombytes, FileClient

@DATASET_REGISTRY.register()
class PairedSiameseImageDataset(PairedImageDataset):
    def __init__(self, opt):
        print(">>> Loading PAIRED SIAMESE IMAGE DATASET")
        # Gọi init của PairedImageDataset để setup self.paths_gt, self.paths_lq
        super().__init__(opt)
        self.paths_lq_a = self.get_image_paths(opt['io_backend'], opt['dataroot_lq_a'])
        self.paths_lq_b = self.get_image_paths(opt['io_backend'], opt['dataroot_lq_b'])

        assert len(self.paths_gt) == len(self.paths_lq_a) == len(self.paths_lq_b), \
            f"Mismatch: GT={len(self.paths_gt)}, LQ_A={len(self.paths_lq_a)}, LQ_B={len(self.paths_lq_b)}"

    def __getitem__(self, index):
        gt_path = self.paths_gt[index]
        lq_a_path = self.paths_lq_a[index]
        lq_b_path = self.paths_lq_b[index]

        img_gt = imfrombytes(self.file_client.get(gt_path), float32=True)
        img_lq_a = imfrombytes(self.file_client.get(lq_a_path), float32=True)
        img_lq_b = imfrombytes(self.file_client.get(lq_b_path), float32=True)

        # Paired crop
        img_gt, img_lq_a = paired_random_crop(img_gt, img_lq_a, self.opt['gt_size'], self.opt['scale'])
        _, img_lq_b = paired_random_crop(img_gt, img_lq_b, self.opt['gt_size'], self.opt['scale'])

        # Augment
        img_gt, img_lq_a, img_lq_b = augment([img_gt, img_lq_a, img_lq_b], self.opt['use_hflip'], self.opt['use_rot'])

        # To tensor
        img_gt, img_lq_a, img_lq_b = img2tensor([img_gt, img_lq_a, img_lq_b], bgr2rgb=True, float32=True)

        return {
            'gt': img_gt,
            'lq_a': img_lq_a,
            'lq_b': img_lq_b,
            'gt_path': gt_path
        }
