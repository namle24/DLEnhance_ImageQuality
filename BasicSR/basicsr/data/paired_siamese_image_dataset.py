import os
from basicsr.data.paired_image_dataset import PairedImageDataset
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.data.data_util import paths_from_folder
import cv2
import numpy as np
import random
import torch
from torchvision.transforms.functional import normalize

@DATASET_REGISTRY.register()
class PairedImageDatasetSiamese(BaseDataset):
    def __init__(self, opt):
        super().__init__(opt)
        self.paths_lq_a = paths_from_folder(self.opt['dataroot_lq_a'])
        self.paths_lq_b = paths_from_folder(self.opt['dataroot_lq_b'])
        self.paths_gt = paths_from_folder(self.opt['dataroot_gt'])

        assert len(self.paths_lq_a) == len(self.paths_gt), "LQ_A and GT mismatch"
        assert len(self.paths_lq_b) == len(self.paths_gt), "LQ_B and GT mismatch"

    def __getitem__(self, index):
        # get paths
        lq_a_path = self.paths_lq_a[index]
        lq_b_path = self.paths_lq_b[index]
        gt_path = self.paths_gt[index]

        # read images
        img_lq_a = cv2.imread(lq_a_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img_lq_b = cv2.imread(lq_b_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img_gt = cv2.imread(gt_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.

        # Convert BGR to RGB
        img_lq_a = img_lq_a[:, :, ::-1]
        img_lq_b = img_lq_b[:, :, ::-1]
        img_gt = img_gt[:, :, ::-1]

        # Resize or crop (tuỳ config), augment
        if self.opt['phase'] == 'train':
            # random crop
            gt_size = self.opt['gt_size']
            h, w = img_gt.shape[0:2]
            rnd_h = random.randint(0, max(0, h - gt_size))
            rnd_w = random.randint(0, max(0, w - gt_size))
            img_gt = img_gt[rnd_h:rnd_h + gt_size, rnd_w:rnd_w + gt_size, :]
            img_lq_a = img_lq_a[rnd_h:rnd_h + gt_size, rnd_w:rnd_w + gt_size, :]
            img_lq_b = img_lq_b[rnd_h:rnd_h + gt_size, rnd_w:rnd_w + gt_size, :]

            # flip, rotate
            if random.random() < 0.5:
                img_gt = np.fliplr(img_gt).copy()
                img_lq_a = np.fliplr(img_lq_a).copy()
                img_lq_b = np.fliplr(img_lq_b).copy()

            if random.random() < 0.5:
                img_gt = np.flipud(img_gt).copy()
                img_lq_a = np.flipud(img_lq_a).copy()
                img_lq_b = np.flipud(img_lq_b).copy()

        # To Tensor
        img_gt = torch.from_numpy(np.ascontiguousarray(np.transpose(img_gt, (2, 0, 1)))).float()
        img_lq_a = torch.from_numpy(np.ascontiguousarray(np.transpose(img_lq_a, (2, 0, 1)))).float()
        img_lq_b = torch.from_numpy(np.ascontiguousarray(np.transpose(img_lq_b, (2, 0, 1)))).float()

        return {
            'lq_a': img_lq_a,
            'lq_b': img_lq_b,
            'gt': img_gt,
            'lq_a_path': lq_a_path,
            'lq_b_path': lq_b_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths_gt)
