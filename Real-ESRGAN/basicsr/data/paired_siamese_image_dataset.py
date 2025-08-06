import os
import random
import cv2
from torch.utils.data import Dataset
from basicsr.utils import FileClient, imfrombytes, img2tensor, scandir
from basicsr.data.transforms import augment
from basicsr.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class PairedSiameseImageDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.gt_folder = opt['dataroot_gt']
        self.lq_folder_a = opt['dataroot_lq_a']
        self.lq_folder_b = opt['dataroot_lq_b']
        self.io_backend_opt = opt['io_backend']

        self.paths_gt = sorted(scandir(self.gt_folder, full_path=True))
        self.paths_lq_a = sorted(scandir(self.lq_folder_a, full_path=True))
        self.paths_lq_b = sorted(scandir(self.lq_folder_b, full_path=True))

        assert len(self.paths_gt) == len(self.paths_lq_a) == len(self.paths_lq_b), \
            f"Mismatch dataset length: GT={len(self.paths_gt)}, A={len(self.paths_lq_a)}, B={len(self.paths_lq_b)}"

        self.file_client = None
        self.mean = opt.get('mean', [0.0, 0.0, 0.0])
        self.std = opt.get('std', [1.0, 1.0, 1.0])
        self.use_flip = opt.get('use_flip', True)
        self.use_rot = opt.get('use_rot', True)
        self.gt_size = opt.get('gt_size', None) or opt.get('gt_size', None) or opt.get('patch_size', None)


    def __len__(self):
        return len(self.paths_gt)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # Lấy đường dẫn ảnh
        gt_path = self.paths_gt[index]
        lq_a_path = self.paths_lq_a[index]
        lq_b_path = self.paths_lq_b[index]

        # Đọc ảnh dạng bytes (dùng FileClient)
        gt_bytes = self.file_client.get(gt_path, 'gt')
        lq_a_bytes = self.file_client.get(lq_a_path, 'lq_a')
        lq_b_bytes = self.file_client.get(lq_b_path, 'lq_b')

        # Chuyển bytes sang ảnh float32
        img_gt = imfrombytes(gt_bytes, float32=True)
        img_lq_a = imfrombytes(lq_a_bytes, float32=True)
        img_lq_b = imfrombytes(lq_b_bytes, float32=True)

        # Kiểm tra ảnh có bị None không
        if img_gt is None:
            raise IOError(f"Cannot decode GT image from: {gt_path}")
        if img_lq_a is None:
            raise IOError(f"Cannot decode LQ_A image from: {lq_a_path}")
        if img_lq_b is None:
            raise IOError(f"Cannot decode LQ_B image from: {lq_b_path}")

        # Augment (nếu có)
        if self.opt['phase'] == 'train':
            gt_size = self.gt_size
            h, w = img_gt.shape[:2]
            if h < gt_size or w < gt_size:
                raise ValueError(f"GT image too small ({h}x{w}) for crop size {gt_size}")

            # random crop
            rnd_h = random.randint(0, h - gt_size)
            rnd_w = random.randint(0, w - gt_size)
            img_gt = img_gt[rnd_h:rnd_h + gt_size, rnd_w:rnd_w + gt_size, :]
            img_lq_a = img_lq_a[rnd_h:rnd_h + gt_size, rnd_w:rnd_w + gt_size, :]
            img_lq_b = img_lq_b[rnd_h:rnd_h + gt_size, rnd_w:rnd_w + gt_size, :]

            # flip, rotate
            img_gt, img_lq_a, img_lq_b = augment(
                [img_gt, img_lq_a, img_lq_b], self.use_flip, self.use_rot)
            
        # Chuyển sang tensor
        img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)
        img_lq_a = img2tensor(img_lq_a, bgr2rgb=True, float32=True)
        img_lq_b = img2tensor(img_lq_b, bgr2rgb=True, float32=True)

        # Normalize
        for t in [img_gt, img_lq_a, img_lq_b]:
            for c in range(3):
                t[c, :, :] = (t[c, :, :] - self.mean[c]) / self.std[c]

        return {
            'gt': img_gt,
            'lq_a': img_lq_a,
            'lq_b': img_lq_b,
            'gt_path': gt_path,
            'lq_a_path': lq_a_path,
            'lq_b_path': lq_b_path
        }

