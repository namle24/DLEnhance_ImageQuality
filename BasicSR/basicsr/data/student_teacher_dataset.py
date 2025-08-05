import numpy as np
import random
import torch
from pathlib import Path
from torch.utils import data as data
import cv2
import os

try:
    from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb
    from basicsr.data.transforms import augment, paired_random_crop
    from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
    from basicsr.utils.registry import DATASET_REGISTRY
    BASICSR_AVAILABLE = True
except ImportError:
    try:
        # Alternative imports for different BasicSR versions
        from basicsr.data.data_util import paired_paths_from_folder
        from basicsr.data.transforms import augment, paired_random_crop  
        from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
        from basicsr.utils.registry import DATASET_REGISTRY
        
        # Fallback for missing function
        def paired_paths_from_lmdb(folders, keys, meta_info_file):
            return paired_paths_from_folder(folders, keys, '{}.png')
        
        BASICSR_AVAILABLE = True
    except ImportError:
        print("Warning: BasicSR not properly installed, using fallback implementations")
        BASICSR_AVAILABLE = False
        
        # Fallback implementations
        class FileClient:
            def __init__(self, backend='disk', **kwargs):
                self.backend = backend
                
            def get(self, filepath, key=None):
                with open(filepath, 'rb') as f:
                    return f.read()
        
        def paired_paths_from_folder(folders, keys, filename_tmpl):
            """Simple implementation for paired paths"""
            paths = []
            lq_folder, gt_folder = folders
            
            lq_files = sorted([f for f in os.listdir(lq_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
            
            for lq_file in lq_files:
                gt_file = lq_file  # Assume same filename
                paths.append({
                    'lq_path': os.path.join(lq_folder, lq_file),
                    'gt_path': os.path.join(gt_folder, gt_file)
                })
            return paths
        
        def paired_paths_from_lmdb(folders, keys, meta_info_file):
            return paired_paths_from_folder(folders, keys, '{}.png')
            
        def imfrombytes(content, float32=False):
            """Read image from bytes"""
            img_np = np.frombuffer(content, np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            if float32:
                img = img.astype(np.float32) / 255.0
            return img
            
        def img2tensor(imgs, bgr2rgb=True, float32=True):
            """Convert images to tensor"""
            def _totensor(img):
                if img.shape[2] == 3 and bgr2rgb:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = torch.from_numpy(img.transpose(2, 0, 1))
                if float32:
                    img = img.float()
                return img
            
            if isinstance(imgs, list):
                return [_totensor(img) for img in imgs]
            else:
                return _totensor(imgs)
        
        def bgr2ycbcr(img, y_only=False):
            """Convert BGR to YCbCr"""
            img_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            if y_only:
                return img_ycbcr[:, :, 0:1]
            return img_ycbcr
            
        def augment(imgs, hflip=True, rotation=True):
            """Simple augmentation"""
            if hflip and random.random() < 0.5:
                imgs = [cv2.flip(img, 1) for img in imgs]
            
            if rotation and random.random() < 0.5:
                k = random.randint(1, 3)
                imgs = [np.rot90(img, k) for img in imgs]
            
            return imgs
            
        def paired_random_crop(img_gts, img_lqs, gt_patch_size, scale, gt_path=None):
            """Random crop for paired images"""
            if not isinstance(img_gts, list):
                img_gts = [img_gts]
            if not isinstance(img_lqs, list):
                img_lqs = [img_lqs]
                
            h_lq, w_lq, _ = img_lqs[0].shape
            h_gt, w_gt, _ = img_gts[0].shape
            lq_patch_size = gt_patch_size // scale

            if h_gt != h_lq * scale or w_gt != w_lq * scale:
                raise ValueError('Scale mismatch')

            if h_lq < lq_patch_size or w_lq < lq_patch_size:
                # Pad if too small
                pad_h = max(0, lq_patch_size - h_lq)
                pad_w = max(0, lq_patch_size - w_lq)
                img_lqs = [np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), 'reflect') for img in img_lqs]
                img_gts = [np.pad(img, ((0, pad_h*scale), (0, pad_w*scale), (0, 0)), 'reflect') for img in img_gts]
                h_lq, w_lq = img_lqs[0].shape[:2]

            # Random crop
            top = random.randint(0, h_lq - lq_patch_size)
            left = random.randint(0, w_lq - lq_patch_size)

            img_lqs = [img[top:top + lq_patch_size, left:left + lq_patch_size, ...] for img in img_lqs]
            top_gt, left_gt = int(top * scale), int(left * scale)
            img_gts = [img[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for img in img_gts]

            if len(img_gts) == 1:
                img_gts = img_gts[0]
            if len(img_lqs) == 1:
                img_lqs = img_lqs[0]
            return img_gts, img_lqs
        
        # Mock registry
        class MockRegistry:
            def register(self):
                def decorator(cls):
                    return cls
                return decorator
        
        DATASET_REGISTRY = MockRegistry()


if BASICSR_AVAILABLE:
    @DATASET_REGISTRY.register()
    class StudentTeacherDataset(data.Dataset):
        """Dataset cho Student-Teacher training với 2 loại LQ khác nhau"""

        def __init__(self, opt):
            super(StudentTeacherDataset, self).__init__()
            self.opt = opt
            
            # File client (cho việc đọc file)
            self.file_client = None
            self.io_backend_opt = opt['io_backend']
            
            self.gt_folder = opt['dataroot_gt']
            self.lq_student_folder = opt['dataroot_lq_student']  # LQ cho student (xấu nhiều)
            self.lq_teacher_folder = opt.get('dataroot_lq_teacher', None)  # LQ cho teacher (xấu nhẹ)
            
            # Load paths
            if 'meta_info_file' in opt and opt['meta_info_file'] is not None:
                self.paths = paired_paths_from_lmdb([self.lq_student_folder, self.gt_folder], 
                                                  ['lq', 'gt'], opt['meta_info_file'])
            else:
                self.paths = paired_paths_from_folder([self.lq_student_folder, self.gt_folder], 
                                                   ['lq', 'gt'], opt.get('filename_tmpl', '{}'))
            
            # Teacher paths nếu có folder riêng
            if self.lq_teacher_folder:
                if 'meta_info_file' in opt and opt['meta_info_file'] is not None:
                    self.teacher_paths = paired_paths_from_lmdb([self.lq_teacher_folder, self.gt_folder], 
                                                              ['lq', 'gt'], opt['meta_info_file'])
                else:
                    self.teacher_paths = paired_paths_from_folder([self.lq_teacher_folder, self.gt_folder], 
                                                               ['lq', 'gt'], opt.get('filename_tmpl', '{}'))

        def __getitem__(self, index):
            if self.file_client is None:
                self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

            scale = self.opt['scale']

            # Load GT image
            gt_path = self.paths[index]['gt_path']
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)

            # Load student LQ image
            lq_student_path = self.paths[index]['lq_path']
            img_bytes = self.file_client.get(lq_student_path, 'lq')
            img_lq_student = imfrombytes(img_bytes, float32=True)

            # Load teacher LQ image
            if hasattr(self, 'teacher_paths') and self.lq_teacher_folder:
                lq_teacher_path = self.teacher_paths[index]['lq_path']
                img_bytes = self.file_client.get(lq_teacher_path, 'lq')
                img_lq_teacher = imfrombytes(img_bytes, float32=True)
            else:
                # Nếu không có teacher folder riêng, dùng chung
                img_lq_teacher = img_lq_student.copy()

            # Augmentation for training
            if self.opt['phase'] == 'train':
                gt_size = self.opt['gt_size']
                
                # Random crop and augment
                img_gt, img_lq_student = paired_random_crop(img_gt, img_lq_student, gt_size, scale, gt_path)
                img_gt, img_lq_teacher = paired_random_crop(img_gt, img_lq_teacher, gt_size, scale, gt_path)
                
                # Augmentation
                img_gt, img_lq_student = augment([img_gt, img_lq_student], 
                                                self.opt.get('use_hflip', True), 
                                                self.opt.get('use_rot', True))
                img_gt, img_lq_teacher = augment([img_gt, img_lq_teacher], 
                                               self.opt.get('use_hflip', True), 
                                               self.opt.get('use_rot', True))

            # Color space transform
            if 'color' in self.opt and self.opt['color'] == 'y':
                img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
                img_lq_student = bgr2ycbcr(img_lq_student, y_only=True)[..., None]
                if img_lq_teacher is not None:
                    img_lq_teacher = bgr2ycbcr(img_lq_teacher, y_only=True)[..., None]

            # BGR to RGB, HWC to CHW, numpy to tensor
            img_gt, img_lq_student = img2tensor([img_gt, img_lq_student], bgr2rgb=True, float32=True)
            if img_lq_teacher is not None:
                img_lq_teacher = img2tensor([img_lq_teacher], bgr2rgb=True, float32=True)[0]

            return_dict = {
                'lq': img_lq_student,  # Student input
                'gt': img_gt,
                'lq_path': lq_student_path,
                'gt_path': gt_path
            }
            
            if img_lq_teacher is not None:
                return_dict['lq_teacher'] = img_lq_teacher  # Teacher input
                
            return return_dict

        def __len__(self):
            return len(self.paths)

else:
    # Fallback class when BasicSR is not available
    class StudentTeacherDataset(data.Dataset):
        def __init__(self, opt):
            super(StudentTeacherDataset, self).__init__()
            print("Warning: Using fallback StudentTeacherDataset implementation")
            self.opt = opt
            
        def __getitem__(self, index):
            # Dummy implementation
            return {
                'lq': torch.randn(3, 64, 64),
                'gt': torch.randn(3, 256, 256),
                'lq_teacher': torch.randn(3, 64, 64)
            }
            
        def __len__(self):
            return 100