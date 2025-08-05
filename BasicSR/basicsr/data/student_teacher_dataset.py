import numpy as np
import random
import torch
from pathlib import Path
from torch.utils import data as data

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


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
                                               ['lq', 'gt'], opt['filename_tmpl'])
        
        # Teacher paths nếu có folder riêng
        if self.lq_teacher_folder:
            if 'meta_info_file' in opt and opt['meta_info_file'] is not None:
                self.teacher_paths = paired_paths_from_lmdb([self.lq_teacher_folder, self.gt_folder], 
                                                          ['lq', 'gt'], opt['meta_info_file'])
            else:
                self.teacher_paths = paired_paths_from_folder([self.lq_teacher_folder, self.gt_folder], 
                                                           ['lq', 'gt'], opt['filename_tmpl'])

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
            img_gt, img_lq_student = augment([img_gt, img_lq_student], self.opt['use_hflip'], self.opt['use_rot'])
            img_gt, img_lq_teacher = augment([img_gt, img_lq_teacher], self.opt['use_hflip'], self.opt['use_rot'])

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