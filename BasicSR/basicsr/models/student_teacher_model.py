import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel


@MODEL_REGISTRY.register()
class StudentTeacherModel(SRModel):
    """Student-Teacher Model cho RealESRGAN training"""

    def __init__(self, opt):
        super(StudentTeacherModel, self).__init__(opt)
        
        # Build teacher network
        if self.is_train:
            self.net_teacher = build_network(opt['network_teacher'])
            self.net_teacher = self.model_to_device(self.net_teacher)
            self.print_network(self.net_teacher, 'Teacher')
            
            # Load pretrained teacher nếu có
            load_path = self.opt['path'].get('pretrain_network_teacher', None)
            if load_path is not None:
                param_key = self.opt['path'].get('param_key_teacher', 'params')
                self.load_network(self.net_teacher, load_path, self.opt['path'].get('strict_load_teacher', True), param_key)
        
        # Student network đã được khởi tạo trong parent class (net_g)
        
        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        super().init_training_settings()
        
        # Setup teacher optimizer
        if hasattr(self, 'net_teacher'):
            train_opt = self.opt['train']
            self.setup_teacher_optimizers(train_opt)
            self.setup_teacher_schedulers()
            
            # Distillation loss
            distill_opt = train_opt.get('distillation_loss')
            if distill_opt:
                self.distill_loss_weight = distill_opt.get('loss_weight', 0.5)
                self.temperature = distill_opt.get('temperature', 4.0)
            else:
                self.distill_loss_weight = 0.5
                self.temperature = 4.0
                
            # Feature loss
            feature_opt = train_opt.get('feature_loss')
            if feature_opt:
                self.feature_loss_weight = feature_opt.get('loss_weight', 0.1)
                self.cri_feature = build_loss(feature_opt).to(self.device)
            else:
                self.feature_loss_weight = 0.1
                self.cri_feature = nn.MSELoss().to(self.device)

    def setup_teacher_optimizers(self, train_opt):
        """Setup teacher optimizer"""
        optim_params_teacher = []
        for k, v in self.net_teacher.named_parameters():
            if v.requires_grad:
                optim_params_teacher.append(v)

        optim_type = train_opt['teacher_optim'].pop('type')
        self.optimizer_teacher = self.get_optimizer(optim_type, optim_params_teacher, **train_opt['teacher_optim'])
        self.optimizers.append(self.optimizer_teacher)

    def setup_teacher_schedulers(self):
        """Setup teacher scheduler"""
        train_opt = self.opt['train']
        scheduler_type = train_opt['teacher_scheduler'].pop('type')
        
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            self.scheduler_teacher = self.get_scheduler(scheduler_type, self.optimizer_teacher, **train_opt['teacher_scheduler'])
        else:
            raise NotImplementedError(f'Scheduler {scheduler_type} is not implemented yet.')
        
        self.schedulers.append(self.scheduler_teacher)

    def extract_features(self, network, x):
        """Extract features từ network"""
        features = []
        
        def hook_fn(module, input, output):
            features.append(output)
        
        # Register hooks cho conv layers
        hooks = []
        for name, module in network.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)) and 'final' not in name:
                hooks.append(module.register_forward_hook(hook_fn))
        
        # Forward pass
        output = network(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
        return output, features

    def compute_distillation_loss(self, student_output, teacher_output):
        """Compute knowledge distillation loss"""
        # Ensure same size
        if student_output.shape != teacher_output.shape:
            teacher_output = F.interpolate(teacher_output, size=student_output.shape[2:], mode='bilinear', align_corners=False)
        
        # KL divergence loss with temperature
        teacher_soft = F.softmax(teacher_output / self.temperature, dim=1)
        student_log_soft = F.log_softmax(student_output / self.temperature, dim=1)
        
        distill_loss = F.kl_div(student_log_soft, teacher_soft, reduction='batchmean')
        distill_loss *= (self.temperature ** 2)
        
        return distill_loss

    def compute_feature_loss(self, student_features, teacher_features):
        """Compute feature matching loss"""
        feature_loss = 0.0
        min_len = min(len(student_features), len(teacher_features))
        
        for i in range(min_len):
            s_feat = student_features[i]
            t_feat = teacher_features[i].detach()  # Detach teacher features
            
            # Resize if needed
            if s_feat.shape != t_feat.shape:
                if s_feat.numel() > t_feat.numel():
                    s_feat = F.adaptive_avg_pool2d(s_feat, t_feat.shape[2:])
                else:
                    t_feat = F.adaptive_avg_pool2d(t_feat, s_feat.shape[2:])
            
            feature_loss += self.cri_feature(s_feat, t_feat)
        
        return feature_loss / min_len if min_len > 0 else torch.tensor(0.0).to(self.device)

    def optimize_parameters(self, current_iter):
        # Train teacher first
        if hasattr(self, 'net_teacher'):
            self.net_teacher.train()
            self.optimizer_teacher.zero_grad()
            
            # Teacher forward với teacher data (data được load trong feed_data)
            teacher_output = self.net_teacher(self.lq_teacher)
            teacher_loss = self.cri_pix(teacher_output, self.gt)
            
            teacher_loss.backward()
            self.optimizer_teacher.step()
            
            # Log teacher loss
            self.log_dict['teacher_pix_loss'] = teacher_loss.item()
        
        # Train student
        self.net_g.train()
        if hasattr(self, 'net_teacher'):
            self.net_teacher.eval()
        
        self.optimizer_g.zero_grad()
        
        # Student forward
        student_output, student_features = self.extract_features(self.net_g, self.lq)
        
        # Student pixel loss
        student_pix_loss = self.cri_pix(student_output, self.gt)
        student_loss = student_pix_loss
        
        # Distillation loss nếu có teacher
        if hasattr(self, 'net_teacher') and self.distill_loss_weight > 0:
            with torch.no_grad():
                teacher_output_for_student, teacher_features = self.extract_features(self.net_teacher, self.lq)
            
            distill_loss = self.compute_distillation_loss(student_output, teacher_output_for_student)
            student_loss += self.distill_loss_weight * distill_loss
            self.log_dict['distill_loss'] = distill_loss.item()
            
            # Feature loss
            if self.feature_loss_weight > 0:
                feature_loss = self.compute_feature_loss(student_features, teacher_features)
                student_loss += self.feature_loss_weight * feature_loss
                self.log_dict['feature_loss'] = feature_loss.item()
        
        student_loss.backward()
        self.optimizer_g.step()
        
        # Log losses
        self.log_dict['student_pix_loss'] = student_pix_loss.item()
        self.log_dict['total_student_loss'] = student_loss.item()
        
        self.output = student_output

    def feed_data(self, data):
        """Feed data cho cả teacher và student"""
        self.lq = data['lq'].to(self.device)
        self.gt = data['gt'].to(self.device)
        
        # Teacher data (nếu có trong data)
        if 'lq_teacher' in data:
            self.lq_teacher = data['lq_teacher'].to(self.device)
        else:
            # Fallback: dùng chung lq cho teacher
            self.lq_teacher = self.lq

    def save(self, epoch, current_iter):
        """Override save để lưu cả teacher và student"""
        if hasattr(self, 'net_teacher'):
            self.save_network([self.net_g, self.net_teacher], ['net_g', 'net_teacher'], current_iter)
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
