import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from basicsr.data import build_dataloader
from basicsr.models import build_model
from basicsr.train import parse_options
from basicsr.utils.logger import create_logger  # Updated import for BasicSR 1.4.2

def main():
    opt = parse_options(is_train=True)
    logger, tb_logger = create_logger(opt, True)
    
    opt_teacher = opt.copy()
    opt_student = opt.copy()
    opt_teacher['path']['pretrain_network_g'] = opt.get('path_teacher', None)
    opt_student['path']['pretrain_network_g'] = opt.get('path_student', None)
    
    teacher_model = build_model(opt_teacher)
    student_model = build_model(opt_student)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher_model.to(device)
    student_model.to(device)
    
    opt_teacher['dataset']['train']['name'] = 'DatasetA'
    opt_student['dataset']['train']['name'] = 'DatasetB'
    train_loader_a = build_dataloader(opt_teacher['dataset']['train'], opt_teacher)
    train_loader_b = build_dataloader(opt_student['dataset']['train'], opt_student)
    
    iter_a = iter(train_loader_a)
    iter_b = iter(train_loader_b)
    
    distillation_loss_fn = nn.L1Loss()
    distillation_weight = opt['train'].get('distillation_weight', 0.1)
    
    total_iter = opt['train']['total_iter']
    for current_iter in range(total_iter):
        try:
            data_a = next(iter_a)
        except StopIteration:
            iter_a = iter(train_loader_a)
            data_a = next(iter_a)
        
        try:
            data_b = next(iter_b)
        except StopIteration:
            iter_b = iter(train_loader_b)
            data_b = next(iter_b)
        
        teacher_model.feed_data(data_a)
        teacher_model.optimize_parameters(current_iter)
        teacher_loss = teacher_model.log_dict
        teacher_output = teacher_model.output.detach()
        
        student_model.feed_data(data_b)
        student_model.optimize_parameters(current_iter)
        student_loss = student_model.log_dict
        student_output = student_model.output
        
        if opt['train'].get('distillation_loss', False):
            distillation_loss = distillation_loss_fn(student_output, teacher_output) * distillation_weight
            student_model.optimizer_g.zero_grad()
            distillation_loss.backward()
            student_model.optimizer_g.step()
        
        if opt['train'].get('siamese_loss', False):
            teacher_features = teacher_model.net_g.module.get_intermediate_features(data_a['lq'])
            student_features = student_model.net_g.module.get_intermediate_features(data_b['lq'])
            siamese_loss = nn.MSELoss()(teacher_features[-1], student_features[-1])
            student_model.optimizer_g.zero_grad()
            siamese_loss.backward()
            student_model.optimizer_g.step()
        
        if current_iter % opt['logger']['print_freq'] == 0:
            log_message = f"Iter {current_iter}: Teacher Loss: {teacher_loss}, Student Loss: {student_loss}"
            if opt['train'].get('distillation_loss', False):
                log_message += f", Distillation Loss: {distillation_loss.item():.4f}"
            if opt['train'].get('siamese_loss', False):
                log_message += f", Siamese Loss: {siamese_loss.item():.4f}"
            logger.info(log_message)
            if tb_logger:
                tb_logger.add_scalar('loss/teacher_total', teacher_loss['l_total'], current_iter)
                tb_logger.add_scalar('loss/student_total', student_loss['l_total'], current_iter)
                if opt['train'].get('distillation_loss', False):
                    tb_logger.add_scalar('loss/distillation', distillation_loss.item(), current_iter)
                if opt['train'].get('siamese_loss', False):
                    tb_logger.add_scalar('loss/siamese', siamese_loss.item(), current_iter)
        
        if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
            teacher_model.save(current_iter, 'teacher')
            student_model.save(current_iter, 'student')
    
    logger.info("Training completed.")
    if tb_logger:
        tb_logger.close()

if __name__ == '__main__':
    main()