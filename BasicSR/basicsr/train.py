import datetime
import logging
import math
import time
import torch
import torch.nn as nn
from os import path as osp

from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                           init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)
from basicsr.utils.options import copy_opt_file, dict2str, parse_options

def init_tb_loggers(opt):
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join(opt['root_path'], 'tb_logger', opt['name']))
    return tb_logger

def create_train_val_dataloader(opt, logger):
    train_loaders, val_loaders = {}, []
    for phase, dataset_opt in opt['datasets'].items():
        if phase in ['teacher', 'student']:
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = build_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])
            train_loaders[phase] = train_loader
            train_loaders[f'{phase}_sampler'] = train_sampler

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / num_iter_per_epoch)
            logger.info(f'{phase.capitalize()} Training statistics:'
                        f'\n\tNumber of train images: {len(train_set)}'
                        f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                        f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                        f'\n\tWorld size (gpu number): {opt["world_size"]}'
                        f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                        f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
        elif phase.split('_')[0] == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
            val_loaders.append(val_loader)
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loaders, val_loaders, total_epochs, total_iters

def load_resume_state(opt):
    resume_state_path = None
    if opt['auto_resume']:
        state_path = osp.join('experiments', opt['name'], 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt['path']['resume_state'] = resume_state_path
    else:
        if opt['path'].get('resume_state'):
            resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
        check_resume(opt, resume_state['iter'])
    return resume_state

def train_pipeline(root_path):
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path

    torch.backends.cudnn.benchmark = True

    resume_state = load_resume_state(opt)
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))

    copy_opt_file(args.opt, opt['path']['experiments_root'])

    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    tb_logger = init_tb_loggers(opt)

    result = create_train_val_dataloader(opt, logger)
    train_loaders, val_loaders, total_epochs, total_iters = result

    # Create Teacher and Student models
    teacher_model = build_model(opt)
    student_model = build_model(opt)  # Same architecture
    teacher_model.net_g.cuda()
    student_model.net_g.cuda()

    if resume_state:
        teacher_model.resume_training(resume_state)
        student_model.resume_training(resume_state)
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        start_epoch = 0
        current_iter = 0

    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # Initialize Siamese loss
    siamese_criterion = nn.MSELoss().cuda()

    # Optimizer for both models
    optim_params = list(teacher_model.net_g.parameters()) + list(student_model.net_g.parameters())
    optimizer_g = torch.optim.Adam(optim_params, lr=opt['train']['optim_g']['lr'])

    # Dataloader prefetcher
    prefetch_mode = opt['datasets']['teacher'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        teacher_prefetcher = CPUPrefetcher(train_loaders['teacher'])
        student_prefetcher = CPUPrefetcher(train_loaders['student'])
    elif prefetch_mode == 'cuda':
        teacher_prefetcher = CUDAPrefetcher(train_loaders['teacher'], opt)
        student_prefetcher = CUDAPrefetcher(train_loaders['student'], opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['teacher'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f"Wrong prefetch_mode {prefetch_mode}. Supported ones are: None, 'cuda', 'cpu'.")

    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    for epoch in range(start_epoch, total_epochs + 1):
        train_loaders['teacher_sampler'].set_epoch(epoch)
        train_loaders['student_sampler'].set_epoch(epoch)
        teacher_prefetcher.reset()
        student_prefetcher.reset()
        teacher_data = teacher_prefetcher.next()
        student_data = student_prefetcher.next()

        while teacher_data is not None and student_data is not None:
            data_timer.record()

            current_iter += 1
            if current_iter > total_iters:
                break

            # Update learning rate
            teacher_model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            student_model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))

            # Feed data
            teacher_model.feed_data(teacher_data)
            student_model.feed_data(student_data)

            # Optimize parameters
            teacher_model.optimize_parameters(current_iter)
            student_model.optimize_parameters(current_iter)

            # Siamese loss
            teacher_output = teacher_model.net_g(teacher_data['lq'].cuda())
            student_output = student_model.net_g(student_data['lq'].cuda())
            siamese_loss = siamese_criterion(student_output, teacher_output.detach())

            # Combine losses
            teacher_loss = teacher_model.get_current_log().get('l_g_total', 0)
            student_loss = student_model.get_current_log().get('l_g_total', 0)
            total_loss = teacher_loss + student_loss + opt['train']['siamese_opt']['loss_weight'] * siamese_loss

            # Backward and optimize
            optimizer_g.zero_grad()
            total_loss.backward()
            optimizer_g.step()

            iter_timer.record()
            if current_iter == 1:
                msg_logger.reset_start_time()

            # Log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': teacher_model.get_current_learning_rate()})
                log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                log_vars.update({'teacher_loss': teacher_loss, 'student_loss': student_loss, 'siamese_loss': siamese_loss.item()})
                msg_logger(log_vars)

            # Save models
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                teacher_model.save(epoch, current_iter, save_path=osp.join(opt['path']['models'], 'teacher_net_g'))
                student_model.save(epoch, current_iter, save_path=osp.join(opt['path']['models'], 'student_net_g'))

            # Validation
            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                for val_loader in val_loaders:
                    teacher_model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
                    student_model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])

            data_timer.start()
            iter_timer.start()
            teacher_data = teacher_prefetcher.next()
            student_data = student_prefetcher.next()

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    teacher_model.save(epoch=-1, current_iter=-1, save_path=osp.join(opt['path']['models'], 'teacher_net_g'))
    student_model.save(epoch=-1, current_iter=-1, save_path=osp.join(opt['path']['models'], 'student_net_g'))
    if opt.get('val') is not None:
        for val_loader in val_loaders:
            teacher_model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
            student_model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)