import yaml
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import torch.optim as optim
from datasets.siamese_paired_dataset import SiamesePairedDataset
from models.siamese_model import SiameseESRGAN
from trainers.siamese_trainer import SiameseTrainer

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train_siamese.yml')
    args = parser.parse_args()
    config = load_config(args.config)

    # Khởi tạo dataset
    train_dataset = SiamesePairedDataset({
        'dataroot_gt': config['datasets']['train']['dataroot_gt'],
        'dataroot_lq': config['datasets']['train']['dataroot_low'],
        'dataroot_very_low': config['datasets']['train']['dataroot_very_low'],
        'gt_size': config['datasets']['train']['gt_size']
    })
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['datasets']['train']['batch_size_per_gpu'],
        shuffle=True,
        num_workers=config['datasets']['train']['num_worker_per_gpu']
    )

    # Khởi tạo model và trainer
    model = SiameseESRGAN(config)
    trainer = SiameseTrainer(
        model=model,
        opt_g_teacher=Adam(
            model.teacher.parameters(),
            lr=config['train']['optim_g']['lr'],
            betas=config['train']['optim_g']['betas']
        ),
        opt_g_student=Adam(
            model.student.parameters(),
            lr=config['train']['optim_g']['lr'],
            betas=config['train']['optim_g']['betas']
        ),
        opt_d=Adam(
            model.discriminator.parameters(),
            lr=config['train']['optim_d']['lr'],
            betas=config['train']['optim_d']['betas']
        ),
        config=config
    )

    # Training
    for epoch in range(config['train']['total_iter'] // len(train_loader)):
        trainer.train_epoch(train_loader, epoch)

if __name__ == '__main__':
    main()