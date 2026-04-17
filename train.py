import os
import argparse
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from config import config
from models import build_models, weights_init_normal
from datasets import get_data_loaders
from utils import (
    ReplayBuffer, LambdaLR as LRPolicy, GANLoss,
    save_checkpoint, load_checkpoint, set_requires_grad,
    Logger, visualize_results
)

class CycleGANTrainer:
    def __init__(self, 
                 data_dir: str,
                 checkpoint_dir: str,
                 results_dir: str,
                 device: torch.device = config.device):
        self.data_dir = data_dir
        self.checkpoint_dir = checkpoint_dir
        self.results_dir = results_dir
        self.device = device
        
        self.G_AB, self.G_BA, self.D_A, self.D_B = build_models()
        
        self.G_AB.apply(weights_init_normal)
        self.G_BA.apply(weights_init_normal)
        self.D_A.apply(weights_init_normal)
        self.D_B.apply(weights_init_normal)
        
        self.criterion_GAN = GANLoss().to(device)
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        
        self.optimizer_G = Adam(
            list(self.G_AB.parameters()) + list(self.G_BA.parameters()),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2)
        )
        self.optimizer_D_A = Adam(
            self.D_A.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2)
        )
        self.optimizer_D_B = Adam(
            self.D_B.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2)
        )
        
        decay_start_epoch = config.num_epochs // 2
        lr_lambda = LRPolicy(config.num_epochs, 0, decay_start_epoch).step
        self.scheduler_G = LambdaLR(self.optimizer_G, lr_lambda=lr_lambda)
        self.scheduler_D_A = LambdaLR(self.optimizer_D_A, lr_lambda=lr_lambda)
        self.scheduler_D_B = LambdaLR(self.optimizer_D_B, lr_lambda=lr_lambda)
        
        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()
        
        self.logger = Logger(os.path.join(results_dir, 'logs'))
        
        self.start_epoch = 0
        self.global_step = 0
    
    def load_checkpoint(self, checkpoint_path: str):
        models = {
            'G_AB': self.G_AB,
            'G_BA': self.G_BA,
            'D_A': self.D_A,
            'D_B': self.D_B
        }
        optimizers = {
            'optimizer_G': self.optimizer_G,
            'optimizer_D_A': self.optimizer_D_A,
            'optimizer_D_B': self.optimizer_D_B
        }
        
        self.start_epoch = load_checkpoint(models, optimizers, checkpoint_path, self.device)
        print(f'Resuming training from epoch {self.start_epoch}')
    
    def train_step(self, real_A: torch.Tensor, real_B: torch.Tensor) -> Dict[str, float]:
        real_A = real_A.to(self.device)
        real_B = real_B.to(self.device)
        
        set_requires_grad([self.D_A, self.D_B], False)
        self.optimizer_G.zero_grad()
        
        if config.use_identity_loss:
            loss_id_A = self.criterion_identity(self.G_BA(real_A), real_A) * config.lambda_cycle * config.lambda_identity
            loss_id_B = self.criterion_identity(self.G_AB(real_B), real_B) * config.lambda_cycle * config.lambda_identity
        else:
            loss_id_A = 0
            loss_id_B = 0
        
        fake_B = self.G_AB(real_A)
        pred_fake = self.D_B(fake_B)
        loss_GAN_AB = self.criterion_GAN(pred_fake, True)
        
        fake_A = self.G_BA(real_B)
        pred_fake = self.D_A(fake_A)
        loss_GAN_BA = self.criterion_GAN(pred_fake, True)
        
        rec_A = self.G_BA(fake_B)
        loss_cycle_A = self.criterion_cycle(rec_A, real_A) * config.lambda_cycle
        
        rec_B = self.G_AB(fake_A)
        loss_cycle_B = self.criterion_cycle(rec_B, real_B) * config.lambda_cycle
        
        loss_G = loss_GAN_AB + loss_GAN_BA + loss_cycle_A + loss_cycle_B + loss_id_A + loss_id_B
        loss_G.backward()
        self.optimizer_G.step()
        
        set_requires_grad([self.D_A, self.D_B], True)
        
        self.optimizer_D_B.zero_grad()
        pred_real = self.D_B(real_B)
        loss_D_real = self.criterion_GAN(pred_real, True)
        
        fake_B_ = self.fake_B_buffer.push_and_pop(fake_B)
        pred_fake = self.D_B(fake_B_.detach())
        loss_D_fake = self.criterion_GAN(pred_fake, False)
        
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward()
        self.optimizer_D_B.step()
        
        self.optimizer_D_A.zero_grad()
        pred_real = self.D_A(real_A)
        loss_D_real = self.criterion_GAN(pred_real, True)
        
        fake_A_ = self.fake_A_buffer.push_and_pop(fake_A)
        pred_fake = self.D_A(fake_A_.detach())
        loss_D_fake = self.criterion_GAN(pred_fake, False)
        
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A.backward()
        self.optimizer_D_A.step()
        
        losses = {
            'G_GAN': (loss_GAN_AB + loss_GAN_BA).item(),
            'G_cycle': (loss_cycle_A + loss_cycle_B).item(),
            'G_identity': (loss_id_A + loss_id_B).item() if config.use_identity_loss else 0,
            'G_total': loss_G.item(),
            'D_A': loss_D_A.item(),
            'D_B': loss_D_B.item()
        }
        
        images = {
            'real_A': real_A,
            'fake_B': fake_B,
            'rec_A': rec_A,
            'real_B': real_B,
            'fake_A': fake_A,
            'rec_B': rec_B
        }
        
        return losses, images
    
    def train(self, train_loader: DataLoader, test_loader: DataLoader):
        print(f'Starting training from epoch {self.start_epoch + 1} to {config.num_epochs}')
        print(f'Training on {self.device}')
        print(f'Training samples: {len(train_loader.dataset)}')
        
        for epoch in range(self.start_epoch, config.num_epochs):
            self.G_AB.train()
            self.G_BA.train()
            self.D_A.train()
            self.D_B.train()
            
            for i, batch in enumerate(train_loader):
                self.global_step += 1
                
                real_A = batch['A']
                real_B = batch['B']
                
                losses, images = self.train_step(real_A, real_B)
                
                if self.global_step % config.print_freq == 0:
                    self.logger.log(losses, self.global_step, epoch + 1)
                
                if self.global_step % config.sample_freq == 0:
                    self.save_sample_images(images, epoch + 1, self.global_step)
                
                if self.global_step % config.save_freq == 0:
                    self.save_models(epoch + 1)
            
            self.scheduler_G.step()
            self.scheduler_D_A.step()
            self.scheduler_D_B.step()
            
            self.evaluate(test_loader, epoch + 1)
            self.save_models(epoch + 1, 'latest')
        
        self.logger.plot_losses(os.path.join(self.results_dir, 'loss_plot.png'))
        self.logger.close()
    
    def evaluate(self, test_loader: DataLoader, epoch: int):
        self.G_AB.eval()
        self.G_BA.eval()
        
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                real_A = batch['A'].to(self.device)
                real_B = batch['B'].to(self.device)
                
                fake_B = self.G_AB(real_A)
                rec_A = self.G_BA(fake_B)
                fake_A = self.G_BA(real_B)
                rec_B = self.G_AB(fake_A)
                
                save_path = os.path.join(self.results_dir, f'evaluation_epoch_{epoch}_sample_{i}.png')
                visualize_results(
                    real_A, fake_B, rec_A,
                    real_B, fake_A, rec_B,
                    save_path=save_path,
                    epoch=epoch
                )
                
                if i >= 5:
                    break
        
        print(f'Evaluation completed for epoch {epoch}')
    
    def save_sample_images(self, images: dict, epoch: int, step: int):
        real_A = images['real_A']
        fake_B = images['fake_B']
        rec_A = images['rec_A']
        real_B = images['real_B']
        fake_A = images['fake_A']
        rec_B = images['rec_B']
        
        save_path = os.path.join(self.results_dir, f'train_epoch_{epoch}_step_{step}.png')
        visualize_results(
            real_A, fake_B, rec_A,
            real_B, fake_A, rec_B,
            save_path=save_path,
            epoch=epoch,
            step=step
        )
    
    def save_models(self, epoch: int, name: str = None):
        models = {
            'G_AB': self.G_AB,
            'G_BA': self.G_BA,
            'D_A': self.D_A,
            'D_B': self.D_B
        }
        optimizers = {
            'optimizer_G': self.optimizer_G,
            'optimizer_D_A': self.optimizer_D_A,
            'optimizer_D_B': self.optimizer_D_B
        }
        
        model_name = name if name else f'epoch_{epoch}'
        save_checkpoint(models, optimizers, epoch, self.checkpoint_dir, model_name)

def main():
    parser = argparse.ArgumentParser(description='Train CycleGAN')
    parser.add_argument('--data_dir', type=str, default=config.data_dir, help='Data directory')
    parser.add_argument('--checkpoint_dir', type=str, default=config.checkpoint_dir, help='Checkpoint directory')
    parser.add_argument('--results_dir', type=str, default=config.results_dir, help='Results directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--batch_size', type=int, default=config.batch_size, help='Batch size')
    parser.add_argument('--epochs', type=int, default=config.num_epochs, help='Number of epochs')
    
    args = parser.parse_args()
    
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    print('Loading datasets...')
    train_loader, test_loader = get_data_loaders(args.data_dir, args.batch_size)
    
    print('Building trainer...')
    trainer = CycleGANTrainer(
        args.data_dir,
        args.checkpoint_dir,
        args.results_dir
    )
    
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    trainer.train(train_loader, test_loader)

if __name__ == '__main__':
    main()
