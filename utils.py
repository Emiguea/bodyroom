import os
import glob
import random
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from config import config

class ReplayBuffer:
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.data = []
    
    def push_and_pop(self, data: torch.Tensor) -> torch.Tensor:
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)

class LambdaLR:
    def __init__(self, n_epochs: int, offset: int, decay_start_epoch: int):
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch
    
    def step(self, epoch: int) -> float:
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

class GANLoss(nn.Module):
    def __init__(self, use_lsgan: bool = True, target_real_label: float = 1.0, target_fake_label: float = 0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()
    
    def get_target_tensor(self, input: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)
    
    def __call__(self, input: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

def save_checkpoint(models: Dict[str, nn.Module], 
                    optimizers: Dict[str, torch.optim.Optimizer],
                    epoch: int, 
                    checkpoint_dir: str,
                    model_name: str = 'latest'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'models': {},
        'optimizers': {}
    }
    
    for name, model in models.items():
        checkpoint['models'][name] = model.state_dict()
    
    for name, optimizer in optimizers.items():
        checkpoint['optimizers'][name] = optimizer.state_dict()
    
    checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f'Checkpoint saved to {checkpoint_path}')

def load_checkpoint(models: Dict[str, nn.Module], 
                    optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
                    checkpoint_path: str = '',
                    device: torch.device = config.device) -> int:
    if not os.path.exists(checkpoint_path):
        print(f'Checkpoint not found at {checkpoint_path}')
        return 0
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    for name, model in models.items():
        if name in checkpoint['models']:
            model.load_state_dict(checkpoint['models'][name])
            print(f'Loaded model: {name}')
    
    if optimizers and 'optimizers' in checkpoint:
        for name, optimizer in optimizers.items():
            if name in checkpoint['optimizers']:
                optimizer.load_state_dict(checkpoint['optimizers'][name])
                print(f'Loaded optimizer: {name}')
    
    return checkpoint.get('epoch', 0)

def load_pretrained_model(model: nn.Module, 
                          model_path: str,
                          device: torch.device = config.device) -> nn.Module:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Pretrained model not found at {model_path}')
    
    state_dict = torch.load(model_path, map_location=device)
    
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    print(f'Pretrained model loaded from {model_path}')
    return model

def visualize_results(real_A: torch.Tensor, 
                      fake_B: torch.Tensor, 
                      rec_A: torch.Tensor,
                      real_B: Optional[torch.Tensor] = None,
                      fake_A: Optional[torch.Tensor] = None,
                      rec_B: Optional[torch.Tensor] = None,
                      save_path: Optional[str] = None,
                      epoch: Optional[int] = None,
                      step: Optional[int] = None):
    real_A = (real_A.cpu().squeeze(0) + 1) / 2.0
    fake_B = (fake_B.cpu().squeeze(0) + 1) / 2.0
    rec_A = (rec_A.cpu().squeeze(0) + 1) / 2.0
    
    num_cols = 3
    if real_B is not None:
        real_B = (real_B.cpu().squeeze(0) + 1) / 2.0
        fake_A = (fake_A.cpu().squeeze(0) + 1) / 2.0
        rec_B = (rec_B.cpu().squeeze(0) + 1) / 2.0
        num_cols = 6
    
    fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4))
    
    axes[0].imshow(real_A.permute(1, 2, 0))
    axes[0].set_title('Real A')
    axes[0].axis('off')
    
    axes[1].imshow(fake_B.permute(1, 2, 0))
    axes[1].set_title('Fake B')
    axes[1].axis('off')
    
    axes[2].imshow(rec_A.permute(1, 2, 0))
    axes[2].set_title('Recovered A')
    axes[2].axis('off')
    
    if real_B is not None:
        axes[3].imshow(real_B.permute(1, 2, 0))
        axes[3].set_title('Real B')
        axes[3].axis('off')
        
        axes[4].imshow(fake_A.permute(1, 2, 0))
        axes[4].set_title('Fake A')
        axes[4].axis('off')
        
        axes[5].imshow(rec_B.permute(1, 2, 0))
        axes[5].set_title('Recovered B')
        axes[5].axis('off')
    
    title = ''
    if epoch is not None:
        title += f'Epoch: {epoch} '
    if step is not None:
        title += f'Step: {step}'
    if title:
        plt.suptitle(title, y=0.95)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()

class Logger:
    def __init__(self, log_dir: str, use_tensorboard: bool = True):
        self.log_dir = log_dir
        self.use_tensorboard = use_tensorboard
        
        os.makedirs(log_dir, exist_ok=True)
        
        if use_tensorboard:
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
        
        self.loss_history = {}
    
    def log(self, 
            losses: Dict[str, float], 
            step: int,
            epoch: Optional[int] = None,
            images: Optional[Dict[str, torch.Tensor]] = None):
        
        print_str = ''
        if epoch is not None:
            print_str += f'[Epoch {epoch}] '
        print_str += f'[Step {step}] '
        
        for name, value in losses.items():
            print_str += f'{name}: {value:.4f} '
            
            if name not in self.loss_history:
                self.loss_history[name] = []
            self.loss_history[name].append(value)
            
            if self.writer:
                self.writer.add_scalar(f'Loss/{name}', value, step)
        
        print(print_str)
        
        if images and self.writer:
            for name, image in images.items():
                image = (image + 1) / 2.0
                self.writer.add_images(f'Images/{name}', image, step)
    
    def plot_losses(self, save_path: Optional[str] = None):
        plt.figure(figsize=(12, 8))
        for name, losses in self.loss_history.items():
            plt.plot(losses, label=name, alpha=0.7)
        
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
        else:
            plt.show()
    
    def close(self):
        if self.writer:
            self.writer.close()

def set_requires_grad(nets: list, requires_grad: bool):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
