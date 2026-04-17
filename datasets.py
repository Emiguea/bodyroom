import os
import glob
import random
from PIL import Image
from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import config

class ImageDataset(Dataset):
    def __init__(self, 
                 root: str, 
                 transforms_: Optional[transforms.Compose] = None, 
                 unaligned: bool = True,
                 mode: str = 'train'):
        self.root = root
        self.unaligned = unaligned
        self.transform = transforms_ if transforms_ else self._default_transforms(mode)
        
        self.files_A = sorted(glob.glob(os.path.join(root, mode, 'A') + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, mode, 'B') + '/*.*'))
        
        self.len_A = len(self.files_A)
        self.len_B = len(self.files_B)
    
    def _default_transforms(self, mode: str) -> transforms.Compose:
        if mode == 'train':
            return transforms.Compose([
                transforms.Resize(int(config.image_size[0] * 1.12), Image.BICUBIC),
                transforms.RandomCrop(config.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            return transforms.Compose([
                transforms.Resize(config.image_size, Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    
    def __getitem__(self, index: int) -> dict:
        item_A = self.transform(Image.open(self.files_A[index % self.len_A]).convert('RGB'))
        
        if self.unaligned:
            item_B = self.transform(Image.open(self.files_B[random.randint(0, self.len_B - 1)]).convert('RGB'))
        else:
            item_B = self.transform(Image.open(self.files_B[index % self.len_B]).convert('RGB'))
        
        return {'A': item_A, 'B': item_B}
    
    def __len__(self) -> int:
        return max(self.len_A, self.len_B)

class SingleImageDataset(Dataset):
    def __init__(self, 
                 image_path: str, 
                 transforms_: Optional[transforms.Compose] = None):
        self.image_path = image_path
        self.transform = transforms_ if transforms_ else self._default_transforms()
    
    def _default_transforms(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize(config.image_size, Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.image_path).convert('RGB')
        return self.transform(image)
    
    def __len__(self) -> int:
        return 1

def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = (image + 1) / 2.0
    image = transforms.ToPILImage()(image)
    return image

def image_to_tensor(image: Image.Image, 
                    image_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    if image_size is None:
        image_size = config.image_size
    
    transform = transforms.Compose([
        transforms.Resize(image_size, Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    return transform(image).unsqueeze(0)

def get_data_loaders(data_dir: str, 
                     batch_size: Optional[int] = None,
                     image_size: Optional[Tuple[int, int]] = None) -> Tuple[DataLoader, DataLoader]:
    if batch_size is None:
        batch_size = config.batch_size
    
    train_transform = transforms.Compose([
        transforms.Resize(int(config.image_size[0] * 1.12) if image_size is None else int(image_size[0] * 1.12), Image.BICUBIC),
        transforms.RandomCrop(config.image_size if image_size is None else image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(config.image_size if image_size is None else image_size, Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = ImageDataset(data_dir, transforms_=train_transform, mode='train')
    test_dataset = ImageDataset(data_dir, transforms_=test_transform, mode='test')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, test_loader

def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor + 1) / 2.0

def get_inference_transform(image_size: Optional[Tuple[int, int]] = None) -> transforms.Compose:
    if image_size is None:
        image_size = config.image_size
    
    return transforms.Compose([
        transforms.Resize(image_size, Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
