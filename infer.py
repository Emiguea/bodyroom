import os
import argparse
from typing import Optional, Tuple, List
import urllib.request
import zipfile

import torch
from PIL import Image
import matplotlib.pyplot as plt

from config import config
from models import GeneratorResNet
from datasets import image_to_tensor, tensor_to_image, get_inference_transform
from utils import load_pretrained_model

PRETRAINED_MODELS_INFO = {
    'summer2winter_yosemite': {
        'url': 'https://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/summer2winter_yosemite.pth',
        'description': 'Summer <-> Winter (Yosemite)',
        'direction': ['summer2winter', 'winter2summer']
    },
    'apple2orange': {
        'url': 'https://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/apple2orange.pth',
        'description': 'Apple <-> Orange',
        'direction': ['apple2orange', 'orange2apple']
    },
    'horse2zebra': {
        'url': 'https://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/horse2zebra.pth',
        'description': 'Horse <-> Zebra',
        'direction': ['horse2zebra', 'zebra2horse']
    },
    'monet2photo': {
        'url': 'https://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/monet2photo.pth',
        'description': 'Monet <-> Photo',
        'direction': ['monet2photo', 'photo2monet']
    },
    'cezanne2photo': {
        'url': 'https://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/cezanne2photo.pth',
        'description': 'Cezanne <-> Photo',
        'direction': ['cezanne2photo', 'photo2cezanne']
    },
    'ukiyoe2photo': {
        'url': 'https://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/ukiyoe2photo.pth',
        'description': 'Ukiyo-e <-> Photo',
        'direction': ['ukiyoe2photo', 'photo2ukiyoe']
    },
    'vangogh2photo': {
        'url': 'https://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/vangogh2photo.pth',
        'description': 'Van Gogh <-> Photo',
        'direction': ['vangogh2photo', 'photo2vangogh']
    }
}

class CycleGANInferencer:
    def __init__(self, 
                 model_name: str = 'summer2winter_yosemite',
                 checkpoint_dir: str = './checkpoints/pretrained',
                 device: torch.device = config.device):
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        
        if model_name not in PRETRAINED_MODELS_INFO:
            raise ValueError(f'Unknown model: {model_name}. Available models: {list(PRETRAINED_MODELS_INFO.keys())}')
        
        self.model_info = PRETRAINED_MODELS_INFO[model_name]
        
        self.G_AB = GeneratorResNet(
            config.input_channels, 
            config.output_channels, 
            config.gen_features
        ).to(device)
        
        self.G_BA = GeneratorResNet(
            config.input_channels, 
            config.output_channels, 
            config.gen_features
        ).to(device)
        
        self._load_pretrained()
    
    def _download_model(self, url: str, save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        print(f'Downloading pretrained model from {url}...')
        
        def progress_callback(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size)
            print(f'\rDownloading... {percent:.1f}%', end='')
        
        urllib.request.urlretrieve(url, save_path, reporthook=progress_callback)
        print(f'\nModel downloaded to {save_path}')
    
    def _load_pretrained(self):
        model_path = os.path.join(self.checkpoint_dir, f'{self.model_name}.pth')
        
        if not os.path.exists(model_path):
            print(f'Pretrained model not found at {model_path}')
            self._download_model(self.model_info['url'], model_path)
        
        print(f'Loading pretrained model: {self.model_name}')
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        def load_generator(model, state_dict_prefix):
            new_state_dict = {}
            for k, v in checkpoint.items():
                if k.startswith(state_dict_prefix):
                    new_key = k.replace(state_dict_prefix, '')
                    new_state_dict[new_key] = v
            
            if new_state_dict:
                model.load_state_dict(new_state_dict)
                return True
            return False
        
        loaded_AB = load_generator(self.G_AB, 'G_AB.') or load_generator(self.G_AB, 'netG_A.') or load_generator(self.G_AB, '')
        loaded_BA = load_generator(self.G_BA, 'G_BA.') or load_generator(self.G_BA, 'netG_B.')
        
        if not loaded_AB:
            print('Warning: Could not load G_AB weights, trying alternative structure...')
            self.G_AB.load_state_dict(checkpoint, strict=False)
        
        if not loaded_BA:
            print('Warning: Could not load G_BA weights, trying alternative structure...')
            if loaded_AB:
                self.G_BA.load_state_dict(self.G_AB.state_dict())
        
        self.G_AB.eval()
        self.G_BA.eval()
        
        print(f'Model loaded successfully: {self.model_info["description"]}')
    
    def transform(self, 
                  image: Image.Image, 
                  direction: str = 'AtoB',
                  return_tensor: bool = False) -> Image.Image:
        direction = direction.upper()
        if direction == 'ATOB':
            model = self.G_AB
        elif direction == 'BTOA':
            model = self.G_BA
        else:
            raise ValueError(f'Unknown direction: {direction}. Use "AtoB" or "BtoA"')
        
        original_size = image.size
        tensor = image_to_tensor(image).to(self.device)
        
        with torch.no_grad():
            transformed_tensor = model(tensor)
        
        if return_tensor:
            return transformed_tensor
        
        transformed_image = tensor_to_image(transformed_tensor)
        
        if transformed_image.size != original_size:
            transformed_image = transformed_image.resize(original_size, Image.BICUBIC)
        
        return transformed_image
    
    def transform_with_cycle(self, 
                              image: Image.Image, 
                              direction: str = 'AtoB') -> Tuple[Image.Image, Image.Image]:
        direction = direction.upper()
        original_size = image.size
        tensor = image_to_tensor(image).to(self.device)
        
        with torch.no_grad():
            if direction == 'ATOB':
                fake_B = self.G_AB(tensor)
                rec_A = self.G_BA(fake_B)
                transformed = fake_B
                recovered = rec_A
            else:
                fake_A = self.G_BA(tensor)
                rec_B = self.G_AB(fake_A)
                transformed = fake_A
                recovered = rec_B
        
        transformed_image = tensor_to_image(transformed)
        recovered_image = tensor_to_image(recovered)
        
        if transformed_image.size != original_size:
            transformed_image = transformed_image.resize(original_size, Image.BICUBIC)
        if recovered_image.size != original_size:
            recovered_image = recovered_image.resize(original_size, Image.BICUBIC)
        
        return transformed_image, recovered_image
    
    def transform_file(self, 
                       input_path: str, 
                       output_path: Optional[str] = None,
                       direction: str = 'AtoB') -> Image.Image:
        image = Image.open(input_path).convert('RGB')
        transformed_image = self.transform(image, direction)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            transformed_image.save(output_path)
            print(f'Saved transformed image to {output_path}')
        
        return transformed_image
    
    def visualize_transformation(self, 
                                  image: Image.Image, 
                                  direction: str = 'AtoB',
                                  save_path: Optional[str] = None):
        transformed_image, recovered_image = self.transform_with_cycle(image, direction)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image)
        axes[0].set_title(f'Original ({direction[:-3]})')
        axes[0].axis('off')
        
        axes[1].imshow(transformed_image)
        axes[1].set_title(f'Transformed ({direction[-3:]})')
        axes[1].axis('off')
        
        axes[2].imshow(recovered_image)
        axes[2].set_title('Recovered (Cycle)')
        axes[2].axis('off')
        
        plt.suptitle(f'Model: {self.model_info["description"]}', y=1.02)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
        else:
            plt.show()
        
        return transformed_image, recovered_image
    
    def batch_transform(self, 
                        input_dir: str, 
                        output_dir: str,
                        direction: str = 'AtoB',
                        image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')):
        os.makedirs(output_dir, exist_ok=True)
        
        image_files = []
        for ext in image_extensions:
            image_files.extend([f for f in os.listdir(input_dir) if f.lower().endswith(ext)])
        
        print(f'Found {len(image_files)} images for transformation')
        
        for i, filename in enumerate(image_files):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            try:
                self.transform_file(input_path, output_path, direction)
                print(f'[{i+1}/{len(image_files)}] Processed: {filename}')
            except Exception as e:
                print(f'Error processing {filename}: {e}')
        
        print(f'Batch transformation completed. Results saved to {output_dir}')

def list_available_models():
    print('Available pretrained models:')
    for name, info in PRETRAINED_MODELS_INFO.items():
        print(f'\n  - {name}:')
        print(f'    Description: {info["description"]}')
        print(f'    Directions: {", ".join(info["direction"])}')

def main():
    parser = argparse.ArgumentParser(description='CycleGAN Inference')
    parser.add_argument('--model', type=str, default='summer2winter_yosemite', 
                        help='Pretrained model name')
    parser.add_argument('--list_models', action='store_true', 
                        help='List all available pretrained models')
    parser.add_argument('--input', type=str, default=None, 
                        help='Input image path')
    parser.add_argument('--output', type=str, default=None, 
                        help='Output image path')
    parser.add_argument('--direction', type=str, default='AtoB', 
                        choices=['AtoB', 'BtoA'],
                        help='Transformation direction: AtoB or BtoA')
    parser.add_argument('--input_dir', type=str, default=None, 
                        help='Input directory for batch transformation')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='Output directory for batch transformation')
    parser.add_argument('--visualize', action='store_true', 
                        help='Visualize transformation with cycle consistency')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/pretrained', 
                        help='Checkpoint directory')
    
    args = parser.parse_args()
    
    if args.list_models:
        list_available_models()
        return
    
    inferencer = CycleGANInferencer(
        model_name=args.model,
        checkpoint_dir=args.checkpoint_dir
    )
    
    if args.input_dir and args.output_dir:
        inferencer.batch_transform(args.input_dir, args.output_dir, args.direction)
    elif args.input:
        if args.visualize:
            image = Image.open(args.input).convert('RGB')
            inferencer.visualize_transformation(image, args.direction, args.output)
        else:
            inferencer.transform_file(args.input, args.output, args.direction)
    else:
        print('No input specified. Use --input or --input_dir')
        print('Use --list_models to see available models')

if __name__ == '__main__':
    main()
