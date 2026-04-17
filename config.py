import torch
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class Config:
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    input_channels: int = 3
    output_channels: int = 3
    
    image_size: Tuple[int, int] = (256, 256)
    
    batch_size: int = 1
    num_epochs: int = 200
    learning_rate: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    
    lambda_cycle: float = 10.0
    lambda_identity: float = 0.5
    
    gen_features: int = 64
    disc_features: int = 64
    
    data_dir: str = './data'
    train_dir: str = './data/train'
    test_dir: str = './data/test'
    checkpoint_dir: str = './checkpoints'
    results_dir: str = './results'
    
    pretrained_model_name: str = 'summer2winter_yosemite'
    pretrained_models_url: str = 'https://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/'
    
    print_freq: int = 100
    save_freq: int = 1000
    sample_freq: int = 500
    
    use_identity_loss: bool = True
    
    model_names: Tuple[str, ...] = (
        'summer2winter_yosemite',
        'apple2orange',
        'horse2zebra',
        'monet2photo',
        'cezanne2photo',
        'ukiyoe2photo',
        'vangogh2photo'
    )

config = Config()
