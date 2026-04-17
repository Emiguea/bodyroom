import torch
import torch.nn as nn
from config import config

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int):
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3),
            nn.InstanceNorm2d(in_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)

class UNetDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalize: bool = True):
        super(UNetDown, self).__init__()
        
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: bool = False):
        super(UNetUp, self).__init__()
        
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, skip_input: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return torch.cat((x, skip_input), 1)

class GeneratorUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, features: int = 64):
        super(GeneratorUNet, self).__init__()
        
        self.down1 = UNetDown(in_channels, features, normalize=False)
        self.down2 = UNetDown(features, features * 2)
        self.down3 = UNetDown(features * 2, features * 4)
        self.down4 = UNetDown(features * 4, features * 8)
        self.down5 = UNetDown(features * 8, features * 8)
        self.down6 = UNetDown(features * 8, features * 8)
        self.down7 = UNetDown(features * 8, features * 8)
        self.down8 = UNetDown(features * 8, features * 8, normalize=False)
        
        self.up1 = UNetUp(features * 8, features * 8, dropout=True)
        self.up2 = UNetUp(features * 16, features * 8, dropout=True)
        self.up3 = UNetUp(features * 16, features * 8, dropout=True)
        self.up4 = UNetUp(features * 16, features * 8)
        self.up5 = UNetUp(features * 16, features * 4)
        self.up6 = UNetUp(features * 8, features * 2)
        self.up7 = UNetUp(features * 4, features)
        
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(features * 2, out_channels, kernel_size=4, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        
        return self.final(u7)

class GeneratorResNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, features: int = 64, num_residual_blocks: int = 9):
        super(GeneratorResNet, self).__init__()
        
        out_features = features
        
        model = [
            nn.ReflectionPad2d(in_channels),
            nn.Conv2d(in_channels, out_features, kernel_size=7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True)
        ]
        
        for _ in range(2):
            in_features = out_features
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
        
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]
        
        for _ in range(2):
            in_features = out_features
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
        
        model += [
            nn.ReflectionPad2d(in_channels),
            nn.Conv2d(out_features, out_channels, kernel_size=7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 3, features: int = 64):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(in_channels, features, normalize=False),
            *discriminator_block(features, features * 2),
            *discriminator_block(features * 2, features * 4),
            *discriminator_block(features * 4, features * 8),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(features * 8, 1, kernel_size=4, padding=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def build_models():
    G_AB = GeneratorResNet(
        config.input_channels, 
        config.output_channels, 
        config.gen_features
    ).to(config.device)
    
    G_BA = GeneratorResNet(
        config.input_channels, 
        config.output_channels, 
        config.gen_features
    ).to(config.device)
    
    D_A = Discriminator(
        config.input_channels, 
        config.disc_features
    ).to(config.device)
    
    D_B = Discriminator(
        config.input_channels, 
        config.disc_features
    ).to(config.device)
    
    return G_AB, G_BA, D_A, D_B

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
