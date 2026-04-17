import sys
import os

print("=" * 60)
print("CycleGAN 项目导入测试")
print("=" * 60)
print(f"Python 版本: {sys.version}")
print(f"工作目录: {os.getcwd()}")
print("=" * 60)

errors = []

print("\n[1/6] 测试 config.py...")
try:
    from config import Config, config
    print(f"    OK - 设备: {Config.device.default}")
    print(f"    OK - 图像尺寸: {config.image_size}")
    print(f"    OK - 模型列表: {config.model_names}")
except Exception as e:
    print(f"    错误: {e}")
    errors.append(('config.py', str(e)))

print("\n[2/6] 测试 models.py...")
try:
    from models import (
        ResidualBlock, UNetDown, UNetUp, 
        GeneratorUNet, GeneratorResNet, Discriminator,
        build_models, weights_init_normal
    )
    print("    OK - ResidualBlock 导入成功")
    print("    OK - UNetDown/UNetUp 导入成功")
    print("    OK - GeneratorUNet 导入成功")
    print("    OK - GeneratorResNet 导入成功")
    print("    OK - Discriminator 导入成功")
    print("    OK - build_models 函数导入成功")
except Exception as e:
    print(f"    错误: {e}")
    errors.append(('models.py', str(e)))

print("\n[3/6] 测试 datasets.py...")
try:
    from datasets import (
        ImageDataset, SingleImageDataset,
        tensor_to_image, image_to_tensor,
        get_data_loaders, denormalize, get_inference_transform
    )
    print("    OK - ImageDataset 导入成功")
    print("    OK - SingleImageDataset 导入成功")
    print("    OK - 转换函数导入成功")
    print("    OK - get_data_loaders 函数导入成功")
except Exception as e:
    print(f"    错误: {e}")
    errors.append(('datasets.py', str(e)))

print("\n[4/6] 测试 utils.py...")
try:
    from utils import (
        ReplayBuffer, LambdaLR, GANLoss,
        save_checkpoint, load_checkpoint, load_pretrained_model,
        visualize_results, Logger, set_requires_grad
    )
    print("    OK - ReplayBuffer 导入成功")
    print("    OK - LambdaLR 导入成功")
    print("    OK - GANLoss 导入成功")
    print("    OK - 模型保存/加载函数导入成功")
    print("    OK - Logger 类导入成功")
except Exception as e:
    print(f"    错误: {e}")
    errors.append(('utils.py', str(e)))

print("\n[5/6] 测试 infer.py...")
try:
    from infer import (
        CycleGANInferencer, PRETRAINED_MODELS_INFO,
        list_available_models
    )
    print(f"    OK - 预训练模型数量: {len(PRETRAINED_MODELS_INFO)}")
    print(f"    OK - 模型列表: {list(PRETRAINED_MODELS_INFO.keys())}")
except Exception as e:
    print(f"    错误: {e}")
    errors.append(('infer.py', str(e)))

print("\n[6/6] 测试 app.py...")
try:
    from app import (
        GradioApp, HTML_HEADER, HTML_FOOTER, 
        MODEL_CARD_TEMPLATE
    )
    print("    OK - GradioApp 类导入成功")
    print("    OK - HTML模板导入成功")
except Exception as e:
    print(f"    错误: {e}")
    errors.append(('app.py', str(e)))

print("\n" + "=" * 60)
if errors:
    print(f"测试完成，发现 {len(errors)} 个错误:")
    for module, error in errors:
        print(f"  - {module}: {error}")
else:
    print("✅ 所有模块导入成功！代码结构正确。")
    print("\n项目文件结构:")
    for f in ['config.py', 'models.py', 'datasets.py', 'utils.py', 'train.py', 'infer.py', 'app.py', 'requirements.txt']:
        if os.path.exists(f):
            size = os.path.getsize(f)
            print(f"  ✓ {f} ({size} 字节)")
print("=" * 60)

print("\n💡 使用说明:")
print("  1. 安装依赖: pip install -r requirements.txt")
print("  2. 启动Web界面: python app.py")
print("  3. 命令行推理: python infer.py --help")
print("  4. 训练模型: python train.py --help")
