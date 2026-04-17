import ast
import os
import sys

print("=" * 70)
print("CycleGAN 项目语法检查")
print("=" * 70)

files_to_check = [
    'config.py',
    'models.py', 
    'datasets.py',
    'utils.py',
    'train.py',
    'infer.py',
    'app.py'
]

all_passed = True
errors = []

for filename in files_to_check:
    if not os.path.exists(filename):
        print(f"\n[?] {filename}: 文件不存在，跳过")
        continue
    
    print(f"\n[检查] {filename}...")
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        ast.parse(content, filename=filename)
        file_size = os.path.getsize(filename)
        lines = len(content.splitlines())
        print(f"    [OK] 语法正确 - {lines} 行, {file_size} 字节")
        
    except SyntaxError as e:
        all_passed = False
        error_msg = f"语法错误: 第 {e.lineno} 行: {e.msg}"
        print(f"    [ERROR] {error_msg}")
        errors.append((filename, error_msg))
        
    except UnicodeDecodeError as e:
        all_passed = False
        error_msg = f"编码错误: {e}"
        print(f"    [ERROR] {error_msg}")
        errors.append((filename, error_msg))
        
    except Exception as e:
        all_passed = False
        error_msg = f"未知错误: {e}"
        print(f"    [ERROR] {error_msg}")
        errors.append((filename, error_msg))

print("\n" + "=" * 70)
if all_passed:
    print("✅ 所有Python文件语法检查通过！")
else:
    print(f"❌ 发现 {len(errors)} 个错误:")
    for filename, error in errors:
        print(f"  - {filename}: {error}")
print("=" * 70)

print("\n📋 项目文件清单:")
total_size = 0
for f in sorted(os.listdir('.')):
    if os.path.isfile(f):
        size = os.path.getsize(f)
        total_size += size
        is_python = f.endswith('.py')
        marker = '🐍' if is_python else '📄'
        print(f"  {marker} {f:<20} ({size:>8} 字节)")

print(f"\n总大小: {total_size} 字节 ({total_size/1024:.1f} KB)")

print("\n" + "=" * 70)
print("📊 依赖问题诊断")
print("=" * 70)

print("\n当前环境已安装的相关包:")
try:
    import subprocess
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'list', '--format=freeze'],
        capture_output=True, text=True
    )
    
    relevant_packages = ['torch', 'numpy', 'pillow', 'gradio', 'matplotlib', 'torchvision']
    found = []
    for line in result.stdout.split('\n'):
        line_lower = line.lower()
        for pkg in relevant_packages:
            if pkg in line_lower and '==' in line:
                found.append(line)
                print(f"  - {line}")
                break
    
    if not found:
        print("  (未找到相关包，需要安装)")
        
except Exception as e:
    print(f"  无法获取包列表: {e}")

print("\n⚠️  已知问题:")
print("  - NumPy 2.x 与 PyTorch 1.x 可能存在兼容性问题")
print("  - 建议: pip install 'numpy<2.0'")
print("  - 如果网络有问题，可以使用国内镜像:")
print("    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple 'numpy<2.0'")

print("\n" + "=" * 70)
print("✅ 代码结构检查完成")
print("=" * 70)
print("""
项目结构:
├── config.py      # 配置参数
├── models.py      # 网络模型 (U-Net生成器, PatchGAN判别器)
├── datasets.py    # 数据加载和预处理
├── utils.py       # 工具函数和损失函数
├── train.py       # 训练流程
├── infer.py       # 推理功能 (支持预训练模型)
├── app.py         # Gradio Web界面
└── requirements.txt # 依赖列表

快速启动:
1. 安装依赖: pip install -r requirements.txt
2. 启动Web界面: python app.py
3. 打开浏览器: http://localhost:7860
""")
