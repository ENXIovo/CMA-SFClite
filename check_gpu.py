"""检查 GPU 和 PyTorch 配置"""
import torch

print("="*60)
print("PyTorch 配置检查")
print("="*60)

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"    显存: {props.total_memory / 1024**3:.1f} GB")
else:
    print("⚠️  未检测到 CUDA 支持")
    print("   可能原因：")
    print("   1. 安装了 CPU 版本的 PyTorch")
    print("   2. GPU 驱动未安装或版本不兼容")
    print("   3. CUDA toolkit 未安装")
    
    print("\n推荐安装 CUDA 版 PyTorch：")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

print("="*60)

