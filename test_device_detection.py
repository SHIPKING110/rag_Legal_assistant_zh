# -*- coding: utf-8 -*-
"""
测试设备检测和内存检查功能
"""
import torch
import psutil

def detect_device():
    """检测设备是否支持GPU，返回设备类型"""
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        return device, f"GPU ({device_name})"
    else:
        return "cpu", "CPU"

def get_available_memory_gb():
    """获取系统可用内存（GB）"""
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024 ** 3)
    return available_gb

def check_rank_model_memory(min_memory_gb=4):
    """检查是否有足够内存加载rank模型"""
    available_memory = get_available_memory_gb()
    required_memory = min_memory_gb
    
    return available_memory >= required_memory, available_memory, required_memory

if __name__ == "__main__":
    print("=" * 60)
    print("设备和内存检测测试")
    print("=" * 60)
    
    # 测试设备检测
    device, device_name = detect_device()
    print(f"\n✅ 检测到设备: {device_name}")
    print(f"   设备类型: {device}")
    
    # 测试内存检查
    print("\n" + "=" * 60)
    print("内存信息")
    print("=" * 60)
    
    available_memory = get_available_memory_gb()
    total_memory = psutil.virtual_memory().total / (1024 ** 3)
    used_memory = psutil.virtual_memory().used / (1024 ** 3)
    percent = psutil.virtual_memory().percent
    
    print(f"\n总内存: {total_memory:.2f}GB")
    print(f"已用内存: {used_memory:.2f}GB")
    print(f"可用内存: {available_memory:.2f}GB")
    print(f"使用率: {percent}%")
    
    # 测试rank模型内存检查
    print("\n" + "=" * 60)
    print("Rank模型内存检查")
    print("=" * 60)
    
    min_memory = 4
    memory_sufficient, available_mem, required_mem = check_rank_model_memory(min_memory)
    
    print(f"\n所需内存: {required_mem}GB")
    print(f"可用内存: {available_mem:.2f}GB")
    
    if memory_sufficient:
        print(f"✅ 内存充足，可以启用Rank模型")
    else:
        print(f"❌ 内存不足！")
        print(f"   缺少: {required_mem - available_mem:.2f}GB")
        print(f"   建议: 关闭其他应用以释放内存，或提高最小内存要求")
    
    print("\n" + "=" * 60)
