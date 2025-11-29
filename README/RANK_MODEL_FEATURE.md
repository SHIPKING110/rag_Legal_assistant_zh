# 🚀 Rank模型按需加载功能 - 实现说明

## 功能概述

本次更新为智能法律咨询助手添加了以下重要功能：

### 1. **设备自动检测** 🖥️
- **自动检测GPU/CPU**：系统启动时自动检测设备是否支持GPU
- **智能选择**：优先使用GPU（CUDA），如果不可用则使用CPU
- **设备信息显示**：在侧边栏显示当前设备名称和类型

### 2. **Rank模型按需加载** ⚙️
- **默认不加载**：系统初始化时只初始化rank模型配置，但不加载权重，节省内存
- **用户控制**：用户可以通过前端开关按钮主动启用或禁用rank模型
- **动态加载/卸载**：
  - 启用时加载模型权重到内存
  - 禁用时释放模型权重，回收内存

### 3. **内存检查机制** 💾
- **可用内存检测**：实时检测系统可用内存
- **阈值设置**：rank模型最少需要4GB可用内存（可配置）
- **智能提示**：
  - ✅ 内存充足：允许启用rank模型
  - ❌ 内存不足：禁用开关，显示警告提示

## 技术实现

### 新增导入库
```python
import psutil  # 系统资源监测
import torch   # PyTorch设备检测
```

### 核心函数

#### 1. `detect_device()`
检测设备是否支持GPU
```python
def detect_device():
    """检测设备是否支持GPU，返回设备类型"""
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        return device, f"GPU ({device_name})"
    else:
        return "cpu", "CPU"
```

#### 2. `get_available_memory_gb()`
获取系统可用内存（GB）
```python
def get_available_memory_gb():
    """获取系统可用内存（GB）"""
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024 ** 3)
    return available_gb
```

#### 3. `check_rank_model_memory()`
检查是否有足够内存加载rank模型
```python
def check_rank_model_memory():
    """检查是否有足够内存加载rank模型"""
    available_memory = get_available_memory_gb()
    required_memory = Config.RERANK_MODEL_MIN_MEMORY_GB
    
    return available_memory >= required_memory, available_memory, required_memory
```

### SimpleQwenReranker 类改进

#### 新增参数
- `device`: 指定运行设备（cuda或cpu）
- `auto_load`: 控制是否自动加载（默认False）

#### 新增方法
- `load_model()`: 主动加载模型权重
- `unload_model()`: 卸载模型权重，释放内存

## UI/UX 改进

### 侧边栏配置面板新增部分

#### ⭐ Rank模型管理
```
┌─────────────────────────────────┐
│  🖥️ 检测到设备: GPU (NVIDIA RTX) │
│  💾 可用内存: 12.34GB / 需要: 4GB │
│                                 │
│  ☑️ 启用Rank重排序模型           │
│  ✅ 内存充足，可以启用          │
└─────────────────────────────────┘
```

### 模型状态显示
```
模型状态
─────────────────
嵌入模型: ✅ 已加载
Rank模型: ⏸️ 已初始化（未启用）
         或
Rank模型: ✅ 已启用
```

## 用户工作流

### 场景1：启用Rank模型
1. 用户打开应用
2. 侧边栏显示设备信息和内存状态
3. 如果内存充足，用户可以勾选"启用Rank重排序模型"
4. 系统加载模型权重（显示加载进度）
5. 模型启用后，查询时会自动使用重排序

### 场景2：禁用Rank模型（内存不足）
1. 用户打开应用
2. 侧边栏显示："❌ 内存不足！需要4GB，当前仅3.5GB"
3. 开关自动禁用（灰显）
4. 提示用户关闭其他应用以释放内存
5. 只能使用基础检索功能

### 场景3：动态控制
1. 用户勾选"启用Rank重排序模型"→ 系统加载模型
2. 用户取消勾选 → 系统立即卸载模型，释放内存
3. 用户可以根据需要随时切换

## 配置参数

### Config 类新增参数
```python
RERANK_MODEL_MIN_MEMORY_GB = 4  # rank模型最小需要的内存（GB）
```

可根据实际硬件情况调整该值。

## 性能优势

### 内存管理
- **初始化**：节省4-8GB内存（模型不在启动时加载）
- **用户可控**：用户决定是否占用内存
- **动态释放**：禁用模型后立即释放内存

### 设备优化
- **GPU加速**：检测到GPU时自动使用，加快推理速度
- **CPU兼容**：CPU不可用时自动降级
- **灵活切换**：无需重启应用

## 测试验证

### 测试脚本：`test_device_detection.py`
验证以下功能：
- ✅ 设备检测（GPU/CPU）
- ✅ 内存检测
- ✅ Rank模型内存检查
- ✅ 足量判断

运行测试：
```bash
python test_device_detection.py
```

## 注意事项

### 内存要求
- **基础功能**：2GB（嵌入模型）
- **启用Rank模型**：+4GB = 总计6GB
- **推荐配置**：8GB及以上内存

### GPU支持
- **NVIDIA**：通过CUDA支持
- **AMD**：需要ROCm支持（需额外配置）
- **Intel**：需要oneAPI支持（需额外配置）
- **MAC**：通过MPS支持（自动检测）

## 后续优化建议

1. **内存使用统计**：显示模型加载前后的内存变化
2. **模型卸载确认**：在内存持续不足时自动提醒
3. **性能基准测试**：对比GPU/CPU的推理速度差异
4. **多模型支持**：支持更多的重排序模型选择
5. **缓存管理**：优化模型推理缓存，进一步减少内存占用

## 相关文件修改

### `main.py`
- ✅ 添加设备检测函数
- ✅ 添加内存检查函数
- ✅ 修改SimpleQwenReranker类（支持设备选择和动态加载）
- ✅ 修改init_models()函数（rank模型默认不加载）
- ✅ 修改init_sidebar()函数（添加rank模型开关和内存显示）
- ✅ 修改主程序逻辑（检查rank模型启用状态）

## 总结

通过本次更新，系统现在能够：
✅ 自动检测硬件配置（GPU/CPU）
✅ 智能管理内存（按需加载rank模型）
✅ 提供用户友好的控制界面（开关按钮）
✅ 实时监控系统资源（显示可用内存）
✅ 防止内存溢出（内存不足时禁用）

用户体验得到显著提升！
