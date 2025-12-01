# -*- coding: utf-8 -*-
"""
LLM configuration module.

说明：
- 保持原有 LLM_CONFIGS 结构不变，兼容现有 main.py 逻辑
- 在此基础上，为 deepseek / glm 增加「子模型」配置，方便后续在 UI 中扩展选择
"""

# 顶层配置（保持兼容）——用于当前 main.py 的 LLM_CONFIGS[llm_choice] 访问
LLM_CONFIGS = {
    "deepseek": {
        "model": "deepseek-chat",
        "api_base": "https://api.deepseek.com/v1",
        "context_window": 32768,
        "max_tokens": 2048,
        "temperature": 0.3,
        "top_p": 0.7,
    },
    "glm": {
        "model": "glm-4",
        "api_base": "https://open.bigmodel.cn/api/paas/v4",
        "context_window": 128000,
        "max_tokens": 1024,
        "temperature": 0.3,
        "top_p": 0.7,
    },
    "qwen": {
        "model": "qwen-plus",
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "context_window": 128000,
        "max_tokens": 2048,
        "temperature": 0.3,
        "top_p": 0.7,
    },
    "local": {
        "model": "/home/cw/llms/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "api_base": "http://localhost:23333/v1",
        "context_window": 4096,
        "max_tokens": 1024,
        "temperature": 0.3,
        "top_p": 0.7,
    },
}


# ====== 子模型配置（按厂商细分）======

# DeepSeek 系列模型子选项
DEEPSEEK_MODELS = {
    # 默认聊天模型（当前正在使用）
    "deepseek-chat": {
        "model": "deepseek-chat",
        "api_base": "https://api.deepseek.com/v1",
        "context_window": 32768,
        "max_tokens": 2048,
        "temperature": 0.3,
        "top_p": 0.7,
    },
    # 推理增强模型
    "deepseek-reasoner": {
        "model": "deepseek-reasoner",
        "api_base": "https://api.deepseek.com/v1",
        "context_window": 32768,
        "max_tokens": 2048,
        "temperature": 0.2,
        "top_p": 0.7,
    },
    # 最新 V3 高性能基础模型
    "deepseek-v3": {
        "model": "deepseek-v3",
        "api_base": "https://api.deepseek.com/v1",
        "context_window": 32768,
        "max_tokens": 2048,
        "temperature": 0.3,
        "top_p": 0.7,
    },
    # 强化推理模型（基于 V3 架构）
    "deepseek-r1": {
        "model": "deepseek-r1",
        "api_base": "https://api.deepseek.com/v1",
        "context_window": 32768,
        "max_tokens": 2048,
        "temperature": 0.2,
        "top_p": 0.7,
    },
}


# GLM 系列模型子选项（根据实际需要可继续扩展）
GLM_MODELS = {
    # 默认 GLM-4
    "glm-4": {
        "model": "glm-4",
        "api_base": "https://open.bigmodel.cn/api/paas/v4",
        "context_window": 128000,
        "max_tokens": 1024,
        "temperature": 0.3,
        "top_p": 0.7,
    },
    # 更大或更强的 GLM 模型（名称按实际 API 为准）
    "glm-4-plus": {
        "model": "glm-4-plus",
        "api_base": "https://open.bigmodel.cn/api/paas/v4",
        "context_window": 128000,
        "max_tokens": 2048,
        "temperature": 0.3,
        "top_p": 0.7,
    },
    # 超长上下文版本
    "glm-4-long": {
        "model": "glm-4-long",
        "api_base": "https://open.bigmodel.cn/api/paas/v4",
        "context_window": 128000,
        "max_tokens": 2048,
        "temperature": 0.3,
        "top_p": 0.7,
    },
    # 推理优化 / 极速版本
    "glm-4-flash": {
        "model": "glm-4-flash",
        "api_base": "https://open.bigmodel.cn/api/paas/v4",
        "context_window": 128000,
        "max_tokens": 2048,
        "temperature": 0.2,
        "top_p": 0.7,
    },
    # GLM-4.5 系列
    "glm-4-5": {
        "model": "glm-4-5",
        "api_base": "https://open.bigmodel.cn/api/paas/v4",
        "context_window": 128000,
        "max_tokens": 2048,
        "temperature": 0.3,
        "top_p": 0.7,
    },
    "glm-4-5-air": {
        "model": "glm-4-5-air",
        "api_base": "https://open.bigmodel.cn/api/paas/v4",
        "context_window": 128000,
        "max_tokens": 2048,
        "temperature": 0.3,
        "top_p": 0.7,
    },
    # GLM-4.6 系列（代码 / 长上下文）
    "glm-4-6": {
        "model": "glm-4-6",
        "api_base": "https://open.bigmodel.cn/api/paas/v4",
        "context_window": 200000,
        "max_tokens": 4096,
        "temperature": 0.3,
        "top_p": 0.7,
    },
    "glm-4-6-9b": {
        "model": "glm-4-6-9b",
        "api_base": "https://open.bigmodel.cn/api/paas/v4",
        "context_window": 200000,
        "max_tokens": 4096,
        "temperature": 0.3,
        "top_p": 0.7,
    },
    # Z1 推理系列
    "glm-z1-air": {
        "model": "glm-z1-air",
        "api_base": "https://open.bigmodel.cn/api/paas/v4",
        "context_window": 128000,
        "max_tokens": 2048,
        "temperature": 0.2,
        "top_p": 0.7,
    },
    "glm-z1-flash": {
        "model": "glm-z1-flash",
        "api_base": "https://open.bigmodel.cn/api/paas/v4",
        "context_window": 128000,
        "max_tokens": 2048,
        "temperature": 0.2,
        "top_p": 0.7,
    },
    # 多模态模型
    "glm-4v": {
        "model": "glm-4v",
        "api_base": "https://open.bigmodel.cn/api/paas/v4",
        "context_window": 128000,
        "max_tokens": 2048,
        "temperature": 0.3,
        "top_p": 0.7,
    },
}


# Qwen 系列模型子选项
QWEN_MODELS = {
    # 通义千问 Max 系列（性能最强）
    "qwen3-max": {
        "model": "qwen3-max",
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "context_window": 200000,
        "max_tokens": 4096,
        "temperature": 0.3,
        "top_p": 0.7,
    },
    "qwen3-max-2025-09-23": {
        "model": "qwen3-max-2025-09-23",
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "context_window": 200000,
        "max_tokens": 4096,
        "temperature": 0.3,
        "top_p": 0.7,
    },
    "qwen3-max-preview": {
        "model": "qwen3-max-preview",
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "context_window": 200000,
        "max_tokens": 4096,
        "temperature": 0.3,
        "top_p": 0.7,
    },
    "qwen-max": {
        "model": "qwen-max",
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "context_window": 200000,
        "max_tokens": 4096,
        "temperature": 0.3,
        "top_p": 0.7,
    },

    # 通义千问 Plus 系列（均衡高性能）
    "qwen-plus": {
        "model": "qwen-plus",
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "context_window": 128000,
        "max_tokens": 2048,
        "temperature": 0.3,
        "top_p": 0.7,
    },
    "qwen-plus-latest": {
        "model": "qwen-plus-latest",
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "context_window": 128000,
        "max_tokens": 2048,
        "temperature": 0.3,
        "top_p": 0.7,
    },
    "qwen-plus-2025-09-11": {
        "model": "qwen-plus-2025-09-11",
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "context_window": 128000,
        "max_tokens": 2048,
        "temperature": 0.3,
        "top_p": 0.7,
    },

    # 通义千问 Flash / Turbo 系列（极速低成本）
    "qwen-flash": {
        "model": "qwen-flash",
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "context_window": 128000,
        "max_tokens": 2048,
        "temperature": 0.3,
        "top_p": 0.8,
    },
    "qwen-flash-2025-07-28": {
        "model": "qwen-flash-2025-07-28",
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "context_window": 128000,
        "max_tokens": 2048,
        "temperature": 0.3,
        "top_p": 0.8,
    },
    "qwen-turbo": {
        "model": "qwen-turbo",
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "context_window": 128000,
        "max_tokens": 2048,
        "temperature": 0.3,
        "top_p": 0.8,
    },
    "qwen-turbo-latest": {
        "model": "qwen-turbo-latest",
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "context_window": 128000,
        "max_tokens": 2048,
        "temperature": 0.3,
        "top_p": 0.8,
    },

    # 通义千问长上下文 / Long 系列
    "qwen-long": {
        "model": "qwen-long",
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "context_window": 1000000,
        "max_tokens": 2048,
        "temperature": 0.3,
        "top_p": 0.7,
    },
    "qwen-plus-long": {
        "model": "qwen-plus-long",
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "context_window": 200000,
        "max_tokens": 4096,
        "temperature": 0.3,
        "top_p": 0.7,
    },

    # Coder / 数学等专用模型
    "qwen3-coder-plus": {
        "model": "qwen3-coder-plus",
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "context_window": 128000,
        "max_tokens": 4096,
        "temperature": 0.2,
        "top_p": 0.7,
    },
    "qwen3-coder-plus-2025-09-23": {
        "model": "qwen3-coder-plus-2025-09-23",
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "context_window": 128000,
        "max_tokens": 4096,
        "temperature": 0.2,
        "top_p": 0.7,
    },
    "qwen3-coder-flash": {
        "model": "qwen3-coder-flash",
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "context_window": 128000,
        "max_tokens": 2048,
        "temperature": 0.2,
        "top_p": 0.7,
    },
    "qwen-coder-plus": {
        "model": "qwen-coder-plus",
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "context_window": 128000,
        "max_tokens": 4096,
        "temperature": 0.2,
        "top_p": 0.7,
    },
    "qwen-math-plus": {
        "model": "qwen-math-plus",
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "context_window": 128000,
        "max_tokens": 2048,
        "temperature": 0.2,
        "top_p": 0.7,
    },

    # 开源指令 / Omni 系列
    "qwen2-72b-instruct": {
        "model": "qwen2-72b-instruct",
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "context_window": 200000,
        "max_tokens": 4096,
        "temperature": 0.3,
        "top_p": 0.7,
    },
    "qwen2.5-32b-instruct": {
        "model": "qwen2.5-32b-instruct",
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "context_window": 128000,
        "max_tokens": 4096,
        "temperature": 0.3,
        "top_p": 0.7,
    },
}


__all__ = ["LLM_CONFIGS", "DEEPSEEK_MODELS", "GLM_MODELS", "QWEN_MODELS"]

