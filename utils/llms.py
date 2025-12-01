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
    # R1 系列完整推理模型（根据实际模型名称可再调整）
    "deepseek-r1v3-full": {
        "model": "deepseek-r1v3-full",
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
    # 示例：更大或更强的 GLM 模型（名称按实际 API 为准）
    "glm-4-plus": {
        "model": "glm-4-plus",
        "api_base": "https://open.bigmodel.cn/api/paas/v4",
        "context_window": 128000,
        "max_tokens": 2048,
        "temperature": 0.3,
        "top_p": 0.7,
    },
    # 示例：推理优化模型
    "glm-4-reasoning": {
        "model": "glm-4-reasoning",
        "api_base": "https://open.bigmodel.cn/api/paas/v4",
        "context_window": 128000,
        "max_tokens": 2048,
        "temperature": 0.2,
        "top_p": 0.7,
    },
}


__all__ = ["LLM_CONFIGS", "DEEPSEEK_MODELS", "GLM_MODELS"]

