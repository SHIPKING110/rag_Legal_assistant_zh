# -*- coding: utf-8 -*-
"""
LLM configuration module.
"""

# 统一管理可选 LLM 模型配置
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

__all__ = ["LLM_CONFIGS"]

