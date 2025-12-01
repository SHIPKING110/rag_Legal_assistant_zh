# -*- coding: utf-8 -*-
"""
配置类模块
包含项目的所有配置项
"""


class Config:
    """项目配置类"""
    # 模型路径配置
    EMBED_MODEL_PATH = r"./model/embedding_model/fengshan/ChatLaw-Text2Vec"
    RERANK_MODEL_PATH = r"./model/rank/Qwen/Qwen3-Reranker-0___6B"

    # 数据路径配置
    DATA_DIR = "./json_data"
    VECTOR_DB_DIR = "./chroma_db"
    PERSIST_DIR = "./storage"
    
    # 向量数据库配置
    COLLECTION_NAME = "chinese_labor_laws"
    
    # 检索参数配置
    TOP_K = 10
    RERANK_TOP_K = 3
    
    # 模型资源要求配置
    RERANK_MODEL_MIN_MEMORY_GB = 4  # rank模型最小需要的内存（GB）

