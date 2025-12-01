# -*- coding: utf-8 -*-
import json
import time
import os
from pathlib import Path
from typing import List, Dict, Optional
import re
import chromadb
import traceback
from dotenv import load_dotenv, set_key

import streamlit as st
import psutil
import torch
import requests
from llama_index.core import VectorStoreIndex, StorageContext, Settings, get_response_synthesizer
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core import QueryBundle
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import PromptTemplate
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.llms.openai_like import OpenAILike

# 导入配置和模型配置
from utils import Config, LLM_CONFIGS, DEEPSEEK_MODELS, GLM_MODELS

# 设置环境变量，强制使用本地文件
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# 自动加载.env文件
load_dotenv(dotenv_path=Path(__file__).parent / '.env', override=True)

# ================== Streamlit页面配置 ==================
st.set_page_config(
    page_title="智能法律咨询助手",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="auto"
)

def disable_streamlit_watcher():
    """更安全的方式禁用Streamlit文件监视器"""
    try:
        from streamlit import runtime
        if runtime.exists():
            instance = runtime.get_instance()
            def _on_script_changed(_):
                return
            if hasattr(instance, '_on_script_changed'):
                instance._on_script_changed = _on_script_changed
    except Exception as e:
        print(f"禁用文件监视器时出现警告: {e}")

# ================== 设备检测和内存工具 ==================
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

def check_rank_model_memory():
    """检查是否有足够内存加载rank模型"""
    available_memory = get_available_memory_gb()
    required_memory = Config.RERANK_MODEL_MIN_MEMORY_GB
    
    return available_memory >= required_memory, available_memory, required_memory

# ================== 修复后的自定义重排序器 ==================
class SimpleQwenReranker(BaseNodePostprocessor):
    # 使用 Pydantic 字段定义
    model_path: str
    top_n: int = 3
    device: str = "cpu"
    auto_load: bool = False
    
    def __init__(self, model_path: str, top_n: int = 3, device: str = "cpu", auto_load: bool = False):
        # 使用 Pydantic 的方式初始化（所有参数都必须传递给super().__init__）
        super().__init__(model_path=model_path, top_n=top_n, device=device, auto_load=auto_load)
        
        # 初始化内部状态
        self._is_loaded = False
        self._model = None
        
        # 仅当auto_load为True时才加载模型
        if auto_load:
            self._try_load_model()
    
    def _try_load_model(self):
        """尝试加载模型，但不抛出异常"""
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                self.model_path, 
                trust_remote_code=True, 
                local_files_only=True,
                device=self.device
            )
            
            # 修复：设置填充令牌
            if hasattr(self._model, 'tokenizer') and self._model.tokenizer.pad_token is None:
                self._model.tokenizer.pad_token = self._model.tokenizer.eos_token
            
            self._is_loaded = True
            print(f"✅ Qwen3-Reranker 加载成功 (设备: {self.device}): {self.model_path}")
            
        except Exception as e:
            print(f"❌ 重排序模型加载失败: {e}")
            self._is_loaded = False
    
    def is_loaded(self):
        return self._is_loaded
    
    def load_model(self):
        """主动加载模型"""
        if not self._is_loaded:
            self._try_load_model()
        return self._is_loaded
    
    def unload_model(self):
        """卸载模型释放内存"""
        if self._model is not None:
            del self._model
            self._model = None
            self._is_loaded = False
            print("✅ Rank模型已卸载，内存已释放")
    
    def _postprocess_nodes(self, nodes: List[NodeWithScore], query_bundle: QueryBundle):
        if not nodes or not self._is_loaded or self._model is None:
            return nodes[:self.top_n] if nodes else []
        
        try:
            # 准备查询-文档对
            query_doc_pairs = []
            for node in nodes:
                query_doc_pairs.append([query_bundle.query_str, node.node.get_content()])
            
            # 逐个处理，避免批量填充问题
            scores = []
            for pair in query_doc_pairs:
                score = self._model.predict([pair])
                scores.append(float(score[0]))
            
            # 将分数添加到节点
            for node, score in zip(nodes, scores):
                node.score = score
            
            # 按分数排序并返回前top_n个
            sorted_nodes = sorted(nodes, key=lambda x: x.score, reverse=True)
            return sorted_nodes[:self.top_n]
            
        except Exception as e:
            print(f"重排序失败: {e}")
            return nodes[:self.top_n]

# ================== 缓存资源初始化 ==================
@st.cache_resource(show_spinner="初始化模型中...")
def init_models(llm_choice="deepseek", api_key=None, llm_sub_choice=None):
    # 检查嵌入模型是否存在
    embed_model_path = Path(Config.EMBED_MODEL_PATH)
    if not embed_model_path.exists():
        st.error(f"❌ 嵌入模型路径不存在: {Config.EMBED_MODEL_PATH}")
        st.info("请确保模型已正确下载到指定路径")
        st.stop()
    
    embed_model = HuggingFaceEmbedding(
        model_name=str(embed_model_path),
    )
    
    # 检查重排序模型是否存在（默认不加载）
    rerank_model_path = Path(Config.RERANK_MODEL_PATH)
    if not rerank_model_path.exists():
        st.warning(f"⚠️ 重排序模型路径不存在: {Config.RERANK_MODEL_PATH}")
        st.info("rank模型功能不可用")
        reranker = None
    else:
        try:
            # 检测设备
            device, device_name = detect_device()
            
            # 创建重排序器实例，但不自动加载（auto_load=False）
            reranker = SimpleQwenReranker(
                model_path=str(rerank_model_path),
                top_n=Config.RERANK_TOP_K,
                device=device,
                auto_load=False  # 默认不加载
            )
            print(f"✅ Rank模型已初始化（未加载）, 检测到设备: {device_name}")
                
        except Exception as e:
            # 更详细的错误信息
            import traceback
            error_details = traceback.format_exc()
            st.error(f"❌ 重排序模型初始化失败: {str(e)}")
            st.info("将禁用重排序功能，仅使用基础检索")
            reranker = None
    
    # 基础配置
    config = dict(LLM_CONFIGS[llm_choice])
    
    # 根据子模型选择覆盖配置（仅 deepseek / glm 支持）
    if llm_choice == "deepseek" and llm_sub_choice and llm_sub_choice in DEEPSEEK_MODELS:
        config.update(DEEPSEEK_MODELS[llm_sub_choice])
    elif llm_choice == "glm" and llm_sub_choice and llm_sub_choice in GLM_MODELS:
        config.update(GLM_MODELS[llm_sub_choice])
    
    if llm_choice == "deepseek":
        if not api_key:
            st.error("❌ 请提供DeepSeek API Key")
            Settings.embed_model = embed_model
            return embed_model, None, reranker, llm_choice
        
        llm = OpenAILike(
            model=config["model"],
            api_base=config["api_base"],
            api_key=api_key,
            context_window=config["context_window"],
            is_chat_model=True,
            is_function_calling_model=False,
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
            top_p=config["top_p"]
        )
    elif llm_choice == "glm":
        if not api_key:
            st.error("❌ 请提供GLM API Key")
            Settings.embed_model = embed_model
            return embed_model, None, reranker, llm_choice
            
        llm = OpenAILike(
            model=config["model"],
            api_base=config["api_base"],
            api_key=api_key,
            context_window=config["context_window"],
            is_chat_model=True,
            is_function_calling_model=False,
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
            top_p=config["top_p"]
        )
    else:  # local
        llm = OpenAILike(
            model=config["model"],
            api_base=config["api_base"],
            api_key="fake",
            context_window=config["context_window"],
            is_chat_model=True,
            is_function_calling_model=False,
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
            top_p=config["top_p"]
        )
    
    Settings.embed_model = embed_model
    Settings.llm = llm
    
    return embed_model, llm, reranker, llm_choice

@st.cache_resource(show_spinner="加载知识库中...")
def init_vector_store(_nodes):
    chroma_client = chromadb.PersistentClient(path=Config.VECTOR_DB_DIR)
    chroma_collection = chroma_client.get_or_create_collection(
        name=Config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    if chroma_collection.count() == 0 and _nodes is not None:
        # 新建索引
        storage_context = StorageContext.from_defaults(
            vector_store=ChromaVectorStore(chroma_collection=chroma_collection)
        )
        storage_context.docstore.add_documents(_nodes)  
        index = VectorStoreIndex(
            _nodes,
            storage_context=storage_context,
            show_progress=True
        )
        # 创建persist目录
        Path(Config.PERSIST_DIR).mkdir(parents=True, exist_ok=True)
        storage_context.persist(persist_dir=Config.PERSIST_DIR)
        index.storage_context.persist(persist_dir=Config.PERSIST_DIR)
    else:
        # 加载现有索引
        storage_context = StorageContext.from_defaults(
            vector_store=ChromaVectorStore(chroma_collection=chroma_collection)
        )
        index = VectorStoreIndex.from_vector_store(
            storage_context.vector_store,
            storage_context=storage_context,
            embed_model=Settings.embed_model
        )
    return index

# ================== 数据处理 ==================
def load_and_validate_json_files(data_dir: str) -> List[Dict]:
    """加载并验证JSON法律文件"""
    json_files = list(Path(data_dir).glob("*.json"))
    assert json_files, f"未找到JSON文件于 {data_dir}"
    
    all_data = []
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                # 验证数据结构
                if not isinstance(data, list):
                    raise ValueError(f"文件 {json_file.name} 根元素应为列表")
                for item in data:
                    if not isinstance(item, dict):
                        raise ValueError(f"文件 {json_file.name} 包含非字典元素")
                    for k, v in item.items():
                        if not isinstance(v, str):
                            raise ValueError(f"文件 {json_file.name} 中键 '{k}' 的值不是字符串")
                all_data.extend({
                    "content": item,
                    "metadata": {"source": json_file.name}
                } for item in data)
            except Exception as e:
                raise RuntimeError(f"加载文件 {json_file} 失败: {str(e)}")
    
    print(f"成功加载 {len(all_data)} 个法律文件条目")
    return all_data

def create_nodes(raw_data: List[Dict]) -> List[TextNode]:
    """添加ID稳定性保障"""
    nodes = []
    for entry in raw_data:
        law_dict = entry["content"]
        source_file = entry["metadata"]["source"]
        
        for full_title, content in law_dict.items():
            # 生成稳定ID（避免重复）
            node_id = f"{source_file}::{full_title}"
            
            parts = full_title.split(" ", 1)
            law_name = parts[0] if len(parts) > 0 else "未知法律"
            article = parts[1] if len(parts) > 1 else "未知条款"
            
            node = TextNode(
                text=content,
                id_=node_id,  # 显式设置稳定ID
                metadata={
                    "law_name": law_name,
                    "article": article,
                    "full_title": full_title,
                    "source_file": source_file,
                    "content_type": "legal_article"
                }
            )
            nodes.append(node)
    
    print(f"生成 {len(nodes)} 个文本节点（ID示例：{nodes[0].id_}）")
    return nodes

# ================== 界面组件 ==================
def init_sidebar():
    """侧边栏配置"""
    with st.sidebar:
        st.header("⚙️ 功能模块")
        
        # 默认值初始化，避免未进入折叠面板时返回 None
        temperature = 0.3
        top_p = 0.7
        max_tokens = 1024
        top_k = Config.TOP_K
        rerank_top_k = Config.RERANK_TOP_K
        min_rerank_score = 0.4
        api_key = None
        
        # ========= 模型配置 =========
        with st.expander("模型配置", expanded=False):
            # LLM选择（保存到 session_state，便于按钮切换）
            if 'llm_choice_requested' in st.session_state:
                requested = st.session_state.pop('llm_choice_requested')
                st.session_state.llm_choice_select = requested

            if 'llm_choice_select' not in st.session_state:
                st.session_state.llm_choice_select = 'deepseek'

            llm_choice = st.selectbox(
                "选择LLM模型",
                options=["deepseek", "glm", "local"],
                format_func=lambda x: {
                    "deepseek": "DeepSeek",
                    "glm": "智谱GLM", 
                    "local": "本地模型"
                }[x],
                key='llm_choice_select'
            )

            env_path = str(Path(__file__).parent / '.env')
            if llm_choice == 'deepseek':
                current = os.environ.get('LLM_API_KEY', '')
                api_input = st.text_input('DeepSeek API Key', value=current, type='password', help='DeepSeek API Key，留空则不能调用 DeepSeek')
                if api_input and api_input != current:
                    try:
                        set_key(env_path, 'LLM_API_KEY', api_input)
                        os.environ['LLM_API_KEY'] = api_input
                    except Exception as e:
                        print(f"写 .env 失败: {e}")
                api_key = os.environ.get('LLM_API_KEY')
            elif llm_choice == 'glm':
                current = os.environ.get('GLM_API_KEY', '')
                api_input = st.text_input('GLM API Key', value=current, type='password', help='GLM API Key，留空则不能调用 GLM')
                if api_input and api_input != current:
                    try:
                        set_key(env_path, 'GLM_API_KEY', api_input)
                        os.environ['GLM_API_KEY'] = api_input
                    except Exception as e:
                        print(f"写 .env 失败: {e}")
                api_key = os.environ.get('GLM_API_KEY')
            else:
                local_models_dir = Path(__file__).parent / 'model' / 'chat_models'
                local_available = local_models_dir.exists() and any(local_models_dir.iterdir())
                if not local_available:
                    st.warning(f"⚠️ 未检测到本地聊天模型于: {local_models_dir}。请先将模型放入该目录，或切换到云端模型。")
                    col1, col2 = st.columns(2)
                    if col1.button('切换到 DeepSeek'):
                        st.session_state.llm_choice_requested = 'deepseek'
                    if col2.button('切换到 GLM'):
                        st.session_state.llm_choice_requested = 'glm'

            # 子模型选择
            llm_sub_choice = None
            if llm_choice == "deepseek":
                if "llm_sub_choice" not in st.session_state:
                    st.session_state.llm_sub_choice = "deepseek-chat"
                options = list(DEEPSEEK_MODELS.keys())
                llm_sub_choice = st.selectbox(
                    "DeepSeek 子模型",
                    options=options,
                    index=options.index(st.session_state.llm_sub_choice) if st.session_state.llm_sub_choice in options else 0,
                )
                st.session_state.llm_sub_choice = llm_sub_choice
            elif llm_choice == "glm":
                if "llm_sub_choice" not in st.session_state:
                    st.session_state.llm_sub_choice = "glm-4"
                options = list(GLM_MODELS.keys())
                llm_sub_choice = st.selectbox(
                    "GLM 子模型",
                    options=options,
                    index=options.index(st.session_state.llm_sub_choice) if st.session_state.llm_sub_choice in options else 0,
                )
                st.session_state.llm_sub_choice = llm_sub_choice
            else:
                st.session_state.llm_sub_choice = None

        # ========= 模型参数 =========
        with st.expander("模型参数", expanded=False):
            temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
            top_p = st.slider("Top P", 0.0, 1.0, 0.7, 0.1)
            max_tokens = st.slider("最大生成长度", 512, 4096, 1024, 128)

        # ========= 检索参数 =========
        with st.expander("检索参数", expanded=False):
            top_k = st.slider("检索数量", 5, 30, Config.TOP_K, 5)
            rerank_top_k = st.slider("重排序数量", 1, 10, Config.RERANK_TOP_K, 1)
            min_rerank_score = st.slider("最小重排序分数", 0.0, 1.0, 0.4, 0.1)

        Config.TOP_K = top_k
        Config.RERANK_TOP_K = rerank_top_k

        # ========= Rank 模型管理 =========
        with st.expander("⭐ Rank模型管理", expanded=False):
            if "enable_rank_model" not in st.session_state:
                st.session_state.enable_rank_model = False

            device, device_name = detect_device()
            available_memory, required_memory = get_available_memory_gb(), Config.RERANK_MODEL_MIN_MEMORY_GB
            memory_sufficient = available_memory >= required_memory

            st.info(f"📱 检测到设备: {device_name}")
            st.info(f"💾 可用内存: {available_memory:.2f}GB / 需要: {required_memory}GB")

            reranker = st.session_state.get("reranker")
            rank_model_file_exists = Path(Config.RERANK_MODEL_PATH).exists()
            rank_model_available = rank_model_file_exists

            if not rank_model_available:
                st.warning("⚠️ Rank模型不可用（未找到模型文件）")
                enable_rank = False
            elif not memory_sufficient:
                st.warning(f"⚠️ 内存不足！需要{required_memory}GB，当前仅{available_memory:.2f}GB")
                enable_rank = False
            else:
                enable_rank = st.checkbox(
                    "启用Rank重排序模型",
                    value=st.session_state.enable_rank_model,
                    help="启用后会使用AI模型对检索结果进行智能重排序，可能会消耗较多内存"
                )

            if enable_rank and not st.session_state.enable_rank_model:
                st.session_state.enable_rank_model = True
                if reranker is None:
                    st.info("ℹ️ Rank模型将在首次使用时初始化，请先进行一次对话")
                elif hasattr(reranker, 'load_model'):
                    with st.spinner("正在加载Rank模型..."):
                        if reranker.load_model():
                            st.success("✅ Rank模型加载成功")
                        else:
                            st.error("❌ Rank模型加载失败，已禁用")
                            st.session_state.enable_rank_model = False
            elif not enable_rank and st.session_state.enable_rank_model:
                st.session_state.enable_rank_model = False
                if reranker is not None and hasattr(reranker, 'unload_model'):
                    reranker.unload_model()

        # ========= 模型状态 =========
        with st.expander("模型状态", expanded=False):
            reranker = st.session_state.get("reranker")
            rank_model_file_exists = Path(Config.RERANK_MODEL_PATH).exists()
            rank_model_available = rank_model_file_exists
            embed_status = "✅ 已加载" if Path(Config.EMBED_MODEL_PATH).exists() else "❌ 未找到"

            if reranker is not None and hasattr(reranker, 'is_loaded') and reranker.is_loaded():
                rerank_status = "✅ 已启用"
            elif rank_model_available:
                rerank_status = "⏸️ 已初始化（未启用）"
            else:
                rerank_status = "❌ 不可用"

            st.write(f"嵌入模型: {embed_status}")
            st.write(f"Rank模型: {rerank_status}")

        st.info("💡 提示：DeepSeek模型需要有效的API Key，可在官网申请")

        return llm_choice, st.session_state.llm_sub_choice, api_key, temperature, top_p, max_tokens, min_rerank_score

def init_chat_interface():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg.get("cleaned", msg["content"])  # 优先使用清理后的内容
        
        with st.chat_message(role):
            st.markdown(content)
            
            # 如果是助手消息且包含思维链
            if role == "assistant" and msg.get("think"):
                with st.expander("📝 模型思考过程（历史对话）"):
                    for think_content in msg["think"]:
                        st.markdown(f'<span style="color: #808080">{think_content.strip()}</span>',
                                  unsafe_allow_html=True)
            
            # 如果是助手消息且有参考依据（需要保持原有参考依据逻辑）
            if role == "assistant" and "reference_nodes" in msg:
                show_reference_details(msg["reference_nodes"])

def show_reference_details(nodes):
    with st.expander("查看支持依据"):
        for idx, node in enumerate(nodes, 1):
            meta = node.node.metadata
            st.markdown(f"**[{idx}] {meta['full_title']}**")
            st.caption(f"来源文件：{meta['source_file']} | 法律名称：{meta['law_name']}")
            st.markdown(f"相关度：`{node.score:.4f}`")
            st.info(f"{node.node.text}")


def synthesize_with_retries(synthesizer, prompt: str, nodes: List, retries: int = 3, initial_delay: float = 2.0):
    """对 response_synthesizer.synthesize 添加有限重试和指数退避。

    参数:
        synthesizer: response_synthesizer 实例
        prompt: 用户输入
        nodes: 用于合成的节点列表
        retries: 最大重试次数
        initial_delay: 初始等待秒数，后续按 2^n 指数增长
    返回:
        合成器返回的对象（与原 synthesize 返回相同）
    抛出:
        最后一次异常（若全部重试失败）
    """
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return synthesizer.synthesize(prompt, nodes=nodes)
        except Exception as e:
            last_exc = e
            print(f"[synthesize_with_retries] 尝试 {attempt}/{retries} 失败: {e}")
            traceback.print_exc()
            if attempt == retries:
                # 重试完毕，重复抛出最后的异常
                raise
            # 指数退避
            wait = initial_delay * (2 ** (attempt - 1))
            print(f"[synthesize_with_retries] 等待 {wait}s 后重试...")
            time.sleep(wait)


def is_legal_related(question: str, llm) -> bool:
    """判断用户问题是否与法律相关
    
    参数:
        question: 用户输入的问题
        llm: LLM实例，用于判断
    返回:
        True: 与法律相关，需要启用检索
        False: 与法律无关，直接使用对话模型回答
    """
    try:
        # 构建判断提示
        judgment_prompt = f"""请判断以下问题是否与法律、法规、法律咨询、法律问题相关。

问题：{question}

请只回答"是"或"否"，不要添加任何其他内容。

如果问题涉及：
- 法律法规、法律条文、法律条款
- 法律咨询、法律问题、法律纠纷
- 合同、协议、法律文件
- 诉讼、仲裁、法律程序
- 法律权利、法律义务、法律责任
- 任何需要参考法律条文来回答的问题

请回答"是"。

如果问题是一般性对话、闲聊、非法律相关的技术问题、生活常识等，请回答"否"。

回答："""
        
        # 调用LLM进行判断
        response = llm.complete(judgment_prompt)
        result = response.text.strip().lower()
        
        # 解析结果
        if "是" in result or "yes" in result or "true" in result or "1" in result:
            print(f"[is_legal_related] 判断结果：与法律相关")
            return True
        else:
            print(f"[is_legal_related] 判断结果：与法律无关")
            return False
            
    except Exception as e:
        # 如果判断失败，默认启用检索（保守策略）
        print(f"[is_legal_related] 判断失败: {e}，默认启用检索")
        traceback.print_exc()
        return True


def try_auto_switch_llm(current_choice: str) -> bool:
    """尝试自动切换 LLM（按优先级 deepseek -> glm -> local），如果切换成功返回 True。

    切换会调用 `init_models` 并更新 `st.session_state` 与 `Settings.llm`。
    """
    candidates = ["deepseek", "glm", "local"]
    for cand in candidates:
        if cand == current_choice:
            continue

        # 远端模型需要 API Key 且其 api_base 必须可达
        if cand in ("deepseek", "glm"):
            api_key_name = 'LLM_API_KEY' if cand == 'deepseek' else 'GLM_API_KEY'
            api_key = os.environ.get(api_key_name)
            if not api_key:
                print(f"[try_auto_switch_llm] 跳过 {cand}：未找到环境变量 {api_key_name}")
                continue

            api_base = LLM_CONFIGS.get(cand, {}).get('api_base')
            if not api_base:
                print(f"[try_auto_switch_llm] 跳过 {cand}：未配置 api_base")
                continue

            # 轻量可达性检测（快速 HTTP 请求）
            try:
                resp = requests.get(api_base, timeout=2)
                # 仅要求能够建立连接并返回，不强求 200
                print(f"[try_auto_switch_llm] {cand} api_base 可达: {api_base} (status {resp.status_code})")
            except Exception as e:
                print(f"[try_auto_switch_llm] {cand} api_base 不可达: {api_base}，原因: {e}")
                continue

            # 尝试初始化模型
            try:
                embed_model, llm, reranker, chosen = init_models(cand, api_key, None)
                if llm is not None:
                    st.session_state.current_llm_choice = chosen
                    st.session_state.llm = llm
                    st.session_state.embed_model = embed_model
                    st.session_state.reranker = reranker
                    Settings.llm = llm
                    st.success(f"已切换到备用模型: {chosen}")
                    print(f"[try_auto_switch_llm] 已切换到可用模型: {chosen}")
                    return True
            except Exception as e:
                print(f"[try_auto_switch_llm] 初始化 {cand} 失败: {e}")
                traceback.print_exc()
                continue

        # 本地候选：检查本地聊天模型目录
        if cand == 'local':
            local_models_dir = Path(__file__).parent / 'model' / 'chat_models'
            if not (local_models_dir.exists() and any(local_models_dir.iterdir())):
                print(f"[try_auto_switch_llm] 本地模型目录无可用模型: {local_models_dir}")
                continue
            try:
                embed_model, llm, reranker, chosen = init_models('local', None, None)
                if llm is not None:
                    st.session_state.current_llm_choice = chosen
                    st.session_state.llm = llm
                    st.session_state.embed_model = embed_model
                    st.session_state.reranker = reranker
                    Settings.llm = llm
                    st.success("已切换到本地模型")
                    print("[try_auto_switch_llm] 已切换到本地模型")
                    return True
            except Exception as e:
                print(f"[try_auto_switch_llm] 初始化本地模型失败: {e}")
                traceback.print_exc()
                continue

    print("[try_auto_switch_llm] 未找到可用的远程模型或本地备选")
    st.warning("未找到可用的备用模型。请检查 API Key 或本地模型目录。")
    return False

# ================== 主程序 ==================
def main():
    # 禁用 Streamlit 文件热重载（放在更安全的位置）
    try:
        disable_streamlit_watcher()
    except Exception as e:
        # 忽略这个错误，不影响主要功能
        print(f"禁用文件监视器时出现警告: {e}")
    
    st.title("⚖️ 智能法律咨询助手")
    st.markdown("欢迎使用中华人民共和国法律智能咨询系统，请输入您的问题，我们将基于最新中华人民共和国法律法规为您解答。")

    # 侧边栏配置
    llm_choice, llm_sub_choice, api_key, temperature, top_p, max_tokens, min_rerank_score = init_sidebar()
    
    # 更新LLM配置
    if llm_choice in LLM_CONFIGS:
        LLM_CONFIGS[llm_choice]["temperature"] = temperature
        LLM_CONFIGS[llm_choice]["top_p"] = top_p
        LLM_CONFIGS[llm_choice]["max_tokens"] = max_tokens

    # 初始化会话状态
    if "history" not in st.session_state:
        st.session_state.history = []
    
    # 检查是否需要重新初始化模型（当配置改变时，或者模型未初始化时）
    current_config = f"{llm_choice}_{llm_sub_choice}_{api_key}_{temperature}_{top_p}_{max_tokens}"
    need_init = (
        "last_config" not in st.session_state
        or st.session_state.last_config != current_config
        or st.session_state.get("reranker") is None
        or st.session_state.get("llm") is None
    )
    
    if need_init:
        with st.spinner("正在初始化模型..."):
            embed_model, llm, reranker, current_llm_choice = init_models(llm_choice, api_key, llm_sub_choice)
            st.session_state.last_config = current_config
            st.session_state.current_llm_choice = current_llm_choice
            st.session_state.current_llm_sub_choice = llm_sub_choice
            st.session_state.embed_model = embed_model
            st.session_state.llm = llm
            st.session_state.reranker = reranker
    
    # 初始化数据
    if not Path(Config.VECTOR_DB_DIR).exists():
        with st.spinner("正在构建知识库..."):
            raw_data = load_and_validate_json_files(Config.DATA_DIR)
            nodes = create_nodes(raw_data)
    else:
        nodes = None
    
    index = init_vector_store(nodes)
    retriever = index.as_retriever(
        similarity_top_k=Config.TOP_K,
        vector_store_query_mode="hybrid",
        alpha=0.5
    )
    
    response_synthesizer = get_response_synthesizer(verbose=True)
    
    # 聊天界面
    init_chat_interface()
    
    if prompt := st.chat_input("请输入中华人民共和国法律相关问题"):
        # 检查模型是否已正确初始化
        if st.session_state.get("llm") is None:
            st.error("❌ 请先配置API Key并确保模型初始化成功")
            st.stop()
        
        # 添加用户消息到历史
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 处理查询
        with st.spinner("正在分析问题..."):
            start_time = time.time()
            
            # 首先判断问题是否与法律相关
            llm = st.session_state.get("llm")
            if llm is None:
                st.error("❌ LLM未初始化")
                st.stop()
            
            is_legal = is_legal_related(prompt, llm)
            
            if not is_legal:
                # 与法律无关，直接使用对话模型回答
                st.info("💬 检测到问题与法律无关，使用对话模式回答")
                try:
                    response = llm.complete(prompt)
                    response_text = response.text
                    filtered_nodes = []  # 非法律问题没有参考依据
                except Exception as e:
                    print(f"[main] 直接对话模式失败: {e}")
                    traceback.print_exc()
                    response_text = f"抱歉，我无法回答这个问题。错误信息：{str(e)}"
                    filtered_nodes = []
            else:
                # 与法律相关，启用检索流程
                st.info("⚖️ 检测到问题与法律相关，启用法律检索模式")
                
                # 检索流程
                initial_nodes = retriever.retrieve(prompt)
                
                # 使用会话状态中的 reranker（仅在启用且已加载时使用）
                reranker = st.session_state.reranker
                enable_rank = st.session_state.get("enable_rank_model", False)
                
                if enable_rank and reranker is not None and hasattr(reranker, 'is_loaded') and reranker.is_loaded():
                    try:
                        reranked_nodes = reranker.postprocess_nodes(initial_nodes, query_str=prompt)
                        # 过滤节点
                        filtered_nodes = [node for node in reranked_nodes if node.score > min_rerank_score]
                        st.success("✅ 已使用重排序功能")
                    except Exception as e:
                        st.warning(f"⚠️ 重排序失败: {e}，使用基础检索结果")
                        filtered_nodes = initial_nodes[:Config.RERANK_TOP_K]
                else:
                    # 如果没有启用重排序模型，直接使用初始节点
                    st.info("⚠️ Rank模型未启用，使用基础检索结果")
                    filtered_nodes = initial_nodes[:Config.RERANK_TOP_K]  # 取前几个节点
                
                if not filtered_nodes:
                    response_text = "⚠️ 未找到相关法律条文，请尝试调整问题描述或咨询专业律师。"
                else:
                    # 生成回答（安全调用：带重试与回退）
                    try:
                        response = synthesize_with_retries(response_synthesizer, prompt, filtered_nodes, retries=3)
                        response_text = response.response
                    except Exception as e:
                        # 打印详细跟踪以便调试
                        print("[main] response_synthesizer 生成失败，进入回退逻辑:")
                        traceback.print_exc()
                        # 向用户显示友好提示
                        st.error("⚠️ 后端模型服务异常，正在尝试切换备用模型或回退为临时结果。")

                        # 优先尝试自动切换到其它可用模型并重试一次
                        switched = False
                        try:
                            switched = try_auto_switch_llm(st.session_state.get('current_llm_choice', llm_choice))
                        except Exception as e_switch:
                            print(f"[main] 自动切换模型过程发生错误: {e_switch}")

                        if switched:
                            try:
                                # 使用新的 LLM 重新创建合成器并重试
                                response_synthesizer = get_response_synthesizer(verbose=True)
                                response = synthesize_with_retries(response_synthesizer, prompt, filtered_nodes, retries=2)
                                response_text = response.response
                                # 如果成功则跳过后续回退逻辑
                            except Exception as e2:
                                print("[main] 切换到备用模型后重试仍失败:", e2)
                                traceback.print_exc()
                                switched = False

                        if not switched:
                            # 将前3条检索到的文档拼接为临时内容
                            concatenated = "\n\n".join([n.node.text for n in filtered_nodes[:3]])

                            # 尝试使用本地小模型做快速摘要（如果配置并存在本地模型）
                            summary_text = None
                            try:
                                local_cfg = LLM_CONFIGS.get("local")
                                if local_cfg:
                                    local_model_path = local_cfg.get("model")
                                    if local_model_path and Path(local_model_path).exists():
                                        try:
                                            hf_llm = HuggingFaceLLM(model_name=str(local_model_path), temperature=0.2, max_length=256)
                                            # 使用hf_llm进行快速摘要
                                            summary_text = hf_llm.predict(f"请简要总结以下法律条文要点：\n\n{concatenated}\n\n总结：")
                                        except Exception as e_local:
                                            print(f"[fallback] 本地模型摘要失败: {e_local}")
                            except Exception as e_cfg:
                                print(f"[fallback] 检查本地模型时发生错误: {e_cfg}")

                            if summary_text:
                                cleaned_response = summary_text
                                response_text = f"⚠️ 后端服务异常，使用本地模型生成的临时摘要：\n\n{summary_text}"
                            else:
                                # 回退到拼接的原文
                                cleaned_response = concatenated
                                response_text = f"⚠️ 后端模型服务异常：{e}\n\n相关条文（临时结果）：\n{concatenated}"
            
            # 显示回答
            with st.chat_message("assistant"):
                # 提取思维链内容并清理响应文本
                think_contents = re.findall(r'<think>(.*?)</think>', response_text, re.DOTALL)
                cleaned_response = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
                
                # 显示清理后的回答
                st.markdown(cleaned_response)
                
                # 如果有思维链内容则显示
                if think_contents:
                    with st.expander("📝 模型思考过程（点击展开）"):
                        for content in think_contents:
                            st.markdown(f'<span style="color: #808080">{content.strip()}</span>', 
                                      unsafe_allow_html=True)
                
                # 仅在有参考依据时显示（法律相关问题才有参考依据）
                if filtered_nodes:
                    show_reference_details(filtered_nodes[:3])

            # 添加助手消息到历史（需要存储原始响应）
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,  # 保留原始响应
                "cleaned": cleaned_response,  # 存储清理后的文本
                "think": think_contents,  # 存储思维链内容
                "reference_nodes": filtered_nodes[:3]  # 存储参考节点
            })

if __name__ == "__main__":
    main()