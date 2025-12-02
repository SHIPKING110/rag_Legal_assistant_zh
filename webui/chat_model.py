# -*- coding: utf-8 -*-
"""
聊天模型模块
处理用户输入、法律判断、检索、生成回答等核心聊天功能
"""
import re
import time
import traceback
from pathlib import Path
from typing import List

import streamlit as st
from llama_index.core import get_response_synthesizer
from llama_index.llms.huggingface import HuggingFaceLLM

from utils import Config, LLM_CONFIGS


def init_chat_interface():
    """初始化聊天界面，显示历史消息"""
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
    """显示参考依据详情"""
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


def handle_chat_message(
    prompt: str,
    retriever,
    response_synthesizer,
    llm_choice: str,
    min_rerank_score: float,
    try_auto_switch_llm_func
):
    """处理用户聊天消息
    
    参数:
        prompt: 用户输入的问题
        retriever: 检索器实例
        response_synthesizer: 响应合成器实例
        llm_choice: 当前选择的LLM模型
        min_rerank_score: 最小重排序分数阈值
        try_auto_switch_llm_func: 自动切换LLM的函数
    返回:
        tuple: (response_text, filtered_nodes, used_rank)
    """
    start_time = time.time()
    used_rank = False
    
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
            print(f"[handle_chat_message] 直接对话模式失败: {e}")
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
                used_rank = True
            except Exception as e:
                st.warning(f"⚠️ 重排序失败: {e}，使用基础检索结果")
                # 回退到按检索相似度排序的前 TOP_K 条
                filtered_nodes = initial_nodes[:Config.TOP_K]
        else:
            # 如果没有启用重排序模型，直接使用初始节点
            st.info("⚠️ Rank模型未启用，使用基础检索结果")
            filtered_nodes = initial_nodes[:Config.TOP_K]  # 使用检索得到的前 TOP_K 条
        
        if not filtered_nodes:
            response_text = "⚠️ 未找到相关法律条文，请尝试调整问题描述或咨询专业律师。"
        else:
            # 构造带有法律RAG提示词的系统提示
            legal_prompt_text = ""
            try:
                legal_prompt_path = Path(Config.LEGAL_CHAT_PROMPT_PATH)
                if legal_prompt_path.exists():
                    legal_prompt_text = legal_prompt_path.read_text(encoding="utf-8")
            except Exception as e:
                print(f"[handle_chat_message] 读取法律提示词模版失败: {e}")

            if legal_prompt_text:
                full_prompt = f"{legal_prompt_text}\n\n用户问题：{prompt}"
            else:
                full_prompt = prompt

            # 生成回答（安全调用：带重试与回退）
            try:
                response = synthesize_with_retries(response_synthesizer, full_prompt, filtered_nodes, retries=3)
                response_text = response.response
            except Exception as e:
                # 打印详细跟踪以便调试
                print("[handle_chat_message] response_synthesizer 生成失败，进入回退逻辑:")
                traceback.print_exc()
                # 向用户显示友好提示
                st.error("⚠️ 后端模型服务异常，正在尝试切换备用模型或回退为临时结果。")

                # 优先尝试自动切换到其它可用模型并重试一次
                switched = False
                try:
                    switched = try_auto_switch_llm_func(st.session_state.get('current_llm_choice', llm_choice))
                except Exception as e_switch:
                    print(f"[handle_chat_message] 自动切换模型过程发生错误: {e_switch}")

                if switched:
                    try:
                        # 使用新的 LLM 重新创建合成器并重试
                        response_synthesizer = get_response_synthesizer(verbose=True)
                        response = synthesize_with_retries(response_synthesizer, prompt, filtered_nodes, retries=2)
                        response_text = response.response
                        # 如果成功则跳过后续回退逻辑
                    except Exception as e2:
                        print("[handle_chat_message] 切换到备用模型后重试仍失败:", e2)
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
    
    return response_text, filtered_nodes, used_rank


def display_chat_response(response_text: str, filtered_nodes: List, used_rank: bool):
    """显示聊天响应
    
    参数:
        response_text: 响应文本
        filtered_nodes: 过滤后的节点列表
        used_rank: 是否使用了Rank模型
    """
    # 提取思维链内容并清理响应文本
    think_contents = re.findall(r'<think>(.*?)</think>', response_text, re.DOTALL)
    cleaned_response = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
    
    # 显示回答
    with st.chat_message("assistant"):
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
            # 展示数量与检索/重排序设置联动：
            # - 启用并成功使用 Rank 时：最多展示 RERANK_TOP_K 条
            # - 未启用 Rank 时：展示所有检索得到的条文（已按 TOP_K 截断）
            if used_rank:
                ref_k = min(Config.RERANK_TOP_K, len(filtered_nodes))
            else:
                ref_k = len(filtered_nodes)
            show_reference_details(filtered_nodes[:ref_k])
    
    # 添加助手消息到历史（需要存储原始响应）
    if filtered_nodes:
        if used_rank:
            ref_k = min(Config.RERANK_TOP_K, len(filtered_nodes))
        else:
            ref_k = len(filtered_nodes)
    else:
        ref_k = 0
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,  # 保留原始响应
        "cleaned": cleaned_response,  # 存储清理后的文本
        "think": think_contents,  # 存储思维链内容
        "reference_nodes": filtered_nodes[:ref_k] if filtered_nodes else []  # 存储参考节点
    })

