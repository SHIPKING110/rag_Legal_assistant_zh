# -*- coding: utf-8 -*-
"""
法律条文检索模块
提供关键词搜索和高级筛选功能
"""
import json
from pathlib import Path
from typing import List, Dict, Optional
import streamlit as st

from utils import Config


def load_law_data() -> List[Dict]:
    """加载所有法律数据"""
    if "law_data_cache" in st.session_state:
        return st.session_state.law_data_cache
    
    all_data = []
    data_dir = Path(Config.DATA_DIR)
    
    if not data_dir.exists():
        return []
    
    for json_file in data_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    for full_title, content in item.items():
                        parts = full_title.split(" ", 1)
                        law_name = parts[0] if len(parts) > 0 else "未知法律"
                        article = parts[1] if len(parts) > 1 else "未知条款"
                        all_data.append({
                            "law_name": law_name,
                            "article": article,
                            "full_title": full_title,
                            "content": content,
                            "source_file": json_file.name
                        })
        except Exception as e:
            print(f"[load_law_data] 加载文件失败 {json_file}: {e}")
    
    st.session_state.law_data_cache = all_data
    return all_data


def get_all_law_names(data: List[Dict]) -> List[str]:
    """获取所有法律名称列表"""
    law_names = set()
    for item in data:
        law_names.add(item["law_name"])
    return sorted(list(law_names))


def search_laws(
    data: List[Dict],
    keyword: str = "",
    law_name_filter: Optional[str] = None,
    search_in_content: bool = True,
    search_in_title: bool = True
) -> List[Dict]:
    """搜索法律条文
    
    Args:
        data: 法律数据列表
        keyword: 搜索关键词
        law_name_filter: 法律名称筛选
        search_in_content: 是否在内容中搜索
        search_in_title: 是否在标题中搜索
    
    Returns:
        匹配的法律条文列表
    """
    results = []
    
    for item in data:
        # 法律名称筛选
        if law_name_filter and law_name_filter != "全部" and item["law_name"] != law_name_filter:
            continue
        
        # 关键词搜索
        if keyword:
            keyword_lower = keyword.lower()
            found = False
            
            if search_in_title and keyword_lower in item["full_title"].lower():
                found = True
            if search_in_content and keyword_lower in item["content"].lower():
                found = True
            
            if not found:
                continue
        
        results.append(item)
    
    return results


def highlight_keyword(text: str, keyword: str) -> str:
    """高亮显示关键词"""
    if not keyword:
        return text
    
    import re
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    return pattern.sub(f"**:red[{keyword}]**", text)


def render_legal_search_page():
    """渲染法律检索页面"""
    st.subheader("📚 法律条文检索")
    st.markdown("搜索中华人民共和国法律法规条文")
    
    # 加载数据
    law_data = load_law_data()
    
    if not law_data:
        st.warning("⚠️ 未找到法律数据，请确保数据文件存在")
        return
    
    # 搜索栏 - 横向对齐
    col1, col2 = st.columns([5, 1])
    with col1:
        keyword = st.text_input("🔍 输入关键词搜索", placeholder="例如：劳动合同、婚姻、继承...", label_visibility="collapsed")
    with col2:
        # 添加空行使按钮与输入框对齐
        search_btn = st.button("🔍 搜索", use_container_width=True, type="primary")
    
    # 高级检索选项
    with st.expander("⚙️ 高级检索", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # 法律名称筛选
            law_names = ["全部"] + get_all_law_names(law_data)
            selected_law = st.selectbox("选择法律", law_names)
        
        with col2:
            # 搜索范围
            st.write("搜索范围")
            search_in_title = st.checkbox("标题", value=True)
            search_in_content = st.checkbox("内容", value=True)
        
        with col3:
            # 每页显示条数
            page_size = st.selectbox("每页显示", [20, 50, 100], index=0)
    
    st.divider()
    
    # 初始化分页状态
    if "search_page" not in st.session_state:
        st.session_state.search_page = 1
    
    # 执行搜索
    if keyword or selected_law != "全部":
        results = search_laws(
            law_data,
            keyword=keyword,
            law_name_filter=selected_law if selected_law != "全部" else None,
            search_in_content=search_in_content,
            search_in_title=search_in_title
        )
        
        total_results = len(results)
        total_pages = (total_results + page_size - 1) // page_size if total_results > 0 else 1
        
        # 确保当前页在有效范围内
        if st.session_state.search_page > total_pages:
            st.session_state.search_page = 1
        
        current_page = st.session_state.search_page
        
        # 显示结果统计和分页信息
        col1, col2 = st.columns([2, 1])
        with col1:
            st.caption(f"找到 {total_results} 条相关法律条文")
        with col2:
            st.caption(f"第 {current_page}/{total_pages} 页")
        
        # 计算当前页的数据范围
        start_idx = (current_page - 1) * page_size
        end_idx = min(start_idx + page_size, total_results)
        page_results = results[start_idx:end_idx]
        
        # 显示结果
        for idx, item in enumerate(page_results, start_idx + 1):
            with st.expander(f"**{idx}. {item['full_title']}**", expanded=False):
                # 高亮关键词
                content = item["content"]
                if keyword:
                    content = highlight_keyword(content, keyword)
                
                st.markdown(content)
                st.caption(f"来源: {item['source_file']}")
        
        # 分页控制
        if total_pages > 1:
            st.divider()
            col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
            
            with col1:
                if st.button("⏮ 首页", disabled=current_page == 1):
                    st.session_state.search_page = 1
                    st.rerun()
            
            with col2:
                if st.button("◀ 上一页", disabled=current_page == 1):
                    st.session_state.search_page = current_page - 1
                    st.rerun()
            
            with col3:
                # 页码跳转
                new_page = st.number_input("跳转到", min_value=1, max_value=total_pages, value=current_page, label_visibility="collapsed")
                if new_page != current_page:
                    st.session_state.search_page = new_page
                    st.rerun()
            
            with col4:
                if st.button("下一页 ▶", disabled=current_page == total_pages):
                    st.session_state.search_page = current_page + 1
                    st.rerun()
            
            with col5:
                if st.button("末页 ⏭", disabled=current_page == total_pages):
                    st.session_state.search_page = total_pages
                    st.rerun()
    else:
        # 显示统计信息
        st.info(f"📊 数据库共收录 {len(law_data)} 条法律条文，来自 {len(get_all_law_names(law_data))} 部法律法规")
        st.caption("请输入关键词或选择法律名称进行检索")
    
    # 返回按钮
    st.divider()
    if st.button("← 返回聊天", use_container_width=True):
        st.session_state.show_legal_search = False
        st.rerun()
