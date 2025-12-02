# -*- coding: utf-8 -*-
"""
游客页面模块
游客只能访问法律检索功能
"""
import streamlit as st
from webui.legal_search import render_legal_search_page


def render_guest_page():
    """渲染游客专用页面"""
    st.title("⚖️ 智能法律咨询助手 - 游客模式")
    st.info("💡 游客模式仅提供法律检索功能，注册登录后可使用完整功能（AI对话、会话历史等）")
    
    # 侧边栏提示
    with st.sidebar:
        st.header("👤 游客模式")
        st.caption("当前功能受限")
        st.markdown("---")
        
        st.markdown("### 可用功能")
        st.markdown("✅ 法律检索")
        
        st.markdown("### 受限功能")
        st.markdown("❌ AI法律咨询")
        st.markdown("❌ 会话历史")
        st.markdown("❌ 个人档案")
        
        st.markdown("---")
        
        if st.button("🚪 退出登录", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()
        
        st.markdown("---")
        st.info("💡 注册账号解锁全部功能")
    
    # 显示法律检索页面
    render_legal_search_page()
