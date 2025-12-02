# -*- coding: utf-8 -*-
"""
管理员面板模块
提供系统管理、用户管理、数据监控等功能
"""
import json
import os
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import shutil

import streamlit as st


class AdminManager:
    """管理员管理器"""
    
    ADMIN_USERNAME = "root"
    ADMIN_PASSWORD_HASH = hashlib.sha256("123456".encode()).hexdigest()
    
    def __init__(self):
        self.users_file = "./rag_falv_data/users.json"
        self.users_dir = "./rag_falv_data/users"
    
    @staticmethod
    def is_admin_credentials(username: str, password: str) -> bool:
        """验证管理员凭据"""
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        return (username == AdminManager.ADMIN_USERNAME and 
                password_hash == AdminManager.ADMIN_PASSWORD_HASH)
    
    def get_all_users(self) -> List[Dict]:
        """获取所有用户信息（排除密码哈希）"""
        users_list = []
        try:
            if Path(self.users_file).exists():
                with open(self.users_file, 'r', encoding='utf-8') as f:
                    users = json.load(f)
                    
                    for username, user_info in users.items():
                        # 排除密码哈希
                        user_data = {
                            "username": username,
                            "user_id": user_info.get("user_id", ""),
                            "email": user_info.get("email", ""),
                            "created_at": user_info.get("created_at", ""),
                            "last_login": user_info.get("last_login", "")
                        }
                        users_list.append(user_data)
        except Exception as e:
            print(f"[AdminManager] 获取用户列表失败: {e}")
        
        return sorted(users_list, key=lambda x: x.get("created_at", ""), reverse=True)
    
    def get_system_statistics(self) -> Dict:
        """获取系统统计信息"""
        stats = {
            "total_users": 0,
            "total_sessions": 0,
            "total_messages": 0,
            "data_size_mb": 0
        }
        
        try:
            # 统计用户数
            if Path(self.users_file).exists():
                with open(self.users_file, 'r', encoding='utf-8') as f:
                    users = json.load(f)
                    stats["total_users"] = len(users)
            
            # 统计会话数和消息数
            users_dir = Path(self.users_dir)
            if users_dir.exists():
                for user_dir in users_dir.iterdir():
                    if user_dir.is_dir():
                        chat_data_dir = user_dir / "chat_data"
                        if chat_data_dir.exists():
                            for session_file in chat_data_dir.glob("*.json"):
                                stats["total_sessions"] += 1
                                try:
                                    with open(session_file, 'r', encoding='utf-8') as f:
                                        session_data = json.load(f)
                                        stats["total_messages"] += len(session_data.get("messages", []))
                                except:
                                    continue
            
            # 统计数据大小
            data_dir = Path("./rag_falv_data")
            if data_dir.exists():
                total_size = sum(f.stat().st_size for f in data_dir.rglob("*") if f.is_file())
                stats["data_size_mb"] = round(total_size / (1024 * 1024), 2)
        
        except Exception as e:
            print(f"[AdminManager] 获取系统统计失败: {e}")
        
        return stats
    
    def get_user_sessions(self, user_id: str) -> List[Dict]:
        """获取用户的所有会话"""
        sessions = []
        try:
            chat_dir = Path(f"{self.users_dir}/{user_id}/chat_data")
            if chat_dir.exists():
                for session_file in chat_dir.glob("*.json"):
                    try:
                        with open(session_file, 'r', encoding='utf-8') as f:
                            session_data = json.load(f)
                            sessions.append({
                                "session_id": session_data.get("session_id", session_file.stem),
                                "title": session_data.get("title", "未命名会话"),
                                "created_at": session_data.get("created_at", ""),
                                "updated_at": session_data.get("updated_at", ""),
                                "message_count": len(session_data.get("messages", []))
                            })
                    except Exception as e:
                        print(f"[AdminManager] 读取会话文件失败 {session_file}: {e}")
                        continue
        except Exception as e:
            print(f"[AdminManager] 获取用户会话失败: {e}")
        
        return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)
    
    def get_session_messages(self, user_id: str, session_id: str) -> List[Dict]:
        """获取会话的所有消息"""
        messages = []
        try:
            session_file = Path(f"{self.users_dir}/{user_id}/chat_data/{session_id}.json")
            if session_file.exists():
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                    for msg in session_data.get("messages", []):
                        messages.append({
                            "role": msg.get("role", ""),
                            "content": msg.get("content", ""),
                            "timestamp": msg.get("timestamp", "")
                        })
        except Exception as e:
            print(f"[AdminManager] 获取会话消息失败: {e}")
        
        return messages
    
    def delete_user(self, username: str, user_id: str) -> bool:
        """删除用户及其所有数据"""
        try:
            # 从 users.json 中删除用户
            if Path(self.users_file).exists():
                with open(self.users_file, 'r', encoding='utf-8') as f:
                    users = json.load(f)
                
                if username in users:
                    del users[username]
                    
                    with open(self.users_file, 'w', encoding='utf-8') as f:
                        json.dump(users, f, ensure_ascii=False, indent=2)
            
            # 删除用户数据目录
            user_dir = Path(f"{self.users_dir}/{user_id}")
            if user_dir.exists():
                shutil.rmtree(user_dir)
            
            return True
        except Exception as e:
            print(f"[AdminManager] 删除用户失败: {e}")
            return False


def render_admin_panel():
    """渲染管理员面板主界面"""
    st.set_page_config(page_title="管理员面板", page_icon="🔐", layout="wide")
    
    # 侧边栏导航
    with st.sidebar:
        st.title("🔐 管理员面板")
        st.divider()
        
        menu = st.radio(
            "导航菜单",
            ["📊 系统概览", "👥 用户管理", "💬 会话查看", "🚪 退出登录"],
            key="admin_menu"
        )
        
        st.divider()
        st.caption(f"登录用户: {st.session_state.get('username', 'root')}")
    
    # 主内容区域
    if menu == "📊 系统概览":
        render_dashboard()
    elif menu == "👥 用户管理":
        render_user_list()
    elif menu == "💬 会话查看":
        render_session_viewer()
    elif menu == "🚪 退出登录":
        if st.button("确认退出", type="primary"):
            # 清除管理员会话
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


def render_dashboard():
    """渲染系统概览仪表板"""
    st.title("📊 系统概览")
    
    admin_manager = AdminManager()
    stats = admin_manager.get_system_statistics()
    
    # 统计卡片
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("总用户数", stats["total_users"])
    
    with col2:
        st.metric("总会话数", stats["total_sessions"])
    
    with col3:
        st.metric("总消息数", stats["total_messages"])
    
    with col4:
        st.metric("数据大小", f"{stats['data_size_mb']} MB")
    
    st.divider()
    
    # 用户列表预览
    st.subheader("👥 最近注册用户")
    users = admin_manager.get_all_users()
    if users:
        for user in users[:5]:
            with st.expander(f"👤 {user['username']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**用户ID:** {user['user_id']}")
                    st.write(f"**邮箱:** {user['email'] or '未设置'}")
                with col2:
                    st.write(f"**注册时间:** {user['created_at'][:16] if user['created_at'] else '未知'}")
                    st.write(f"**最后登录:** {user['last_login'][:16] if user['last_login'] else '从未登录'}")
    else:
        st.info("暂无用户数据")


def render_user_list():
    """渲染用户列表页面"""
    st.title("👥 用户管理")
    
    admin_manager = AdminManager()
    users = admin_manager.get_all_users()
    
    # 搜索框
    search_query = st.text_input("🔍 搜索用户", placeholder="输入用户名或邮箱")
    
    # 过滤用户
    if search_query:
        users = [u for u in users if search_query.lower() in u['username'].lower() or 
                 search_query.lower() in u.get('email', '').lower()]
    
    st.caption(f"共 {len(users)} 个用户")
    
    # 分页显示
    page_size = 20
    total_pages = (len(users) + page_size - 1) // page_size
    
    if total_pages > 1:
        page = st.number_input("页码", min_value=1, max_value=total_pages, value=1, step=1)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        users_to_display = users[start_idx:end_idx]
    else:
        users_to_display = users
    
    # 显示用户列表
    for user in users_to_display:
        with st.expander(f"👤 {user['username']}"):
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                st.write(f"**用户ID:** {user['user_id']}")
                st.write(f"**邮箱:** {user['email'] or '未设置'}")
            
            with col2:
                st.write(f"**注册时间:** {user['created_at'][:16] if user['created_at'] else '未知'}")
                st.write(f"**最后登录:** {user['last_login'][:16] if user['last_login'] else '从未登录'}")
            
            with col3:
                if st.button("查看会话", key=f"view_{user['user_id']}"):
                    st.session_state.selected_user_id = user['user_id']
                    st.session_state.selected_username = user['username']
                    st.rerun()
                
                if st.button("🗑️ 删除", key=f"delete_{user['user_id']}", type="secondary"):
                    st.session_state.delete_confirm_user = user


def render_session_viewer():
    """渲染会话查看页面"""
    st.title("💬 会话查看")
    
    admin_manager = AdminManager()
    
    # 检查是否选择了用户
    if "selected_user_id" not in st.session_state:
        st.info("请先从用户管理页面选择一个用户")
        return
    
    user_id = st.session_state.selected_user_id
    username = st.session_state.get("selected_username", "未知用户")
    
    st.subheader(f"👤 {username} 的会话记录")
    
    if st.button("← 返回用户列表"):
        del st.session_state.selected_user_id
        if "selected_username" in st.session_state:
            del st.session_state.selected_username
        st.rerun()
    
    # 获取用户会话
    sessions = admin_manager.get_user_sessions(user_id)
    
    if not sessions:
        st.info("该用户暂无会话记录")
        return
    
    st.caption(f"共 {len(sessions)} 个会话")
    
    # 显示会话列表
    for session in sessions:
        with st.expander(f"📝 {session['title']} ({session['message_count']} 条消息)"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**会话ID:** {session['session_id']}")
                st.write(f"**创建时间:** {session['created_at'][:16] if session['created_at'] else '未知'}")
            
            with col2:
                st.write(f"**更新时间:** {session['updated_at'][:16] if session['updated_at'] else '未知'}")
                st.write(f"**消息数量:** {session['message_count']}")
            
            if st.button("查看对话内容", key=f"view_msg_{session['session_id']}"):
                messages = admin_manager.get_session_messages(user_id, session['session_id'])
                
                st.divider()
                st.subheader("对话内容")
                
                for i, msg in enumerate(messages, 1):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    timestamp = msg.get("timestamp", "")
                    
                    if role == "user":
                        st.markdown(f"**👤 用户 ({timestamp[:16]}):**")
                        st.info(content)
                    elif role == "assistant":
                        st.markdown(f"**🤖 助手 ({timestamp[:16]}):**")
                        st.success(content)
                    else:
                        st.markdown(f"**❓ {role} ({timestamp[:16]}):**")
                        st.text(content)


def render_user_detail(user_id: str, username: str):
    """渲染用户详情页面"""
    st.title(f"👤 {username} 的详细信息")
    
    if st.button("← 返回用户列表"):
        if "selected_user_id" in st.session_state:
            del st.session_state.selected_user_id
        if "selected_username" in st.session_state:
            del st.session_state.selected_username
        st.rerun()
    
    admin_manager = AdminManager()
    sessions = admin_manager.get_user_sessions(user_id)
    
    st.subheader("会话列表")
    st.caption(f"共 {len(sessions)} 个会话")
    
    for session in sessions:
        with st.expander(f"📝 {session['title']}"):
            st.write(f"**会话ID:** {session['session_id']}")
            st.write(f"**创建时间:** {session['created_at']}")
            st.write(f"**消息数量:** {session['message_count']}")


def render_session_detail(user_id: str, session_id: str):
    """渲染会话详情页面"""
    st.title("💬 会话详情")
    
    admin_manager = AdminManager()
    messages = admin_manager.get_session_messages(user_id, session_id)
    
    st.caption(f"共 {len(messages)} 条消息")
    
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        timestamp = msg.get("timestamp", "")
        
        if role == "user":
            st.markdown(f"**👤 用户 ({timestamp}):**")
            st.info(content)
        elif role == "assistant":
            st.markdown(f"**🤖 助手 ({timestamp}):**")
            st.success(content)
