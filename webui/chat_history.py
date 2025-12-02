# -*- coding: utf-8 -*-
"""
会话历史管理模块
处理会话的创建、存储、加载、删除等功能
"""
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st


class UserProfileManager:
    """用户档案管理器 - 存储跨会话的持久化信息"""
    
    PROFILE_FILE = "./rag_falv_data/chat_data/user_profile.json"
    
    def __init__(self, user_id: Optional[str] = None):
        """初始化用户档案管理器
        
        Args:
            user_id: 用户ID，如果提供则使用用户专属目录
        """
        self.user_id = user_id
        if user_id:
            self.PROFILE_FILE = f"./rag_falv_data/users/{user_id}/user_profile.json"
        self._ensure_file()
    
    def _ensure_file(self):
        """确保档案文件存在"""
        Path(self.PROFILE_FILE).parent.mkdir(parents=True, exist_ok=True)
        if not Path(self.PROFILE_FILE).exists():
            self.save_profile({})
    
    def load_profile(self) -> Dict:
        """加载用户档案"""
        try:
            if Path(self.PROFILE_FILE).exists():
                with open(self.PROFILE_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"[UserProfileManager] 加载用户档案失败: {e}")
        return {}
    
    def save_profile(self, profile: Dict) -> bool:
        """保存用户档案"""
        try:
            with open(self.PROFILE_FILE, 'w', encoding='utf-8') as f:
                json.dump(profile, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"[UserProfileManager] 保存用户档案失败: {e}")
            return False
    
    def update_profile(self, key: str, value: str) -> bool:
        """更新用户档案中的某个字段"""
        profile = self.load_profile()
        profile[key] = value
        profile["updated_at"] = datetime.now().isoformat()
        return self.save_profile(profile)
    
    def get_profile_context(self) -> str:
        """获取用户档案的上下文字符串，用于提示词"""
        profile = self.load_profile()
        if not profile:
            return ""
        
        context_parts = []
        if profile.get("name"):
            context_parts.append(f"用户名字: {profile['name']}")
        if profile.get("preferences"):
            context_parts.append(f"用户偏好: {profile['preferences']}")
        if profile.get("notes"):
            context_parts.append(f"备注信息: {profile['notes']}")
        
        # 添加其他自定义字段
        skip_keys = {"name", "preferences", "notes", "updated_at"}
        for key, value in profile.items():
            if key not in skip_keys and value:
                context_parts.append(f"{key}: {value}")
        
        return "\n".join(context_parts) if context_parts else ""
    
    def extract_user_info_from_message(self, message: str, response: str) -> Dict:
        """从对话中提取用户信息（简单的关键词匹配）"""
        extracted = {}
        
        # 提取名字
        name_patterns = [
            r"我(?:的名字)?(?:叫|是|叫做)[\s]*([^\s，。,\.！!？?]+)",
            r"我是[\s]*([^\s，。,\.！!？?]+)",
            r"叫我[\s]*([^\s，。,\.！!？?]+)",
        ]
        import re
        for pattern in name_patterns:
            match = re.search(pattern, message)
            if match:
                name = match.group(1).strip()
                # 过滤掉一些常见的非名字词
                if name and len(name) <= 10 and name not in ["什么", "谁", "哪里", "怎么"]:
                    extracted["name"] = name
                    break
        
        return extracted


class ChatHistoryManager:
    """会话历史管理器"""
    
    CHAT_DATA_DIR = "./rag_falv_data/chat_data"
    
    def __init__(self, user_id: Optional[str] = None):
        """初始化管理器，确保数据目录存在
        
        Args:
            user_id: 用户ID，如果提供则使用用户专属目录
        """
        self.user_id = user_id
        if user_id:
            self.CHAT_DATA_DIR = f"./rag_falv_data/users/{user_id}/chat_data"
        self._ensure_data_dir()
    
    def _ensure_data_dir(self):
        """确保数据目录存在"""
        Path(self.CHAT_DATA_DIR).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def generate_session_id() -> str:
        """生成唯一会话ID，格式: YYYYMMDD_HHMMSS_UUID8"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        return f"{timestamp}_{unique_id}"
    
    def save_session(self, session_id: str, messages: List[Dict], title: Optional[str] = None) -> bool:
        """保存会话到文件
        
        Args:
            session_id: 会话ID
            messages: 消息列表
            title: 会话标题（可选，默认使用第一条用户消息）
        
        Returns:
            bool: 保存是否成功
        """
        # 如果没有消息，不保存
        if not messages or len(messages) == 0:
            return False
        
        try:
            file_path = Path(self.CHAT_DATA_DIR) / f"{session_id}.json"
            
            # 如果文件已存在，读取原有数据保留created_at
            created_at = datetime.now().isoformat()
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                        created_at = existing_data.get('created_at', created_at)
                except:
                    pass
            
            # 生成标题
            if title is None:
                title = self.get_session_preview({"messages": messages})
            
            # 序列化消息，处理不可序列化的对象
            serializable_messages = self._serialize_messages(messages)
            
            session_data = {
                "session_id": session_id,
                "created_at": created_at,
                "updated_at": datetime.now().isoformat(),
                "title": title,
                "messages": serializable_messages
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"[ChatHistoryManager] 保存会话失败: {e}")
            return False
    
    def _serialize_messages(self, messages: List[Dict]) -> List[Dict]:
        """序列化消息列表，移除不可序列化的对象"""
        serializable = []
        for msg in messages:
            serialized_msg = {
                "role": msg.get("role", ""),
                "content": msg.get("content", ""),
                "timestamp": msg.get("timestamp", datetime.now().isoformat())
            }
            # 保留cleaned字段
            if "cleaned" in msg:
                serialized_msg["cleaned"] = msg["cleaned"]
            # 保留think字段
            if "think" in msg:
                serialized_msg["think"] = msg["think"]
            # reference_nodes 包含复杂对象，只保存元数据
            if "reference_nodes" in msg and msg["reference_nodes"]:
                serialized_msg["reference_nodes_meta"] = [
                    {
                        "full_title": node.node.metadata.get("full_title", "") if hasattr(node, 'node') else "",
                        "score": node.score if hasattr(node, 'score') else 0
                    }
                    for node in msg["reference_nodes"]
                    if hasattr(node, 'node')
                ]
            serializable.append(serialized_msg)
        return serializable
    
    def load_session(self, session_id: str) -> Optional[Dict]:
        """加载指定会话
        
        Args:
            session_id: 会话ID
        
        Returns:
            会话数据字典，如果不存在或解析失败返回None
        """
        try:
            file_path = Path(self.CHAT_DATA_DIR) / f"{session_id}.json"
            if not file_path.exists():
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[ChatHistoryManager] 加载会话失败: {e}")
            return None
    
    def delete_session(self, session_id: str) -> bool:
        """删除指定会话
        
        Args:
            session_id: 会话ID
        
        Returns:
            bool: 删除是否成功
        """
        try:
            file_path = Path(self.CHAT_DATA_DIR) / f"{session_id}.json"
            if file_path.exists():
                file_path.unlink()
                return True
            return False
        except Exception as e:
            print(f"[ChatHistoryManager] 删除会话失败: {e}")
            return False
    
    def list_sessions(self) -> List[Dict]:
        """获取所有会话列表，按更新时间倒序排列
        
        Returns:
            会话元数据列表
        """
        sessions = []
        try:
            data_dir = Path(self.CHAT_DATA_DIR)
            if not data_dir.exists():
                return []
            
            for json_file in data_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        sessions.append({
                            "session_id": data.get("session_id", json_file.stem),
                            "title": data.get("title", "未命名会话"),
                            "created_at": data.get("created_at", ""),
                            "updated_at": data.get("updated_at", ""),
                            "message_count": len(data.get("messages", []))
                        })
                except Exception as e:
                    print(f"[ChatHistoryManager] 解析会话文件失败 {json_file}: {e}")
                    continue
            
            # 按更新时间倒序排列
            sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        except Exception as e:
            print(f"[ChatHistoryManager] 列出会话失败: {e}")
        
        return sessions
    
    def get_session_preview(self, session_data: Dict) -> str:
        """获取会话预览文本（第一条用户消息）
        
        Args:
            session_data: 会话数据
        
        Returns:
            预览文本，最多50个字符
        """
        messages = session_data.get("messages", [])
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if len(content) > 50:
                    return content[:47] + "..."
                return content if content else "新会话"
        return "新会话"
    
    def get_most_recent_session(self) -> Optional[str]:
        """获取最近的会话ID
        
        Returns:
            最近会话的session_id，如果没有会话返回None
        """
        sessions = self.list_sessions()
        if sessions:
            return sessions[0].get("session_id")
        return None
    
    def create_new_session(self, current_session_id: Optional[str] = None, 
                           current_messages: Optional[List[Dict]] = None) -> str:
        """创建新会话
        
        Args:
            current_session_id: 当前会话ID（用于保存）
            current_messages: 当前消息列表（用于保存）
        
        Returns:
            新会话的session_id
        """
        # 如果当前会话有消息，先保存
        if current_session_id and current_messages and len(current_messages) > 0:
            self.save_session(current_session_id, current_messages)
        
        # 生成新会话ID
        new_session_id = self.generate_session_id()
        return new_session_id


def get_current_user_id() -> Optional[str]:
    """获取当前登录用户的ID"""
    if st.session_state.get("authenticated") and st.session_state.get("username"):
        username = st.session_state.username
        if username == "游客":
            return None  # 游客使用默认目录
        # 从用户管理器获取user_id
        from webui.auth import UserManager
        user_manager = UserManager()
        user_info = user_manager.get_user_info(username)
        if user_info:
            return user_info.get("user_id")
    return None


def init_session_state_for_chat_history():
    """初始化会话历史相关的 session_state"""
    # 获取当前用户ID
    user_id = get_current_user_id()
    
    # 如果用户切换了，重新初始化管理器
    if "chat_history_manager" not in st.session_state or st.session_state.get("last_user_id") != user_id:
        st.session_state.chat_history_manager = ChatHistoryManager(user_id)
        st.session_state.last_user_id = user_id
        # 清空当前会话，强制重新加载
        if "current_session_id" in st.session_state:
            del st.session_state.current_session_id
    
    if "current_session_id" not in st.session_state:
        # 尝试加载最近的会话
        manager = st.session_state.chat_history_manager
        recent_session_id = manager.get_most_recent_session()
        if recent_session_id:
            st.session_state.current_session_id = recent_session_id
            # 加载会话消息
            session_data = manager.load_session(recent_session_id)
            if session_data:
                st.session_state.messages = session_data.get("messages", [])
        else:
            # 创建新会话
            st.session_state.current_session_id = manager.generate_session_id()
            st.session_state.messages = []


def render_new_session_button():
    """渲染新建会话按钮（放在侧边栏最上面）"""
    init_session_state_for_chat_history()
    manager = st.session_state.chat_history_manager
    
    if st.button("➕ 新建会话", use_container_width=True, key="new_session_btn"):
        # 保存当前会话
        current_id = st.session_state.get("current_session_id")
        current_msgs = st.session_state.get("messages", [])
        
        # 创建新会话
        new_id = manager.create_new_session(current_id, current_msgs)
        st.session_state.current_session_id = new_id
        st.session_state.messages = []
        
        # 关闭其他页面，返回聊天界面
        st.session_state.show_legal_search = False
        st.session_state.show_docs = False
        st.rerun()


def render_chat_history_sidebar():
    """渲染历史会话列表（带内部滚动条）"""
    # 确保初始化
    init_session_state_for_chat_history()
    
    manager = st.session_state.chat_history_manager
    sessions = manager.list_sessions()
    
    # 根据会话数量动态设置滚动样式
    if len(sessions) > 5:
        st.markdown("""
            <style>
                /* 历史会话expander内容区域滚动 */
                section[data-testid="stSidebar"] [data-testid="stExpander"]:first-of-type details > div {
                    max-height: 250px !important;
                    overflow-y: auto !important;
                }
            </style>
        """, unsafe_allow_html=True)
    else:
        # 会话数<=5时，移除滚动限制
        st.markdown("""
            <style>
                section[data-testid="stSidebar"] [data-testid="stExpander"]:first-of-type details > div {
                    max-height: none !important;
                    overflow-y: visible !important;
                }
            </style>
        """, unsafe_allow_html=True)
    
    with st.expander(f"📋 历史会话 ({len(sessions)})", expanded=False):
        if not sessions:
            st.caption("暂无历史会话")
        else:
            for session in sessions:
                session_id = session["session_id"]
                title = session["title"]
                is_current = session_id == st.session_state.get("current_session_id")
                
                # 格式化时间
                try:
                    updated_at = datetime.fromisoformat(session["updated_at"])
                    time_str = updated_at.strftime("%m-%d %H:%M")
                except:
                    time_str = ""
                
                # 截断标题，确保单行显示
                max_title_len = 10
                display_title = title[:max_title_len] + "..." if len(title) > max_title_len else title
                
                # 会话项布局
                col1, col2 = st.columns([0.85, 0.15])
                
                with col1:
                    icon = "📍" if is_current else "💬"
                    btn_text = f"{icon} {display_title}"
                    
                    # 检查是否在其他页面（法律检索或文档页面）
                    in_other_page = st.session_state.get("show_legal_search", False) or st.session_state.get("show_docs", False)
                    
                    # 如果在其他页面，即使是当前会话也允许点击返回
                    should_disable = is_current and not in_other_page
                    
                    if st.button(btn_text, key=f"session_{session_id}", 
                                use_container_width=True,
                                disabled=should_disable):
                        # 保存当前会话
                        current_id = st.session_state.get("current_session_id")
                        current_msgs = st.session_state.get("messages", [])
                        if current_id and current_msgs:
                            manager.save_session(current_id, current_msgs)
                        
                        # 加载选中的会话
                        session_data = manager.load_session(session_id)
                        if session_data:
                            st.session_state.current_session_id = session_id
                            st.session_state.messages = session_data.get("messages", [])
                            
                            # 关闭其他页面，返回聊天界面
                            st.session_state.show_legal_search = False
                            st.session_state.show_docs = False
                            st.rerun()
                
                with col2:
                    if st.button("✕", key=f"del_{session_id}", help="删除"):
                        manager.delete_session(session_id)
                        if is_current:
                            new_id = manager.generate_session_id()
                            st.session_state.current_session_id = new_id
                            st.session_state.messages = []
                        st.rerun()
                
                # 时间显示
                if time_str:
                    st.caption(f"　　{time_str}")
