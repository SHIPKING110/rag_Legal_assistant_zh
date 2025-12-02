# -*- coding: utf-8 -*-
"""
用户认证模块
提供登录和注册功能
"""
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

import streamlit as st

# 导入管理员模块
from webui.admin import AdminManager


class UserManager:
    """用户管理器"""
    
    USERS_FILE = "./rag_falv_data/users.json"
    
    def __init__(self):
        """初始化用户管理器"""
        self._ensure_file()
    
    def _ensure_file(self):
        """确保用户文件存在"""
        Path(self.USERS_FILE).parent.mkdir(parents=True, exist_ok=True)
        if not Path(self.USERS_FILE).exists():
            self._save_users({})
    
    def _load_users(self) -> Dict:
        """加载用户数据"""
        try:
            with open(self.USERS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[UserManager] 加载用户数据失败: {e}")
            return {}
    
    def _save_users(self, users: Dict) -> bool:
        """保存用户数据"""
        try:
            with open(self.USERS_FILE, 'w', encoding='utf-8') as f:
                json.dump(users, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"[UserManager] 保存用户数据失败: {e}")
            return False
    
    @staticmethod
    def _hash_password(password: str) -> str:
        """密码哈希"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    @staticmethod
    def _generate_user_id(username: str) -> str:
        """生成用户ID"""
        return hashlib.md5(username.encode()).hexdigest()[:16]
    
    def register(self, username: str, password: str, email: str = "") -> tuple[bool, str]:
        """注册新用户
        
        Returns:
            (success, message)
        """
        users = self._load_users()
        
        # 检查用户名是否已存在
        if username in users:
            return False, "用户名已存在"
        
        # 验证用户名和密码
        if len(username) < 3:
            return False, "用户名至少3个字符"
        if len(password) < 6:
            return False, "密码至少6个字符"
        
        # 生成用户ID
        user_id = self._generate_user_id(username)
        
        # 创建新用户
        users[username] = {
            "user_id": user_id,
            "password": self._hash_password(password),
            "email": email,
            "created_at": datetime.now().isoformat(),
            "last_login": None
        }
        
        # 创建用户专属数据目录
        user_data_dir = Path(f"./rag_falv_data/users/{user_id}")
        user_data_dir.mkdir(parents=True, exist_ok=True)
        (user_data_dir / "chat_data").mkdir(exist_ok=True)
        
        if self._save_users(users):
            return True, "注册成功"
        else:
            return False, "注册失败，请稍后重试"
    
    def login(self, username: str, password: str) -> tuple[bool, str, str]:
        """用户登录
        
        Returns:
            (success, message, user_role)
        """
        # 首先检查是否为管理员登录
        if AdminManager.is_admin_credentials(username, password):
            return True, "管理员登录成功", "admin"
        
        # 普通用户登录
        users = self._load_users()
        
        if username not in users:
            return False, "用户名不存在", "user"
        
        user = users[username]
        if user["password"] != self._hash_password(password):
            return False, "密码错误", "user"
        
        # 更新最后登录时间
        user["last_login"] = datetime.now().isoformat()
        self._save_users(users)
        
        return True, "登录成功", "user"
    
    def get_user_info(self, username: str) -> Optional[Dict]:
        """获取用户信息"""
        users = self._load_users()
        return users.get(username)


def check_authentication():
    """检查用户是否已登录"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "user_role" not in st.session_state:
        st.session_state.user_role = "user"
    
    return st.session_state.authenticated


def is_admin():
    """检查当前用户是否为管理员"""
    return st.session_state.get("user_role") == "admin"


def logout():
    """用户登出"""
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.user_role = "user"
    st.rerun()


def render_login_page():
    """渲染登录注册页面"""
    st.markdown("""
        <style>
            .auth-container {
                max-width: 400px;
                margin: 0 auto;
                padding: 2rem;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # 居中显示
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("# ⚖️ 智能法律咨询助手")
        st.markdown("---")
        
        # 选项卡：登录/注册
        tab1, tab2 = st.tabs(["🔐 登录", "📝 注册"])
        
        user_manager = UserManager()
        
        # 登录标签页
        with tab1:
            st.markdown("### 用户登录")
            
            login_username = st.text_input("用户名", key="login_username", placeholder="请输入用户名")
            login_password = st.text_input("密码", type="password", key="login_password", placeholder="请输入密码")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("登录", use_container_width=True, type="primary"):
                    if not login_username or not login_password:
                        st.error("请填写完整信息")
                    else:
                        success, message, user_role = user_manager.login(login_username, login_password)
                        if success:
                            st.session_state.authenticated = True
                            st.session_state.username = login_username
                            st.session_state.user_role = user_role
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
            
            with col_b:
                if st.button("游客登录", use_container_width=True):
                    st.session_state.authenticated = True
                    st.session_state.username = "游客"
                    st.session_state.user_role = "user"
                    st.rerun()
        
        # 注册标签页
        with tab2:
            st.markdown("### 新用户注册")
            
            reg_username = st.text_input("用户名", key="reg_username", placeholder="至少3个字符")
            reg_email = st.text_input("邮箱（可选）", key="reg_email", placeholder="example@email.com")
            reg_password = st.text_input("密码", type="password", key="reg_password", placeholder="至少6个字符")
            reg_password_confirm = st.text_input("确认密码", type="password", key="reg_password_confirm", placeholder="再次输入密码")
            
            if st.button("注册", use_container_width=True, type="primary"):
                if not reg_username or not reg_password:
                    st.error("请填写用户名和密码")
                elif reg_password != reg_password_confirm:
                    st.error("两次密码输入不一致")
                else:
                    success, message = user_manager.register(reg_username, reg_password, reg_email)
                    if success:
                        st.success(message + "，请切换到登录标签页登录")
                    else:
                        st.error(message)
        
        st.markdown("---")
        st.caption("💡 提示：可以使用游客模式快速体验系统功能")


def render_user_info_sidebar():
    """在侧边栏显示用户信息"""
    if check_authentication():
        username = st.session_state.username
        with st.sidebar:
            st.markdown("---")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(f"👤 {username}")
            with col2:
                if st.button("🚪", help="登出"):
                    logout()
