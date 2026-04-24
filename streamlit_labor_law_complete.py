#!/usr/bin/env python3
"""
劳动法智能助理 
开发者：罗志远 广东财经大学人工智能法研究中心研究人员
联系方式：1452723426@qq.com
"""

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import sys
import os
import uuid
import re
import json
import hashlib
import urllib3
import warnings
from datetime import datetime

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", module="urllib3")
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ==========================================
# 0. 页面与全局高级样式配置 (终极律政视觉版)
# ==========================================
st.set_page_config(
    page_title="AI 劳动法",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* 隐藏默认元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background-color: transparent !important;}
    
    /* 1. 极致的中文字体栈与背景微调（已修复 Emoji 小挂件显示问题） */
    body, [class*="css"], .stTextInput, .stTextArea, .stMarkdown {
        font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Helvetica Neue", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji" !important;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    .stApp {
        background-color: #f8fafc;
    }
    
    /* 标题区域 */
    .chat-title { font-weight: 800; color: #1e293b; margin-bottom: 0rem; font-size: 2.4rem; letter-spacing: -0.03em; }
    .chat-subtitle { color: #64748b; font-size: 1rem; margin-bottom: 1.5rem; font-weight: 400; }
    
    /* 2. 优雅的入场动画 */
    @keyframes slideUpFade {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stChatMessage {
        animation: slideUpFade 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        position: relative;
        z-index: 2; /* 确保聊天内容在水印之上 */
    }
    
    /* 右侧面板 - 增加弥散阴影、现代感圆角和入场动画 */
    .panel-container { 
        background-color: #ffffff; 
        border-radius: 20px; 
        padding: 28px; 
        border: 1px solid #f1f5f9; 
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.04), 0 4px 6px -4px rgba(0, 0, 0, 0.02);
        height: 80vh; 
        overflow-y: auto; 
        animation: slideUpFade 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        position: relative;
        z-index: 2;
    }
    
    .panel-header { 
        color: #0f172a; 
        font-weight: 700; 
        font-size: 1.2rem; 
        margin-bottom: 20px; 
        border-bottom: 2px solid #f1f5f9; 
        padding-bottom: 12px;
    }
    
    /* 激活态的高亮边框 - 改为柔和的蓝色发光 */
    .highlight-border { 
        border: 1px solid #3b82f6 !important; 
        box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.15) !important; 
        transition: all 0.3s ease;
    }
    
    /* 📄 报告预览 A4 纸质感 */
    .report-preview-box {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 40px 30px;
        margin-top: 15px;
        font-size: 0.95rem;
        line-height: 1.8;
        color: #334155;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.01);
    }
    .report-preview-box h4 { color: #1e3a8a; border-bottom: 1px solid #f1f5f9; padding-bottom: 8px; margin-top: 25px; }
    .report-preview-box p { margin-bottom: 15px; }
    
    /* 输入框样式统一 */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea { 
        border-radius: 10px; 
        border: 1px solid #e2e8f0; 
        transition: all 0.2s ease;
    }
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 2px rgba(59,130,246,0.2);
    }
    
    /* ========================================================= */
    /* 核心优化区：按钮与交互控件 */
    /* ========================================================= */
    
    /* 1. 主按钮 (Primary) - 替换默认的刺眼红色为渐变科技蓝 */
    button[kind="primary"] {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.4) !important;
    }

    /* 2. 次级按钮 (Secondary) - 如清空对话按钮，增加精致感 */
    button[kind="secondary"] {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        color: #475569 !important;
        border-radius: 12px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    button[kind="secondary"]:hover {
        border-color: #94a3b8 !important;
        color: #0f172a !important;
        background-color: #f8fafc !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
    }

    /* 3. Popover 悬浮窗按钮 (模式切换/附件) - 改为胶囊样式 */
    div[data-testid="stPopover"] > button {
        border: 1px solid #e2e8f0 !important;
        background-color: #ffffff !important;
        color: #334155 !important;
        font-weight: 600 !important;
        border-radius: 30px !important; /* 胶囊圆角 */
        padding: 6px 20px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stPopover"] > button:hover {
        border-color: #cbd5e1 !important;
        background-color: #f8fafc !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05) !important;
        transform: translateY(-1px) !important;
    }

    /* 4. 底部聊天输入框优化 */
    .stChatInputContainer {
        border-radius: 24px !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.04) !important;
        background-color: white !important;
        padding-left: 10px;
        z-index: 3 !important;
    }
    .stChatInputContainer:focus-within {
        border: 1px solid #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }

    /* ========================================================= */
    /* 品牌记忆点：动态水印与色彩注入 */
    /* ========================================================= */
    
    /* 1. 顶部品牌呼吸线：打破死板的极细渐变线 */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 60%, #f59e0b 100%);
        z-index: 99999;
    }

    /* 2. 专属文本选中色：带有温度的琥珀金 */
    ::selection {
        background-color: rgba(245, 158, 11, 0.2) !important; 
        color: #1e3a8a !important; 
    }

    /* 3. 滚动条隐秘美化：唤醒品牌蓝 */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: #e2e8f0;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #3b82f6 0%, #1e3a8a 100%);
    }

    /* 4. 左侧边栏底色微调：拉开主次空间感 */
    [data-testid="stSidebar"] {
        background-color: #f8fafc !important;
        border-right: 1px solid #f1f5f9;
        z-index: 3;
    }

    /* 5. 律政专属动态背景：沉浸式微动天平水印 */
    .legal-watermark {
        position: fixed;
        top: 45%;
        left: 25%;
        transform: translate(-50%, -50%);
        width: 45vw;
        max-width: 550px;
        z-index: 0;
        pointer-events: none; /* 绝对不能影响用户点击 */
        color: #cbd5e1; /* 极浅的蓝灰色 */
        opacity: 0.1; /* 隐约可见的透明度，不影响观感 */
        animation: floatBalance 12s ease-in-out infinite; /* 12秒缓慢呼吸动画 */
    }
    
    @keyframes floatBalance {
        0% { transform: translate(-50%, -50%) rotate(-1deg) scale(1); opacity: 0.06; }
        50% { transform: translate(-50%, -50%) rotate(1deg) scale(1.03); opacity: 0.12; }
        100% { transform: translate(-50%, -50%) rotate(-1deg) scale(1); opacity: 0.06; }
    }

    /* 6. 升级版毛玻璃开发者标签 */
    .developer-info {
        position: fixed;
        top: 15px;
        right: 25px;
        font-size: 0.8rem;
        color: #94a3b8;
        z-index: 999;
        text-align: right;
        line-height: 1.6;
        background: rgba(255, 255, 255, 0.65) !important;
        backdrop-filter: blur(10px) saturate(150%);
        -webkit-backdrop-filter: blur(10px) saturate(150%);
        padding: 5px 15px;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.4) !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    .developer-info:hover {
        background: rgba(255, 255, 255, 0.9) !important;
        transform: translateY(-2px);
    }
    .developer-info a { color: #64748b; text-decoration: none; font-weight: 500; }
    .developer-info a:hover { color: #3b82f6; }

    /* ========================================================= */
    /* 移动端适配 (响应式) */
    /* ========================================================= */
    @media (max-width: 768px) {
        .chat-title { font-size: 1.8rem; }
        
        .panel-container { 
            height: auto !important; 
            max-height: none !important;
            min-height: 50vh;
            margin-top: 20px;
        }
        
        div[data-testid="stPopover"] > button {
            padding: 4px 12px !important; 
            font-size: 0.9rem !important;
        }
        
        .legal-watermark { left: 50%; width: 80vw; }
        
        /* 核心：强制 st.columns 在移动端纵向堆叠 */
        div[data-testid="stColumn"] {
            width: 100% !important;
            flex: 0 0 100% !important;
            max-width: 100% !important;
        }
        div[data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
            gap: 1rem !important;
        }
        
        /* 手机端隐藏开发者信息 */
        .developer-info { display: none !important; }
        
        /* 手机端隐藏天平水印 */
        .legal-watermark { display: none !important; }
        
        .report-preview-box { padding: 20px 15px; }
        
        button[kind="primary"], button[kind="secondary"] {
            min-height: 44px !important;
            font-size: 1rem !important;
        }
    }
    
    @media (max-width: 480px) {
        .chat-title { font-size: 1.4rem; }
        .chat-subtitle { font-size: 0.85rem; }
        h1 { font-size: 1.5rem !important; }
        .panel-container { padding: 16px; border-radius: 14px; }
        .stChatInputContainer { border-radius: 18px !important; }
    }
</style>

<div class="legal-watermark">
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="0.8" stroke-linecap="round" stroke-linejoin="round">
        <line x1="12" y1="3" x2="12" y2="21"></line>
        <line x1="9" y1="21" x2="15" y2="21"></line>
        <path d="M5 9l-3 7c0 1.5 1.5 3 3 3s3-1.5 3-3l-3-7z"></path>
        <path d="M19 9l-3 7c0 1.5 1.5 3 3 3s3-1.5 3-3l-3-7z"></path>
        <line x1="12" y1="3" x2="5" y2="9"></line>
        <line x1="12" y1="3" x2="19" y2="9"></line>
    </svg>
</div>

<div class="developer-info">
    开发者：罗志远<br>
    广东财经大学 人工智能法研究中心<br>
    <a href="mailto:1452723426@qq.com">1452723426@qq.com</a>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 0.5 登录系统
# ==========================================
USERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.json")

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def load_users() -> dict:
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    # 默认管理员账号
    default_users = {
        "lzy": {
            "password": hash_password("123456"),
            "name": "罗志远",
            "role": "admin"
        }
    }
    save_users(default_users)
    return default_users

def save_users(users: dict):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

def authenticate(username: str, password: str) -> bool:
    users = load_users()
    if username not in users:
        return False
    return users[username]["password"] == hash_password(password)

def create_user(username: str, password: str, name: str, role: str = "user") -> bool:
    users = load_users()
    if username in users:
        return False
    users[username] = {
        "password": hash_password(password),
        "name": name,
        "role": role
    }
    save_users(users)
    return True

def delete_user(username: str) -> bool:
    users = load_users()
    if username not in users:
        return False
    del users[username]
    save_users(users)
    return True

# --- 登录界面 ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'login_tab' not in st.session_state:
    st.session_state.login_tab = "login"

if not st.session_state.authenticated:
    # 登录页居中布局
    _, login_col, _ = st.columns([3, 4, 3])
    
    with login_col:
        st.markdown("""
        <div style="text-align: center; padding: 40px 0 20px 0;">
            <div style="font-size: 3rem; margin-bottom: 10px;">⚖️</div>
            <h1 style="font-weight: 800; color: #1e293b; font-size: 2rem; letter-spacing: -0.03em; margin: 0;">AI 劳动法</h1>
            <p style="color: #64748b; font-size: 0.95rem; margin-top: 8px;">请登录以使用系统</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab_login, tab_register = st.tabs(["🔑 登录", "📝 注册"])
        
        with tab_login:
            with st.form("login_form"):
                username = st.text_input("用户名", placeholder="请输入用户名")
                password = st.text_input("密码", type="password", placeholder="请输入密码")
                submitted = st.form_submit_button("登 录", type="primary", use_container_width=True)
                
                if submitted:
                    if not username or not password:
                        st.warning("请填写用户名和密码")
                    elif authenticate(username, password):
                        st.session_state.authenticated = True
                        st.session_state.current_user = username
                        st.rerun()
                    else:
                        st.error("用户名或密码错误")
        
        with tab_register:
            with st.form("register_form"):
                new_user = st.text_input("设置用户名", placeholder="英文或数字")
                new_name = st.text_input("您的姓名", placeholder="真实姓名")
                new_pwd = st.text_input("设置密码", type="password", placeholder="至少6位")
                new_pwd2 = st.text_input("确认密码", type="password", placeholder="再次输入密码")
                reg_submitted = st.form_submit_button("注 册", type="primary", use_container_width=True)
                
                if reg_submitted:
                    if not new_user or not new_pwd or not new_name:
                        st.warning("请填写所有字段")
                    elif len(new_pwd) < 6:
                        st.warning("密码至少6位")
                    elif new_pwd != new_pwd2:
                        st.warning("两次密码不一致")
                    elif create_user(new_user, new_pwd, new_name):
                        st.success("注册成功！请切换到登录页登录")
                    else:
                        st.warning("该用户名已存在")
        

    st.stop()

# --- 顶部用户信息栏 ---
current_users = load_users()
user_info = current_users.get(st.session_state.current_user, {})
user_display_name = user_info.get("name", st.session_state.current_user)
user_role = user_info.get("role", "user")

st.markdown(f"""
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; padding: 0 5px;">
    <span></span>
    <span style="color: #64748b; font-size: 0.85rem;">
        👤 {user_display_name}
        {"🔒 管理员" if user_role == "admin" else ""}
        &nbsp;|&nbsp;
        <a href="#" style="color: #ef4444; text-decoration: none; font-weight: 500;" id="logout-link">退出登录</a>
    </span>
</div>
""", unsafe_allow_html=True)

# 退出登录按钮（用 Streamlit 原生实现，放在侧边栏顶部）
with st.sidebar:
    if st.button("🚪 退出登录", use_container_width=True, type="secondary"):
        st.session_state.authenticated = False
        st.session_state.current_user = None
        st.rerun()

# --- 管理员用户管理面板 ---
if user_role == "admin":
    with st.sidebar:
        with st.expander("👥 用户管理 (管理员)"):
            users_data = load_users()
            st.caption(f"当前共 {len(users_data)} 个用户")
            for uname, udata in users_data.items():
                col1, col2 = st.columns([3, 1])
                with col1:
                    role_tag = "🔒" if udata.get("role") == "admin" else "👤"
                    st.text(f"{role_tag} {uname} ({udata.get('name', '')})")
                with col2:
                    if uname != "lzy" and uname != st.session_state.current_user:
                        if st.button("🗑️", key=f"del_{uname}"):
                            delete_user(uname)
                            st.rerun()

# ==========================================
# 1. 导入 LangGraph 后端与 LLM
# ==========================================
@st.cache_resource
def load_backend():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(current_dir)
        from labor_law_complete_fixed import app, llm, llm_fast
        return app, llm, llm_fast
    except ImportError as e:
        return None, None, None

app, llm, llm_fast = load_backend()

if not app:
    st.error("后端引擎导入失败！请确保 `labor_law_complete_fixed.py` 在同一目录下。")
    st.stop()

# ==========================================
# 2. 状态初始化
# ==========================================
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = f"user-{uuid.uuid4().hex[:8]}"
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'form_data' not in st.session_state:
    st.session_state.form_data = {"案件发生地": "", "单位名称": "", "平均月薪": "", "时间节点": "", "核心诉求": "", "详细经过": ""}
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'ready_for_analysis' not in st.session_state:
    st.session_state.ready_for_analysis = False
if 'report_generated' not in st.session_state:
    st.session_state.report_generated = False 
if 'ai_mode' not in st.session_state:
    st.session_state.ai_mode = "PRO"
if 'context_round_count' not in st.session_state:
    st.session_state.context_round_count = 0

# ==========================================
# 3. 辅助函数
# ==========================================
def parse_ai_message(text):
    """解析包含 <thinking> 标签的AI回复，彻底剥离思考过程与最终输出，并强力清洗 JSON 外泄"""
    thinking = ""
    output = text
    
    # 1. 提取思考过程
    if "<thinking>" in text:
        parts = text.split("</thinking>")
        thinking = parts[0].replace("<thinking>", "").strip()
        output = parts[1].strip() if len(parts) > 1 else ""
    
    # 2. 强力清洗输出正文中的 JSON 外壳
    output = output.strip()
    json_match = re.search(r'"reply"\s*:\s*"([^"]+)"', output)
    if json_match:
        output = json_match.group(1).replace('\\n', '\n')
    else:
        output = re.sub(r'^```json\s*', '', output)
        output = re.sub(r'```$', '', output).strip()
        output = re.sub(r'^\{[\s\S]*?\}\s*', '', output).strip()

    return thinking, output

def extract_info_silently(chat_history, current_data):
    """Context 工程：静默提取信息到右侧卷宗，扩大提取范围"""
    if not chat_history: return current_data
    
    # 只看最近4条用户消息（提速），因为之前的已经被提取过了
    recent_user_msgs = [m for m in chat_history[-6:] if isinstance(m, HumanMessage)]
    if not recent_user_msgs: return current_data
    
    prompt = f"""你是一个后台数据提取器。请从【用户消息】中提取关键信息并更新【当前数据】。
    没提到的保持空字符串 ""。
    
    ⚠️ 提取规则：
    1. 案件发生地：提取具体的城市或省份名称
    2. 单位名称：提取公司/单位全称
    3. 平均月薪：提取具体数字
    4. 时间节点：提取关键时间点
    5. 核心诉求：简练概括用户想要什么
    6. 详细经过：整理为150字以内摘要
    
    【当前数据】：{json.dumps(current_data, ensure_ascii=False)}
    【用户消息】：{[m.content for m in recent_user_msgs[-3:]]}
    请严格返回包含这6个键的JSON："案件发生地", "单位名称", "平均月薪", "时间节点", "核心诉求", "详细经过"。"""
    try:
        response = llm_fast.invoke([SystemMessage(content="只输出合法JSON"), HumanMessage(content=prompt)])
        clean_json = response.content.replace('```json', '').replace('```', '').strip()
        new_data = json.loads(clean_json)
        return {k: new_data.get(k) if new_data.get(k) else current_data.get(k, "") for k in current_data.keys()}
    except Exception:
        return current_data

def evaluate_form_completeness(form_data):
    """Context 工程：评估卷宗信息完善度，返回完善百分比和建议"""
    if not form_data:
        return 0, [], "卷宗为空，请开始描述案情"
    
    total = len(form_data)
    filled = {k: v for k, v in form_data.items() if v and v.strip()}
    filled_count = len(filled)
    missing = [k for k, v in form_data.items() if not v or not v.strip()]
    pct = int(filled_count / total * 100) if total > 0 else 0
    
    # 核心字段权重更高
    core_fields = ["核心诉求", "详细经过"]
    core_filled = sum(1 for f in core_fields if form_data.get(f, "").strip())
    
    if pct >= 80:
        suggestion = "信息充分，可以生成报告"
    elif core_filled >= 1 and pct >= 50:
        suggestion = f"核心信息已有，建议补充：{'、'.join(missing[:2])}"
    elif core_filled == 0:
        suggestion = "缺少核心诉求和案情经过，请继续描述"
    else:
        suggestion = f"还需补充：{'、'.join(missing)}"
    
    return pct, missing, suggestion

def strip_markdown(text):
    if not text: return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[*#`>]', '', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def create_markdown_report(form_data, result_dict):
    """生成 Markdown 格式的分析报告"""
    now = datetime.now().strftime('%Y年%m月%d日')
    
    md = f"""# 劳动法律深度分析报告

> 生成日期：{now}

---

## 案件基本信息

"""
    for k, v in form_data.items():
        if v:
            md += f"- **{k}**：{v}\n"
    
    md += f"""

---

## 一、事实梳理

{result_dict.get('legal_facts_summary', '无数据')}

---

## 二、法条适用分析

{result_dict.get('relevant_laws', '无数据')}

---

## 三、合规审查与最终建议

{result_dict.get('final_review', '无数据')}

---

*本报告由 AI 劳动法智能助理自动生成，仅供参考，不构成法律意见。*
"""
    return md

def create_pdf_report(form_data, result_dict):
    """Markdown → 格式化 PDF（使用 fpdf2，手动解析 Markdown 结构）"""
    from fpdf import FPDF
    import re

    md_content = create_markdown_report(form_data, result_dict)

    # 查找中文字体
    font_path = None
    ff = 'Helvetica'
    for p in [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "simhei.ttf"),
        "simhei.ttf",
        "C:/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/truetype/simhei.ttf",
    ]:
        if os.path.exists(p):
            font_path = p
            break

    pdf = FPDF()
    if font_path:
        pdf.add_font('SimHei', '', font_path, uni=True)
        pdf.add_font('SimHei', 'B', font_path, uni=True)
        ff = 'SimHei'
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)

    # 解析 Markdown 表格为二维列表
    def parse_md_table(lines):
        rows = []
        for line in lines:
            line = line.strip()
            if not line.startswith('|') or not line.endswith('|'):
                continue
            cells = [c.strip() for c in line.split('|')[1:-1]]
            if all(set(c) <= set('- :') for c in cells):
                continue  # 分隔行跳过
            rows.append(cells)
        return rows

    # 逐行解析 Markdown
    lines = md_content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]

        # 空行
        if not line.strip():
            i += 1
            continue

        # 水平线
        if line.strip() in ('---', '***', '___'):
            pdf.ln(4)
            y = pdf.get_y()
            pdf.set_draw_color(200, 200, 200)
            pdf.line(10, y, 200, y)
            pdf.ln(6)
            i += 1
            continue

        # 标题
        m = re.match(r'^(#{1,6})\s+(.*)', line)
        if m:
            level = len(m.group(1))
            text = m.group(2).strip()
            # 去掉 Markdown 加粗标记
            text = text.replace('**', '')
            if level == 1:
                pdf.set_font(ff, 'B', 18)
                pdf.set_text_color(30, 58, 138)
                pdf.cell(0, 12, text, new_x="LMARGIN", new_y="NEXT", align='C')
                y = pdf.get_y()
                pdf.set_draw_color(30, 58, 138)
                pdf.set_line_width(0.8)
                pdf.line(10, y, 200, y)
                pdf.set_line_width(0.2)
                pdf.ln(6)
            elif level == 2:
                pdf.ln(3)
                pdf.set_font(ff, 'B', 14)
                pdf.set_text_color(30, 58, 138)
                pdf.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT")
                y = pdf.get_y()
                pdf.set_draw_color(220, 220, 230)
                pdf.line(10, y, 200, y)
                pdf.ln(4)
            else:
                pdf.ln(2)
                pdf.set_font(ff, 'B', 12)
                pdf.set_text_color(51, 65, 85)
                pdf.cell(0, 9, text, new_x="LMARGIN", new_y="NEXT")
                pdf.ln(2)
            pdf.set_text_color(0, 0, 0)
            i += 1
            continue

        # 引用块
        if line.strip().startswith('>'):
            text = line.strip().lstrip('>').strip()
            text = text.replace('**', '')
            pdf.set_font(ff, '', 10)
            pdf.set_text_color(100, 116, 139)
            x = pdf.get_x()
            y = pdf.get_y()
            pdf.set_draw_color(59, 130, 246)
            pdf.set_line_width(1.0)
            pdf.line(12, y, 12, y + 7)
            pdf.set_line_width(0.2)
            pdf.set_x(18)
            pdf.multi_cell(172, 7, text)
            pdf.set_text_color(0, 0, 0)
            i += 1
            continue

        # 表格
        if line.strip().startswith('|'):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i])
                i += 1
            rows = parse_md_table(table_lines)
            if rows:
                n_cols = len(rows[0])
                col_w = 190 / max(n_cols, 1)
                for ri, row in enumerate(rows):
                    if ri == 0:
                        pdf.set_font(ff, 'B', 10)
                        pdf.set_fill_color(30, 58, 138)
                        pdf.set_text_color(255, 255, 255)
                    else:
                        pdf.set_font(ff, '', 9)
                        pdf.set_text_color(0, 0, 0)
                        fill = ri % 2 == 0
                        if fill:
                            pdf.set_fill_color(248, 250, 252)
                        else:
                            pdf.set_fill_color(255, 255, 255)
                    for ci in range(min(len(row), n_cols)):
                        cell_text = row[ci].replace('**', '')
                        pdf.cell(col_w, 8, cell_text, border=1, fill=True)
                    pdf.ln()
                pdf.set_text_color(0, 0, 0)
            continue

        # 列表项
        list_m = re.match(r'^(\s*)([-*]|\d+\.)\s+(.*)', line)
        if list_m:
            indent = len(list_m.group(1))
            text = list_m.group(3)
            bullet = '•' if list_m.group(2) in ('-', '*') else list_m.group(2)
            # 处理加粗
            text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
            pdf.set_font(ff, '', 10)
            pdf.set_text_color(0, 0, 0)
            x_start = 14 + (indent // 2) * 6
            pdf.set_x(x_start)
            pdf.cell(6, 7, bullet)
            pdf.multi_cell(190 - x_start - 6, 7, text)
            i += 1
            continue

        # 普通段落
        text = line.strip()
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # 去加粗
        text = re.sub(r'\*(.*?)\*', r'\1', text)       # 去斜体
        text = re.sub(r'`([^`]+)`', r'\1', text)       # 去行内代码
        pdf.set_font(ff, '', 10)
        pdf.set_text_color(30, 41, 59)
        pdf.multi_cell(0, 7, text)
        pdf.ln(1)

        i += 1

    pdf.set_text_color(0, 0, 0)
    return bytes(pdf.output())

# ==========================================
# 4. 左侧边栏
# ==========================================
with st.sidebar:
    st.markdown("### 🗂️ 卷宗管理")
    st.caption(f"当前案件编号: `{st.session_state.thread_id}`")
    st.markdown("---")
    
    if st.button("🧹 清空当前屏幕对话", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.rerun()
        
    if st.button("🔄 彻底重置并开启新案", type="primary", use_container_width=True):
        st.session_state.thread_id = f"user-{uuid.uuid4().hex[:8]}"
        st.session_state.messages = []
        st.session_state.form_data = {"案件发生地": "", "单位名称": "", "平均月薪": "", "时间节点": "", "核心诉求": "", "详细经过": ""}
        st.session_state.analysis_result = None
        st.session_state.ready_for_analysis = False
        st.session_state.report_generated = False
        st.session_state.context_round_count = 0
        st.session_state.cached_pdf_bytes = None
        st.session_state.pdf_cache_key = None
        st.rerun()

# ==========================================
# 5. 主页面布局：根据模式动态切换右侧栏显示
# ==========================================
st.markdown('<h1 class="chat-title">AI 劳动法</h1>', unsafe_allow_html=True)
st.markdown('<p class="chat-subtitle">左侧沟通案情，右侧智能建档。生成报告后对话将自动销毁。</p>', unsafe_allow_html=True)

# 核心优化：普法模式隐藏右侧面板，让对话框拉满全宽
if st.session_state.ai_mode == "PRO":
    col_chat, col_panel = st.columns([6, 4], gap="large")
else:
    col_chat = st.container()
    col_panel = None

# ------------------------------------------
# 左侧/主体：聊天面板
# ------------------------------------------
with col_chat:

    if st.session_state.report_generated:
        st.success("✅ 深度分析已完成！出于隐私保护，对话记录已自动焚毁。")
        st.info("👉 请在右侧面板预览并下载您的分析报告。")
    else:
        if not st.session_state.messages:
            with st.chat_message("assistant"):
                if st.session_state.ai_mode == "PRO":
                    st.write("您好！我是您的 **案情推演法律助手**。请详细告诉我您的遭遇，我将为您建立卷宗并出具报告。")
                else:
                    st.write("您好！我是您的 **快速普法助手**。有关劳动法的任何法规疑问，我都会快速为您解答。")

        # 渲染历史聊天 (搭载豆包式思考折叠)
        for msg in st.session_state.messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            with st.chat_message(role):
                if role == "assistant":
                    thinking, output = parse_ai_message(msg.content)
                    if thinking:
                        # 历史消息中的思考默认折叠
                        with st.expander("✅ 已完成思考", expanded=False):
                            st.markdown(thinking)
                    if output:
                        st.markdown(output)
                else:
                    st.write(msg.content)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 🌟 工具栏布局：模式切换悬浮窗 & 附件上传
        tool_col1, tool_col2, _ = st.columns([2.5, 2, 5])
        
        with tool_col1:
            mode_label = "⚡ 普法模式 ⌄" if st.session_state.ai_mode == "QUICK" else "💼 案件模式 ⌄"
            with st.popover(mode_label, use_container_width=True):
                st.markdown("**AI 模型选择**")
                selected_mode = st.radio(
                    "Mode",
                    options=["⚡ 普法模式", "💼 案件模式"],
                    captions=["快速回答常见普法问题", "使用多智能体进行深度案情推演"],
                    index=0 if st.session_state.ai_mode == "QUICK" else 1,
                    label_visibility="collapsed"
                )
                new_mode = "QUICK" if "普法" in selected_mode else "PRO"
                if new_mode != st.session_state.ai_mode:
                    st.session_state.ai_mode = new_mode
                    st.rerun()
                    
        with tool_col2:
            with st.popover("📎 附件", use_container_width=True):
                st.file_uploader("上传劳动合同、打卡记录等", type=["pdf", "png", "jpg"], accept_multiple_files=True)
                st.caption("视觉解析能力即将上线...")

        # 聊天输入框及提交逻辑
        if prompt := st.chat_input("描述您的遭遇或提出疑问...", key="main_chat_input"):
            
            # 1. 前端 UI 展示用户的原话
            user_msg_ui = HumanMessage(content=prompt)
            st.session_state.messages.append(user_msg_ui)
            
            with st.chat_message("user"):
                st.write(prompt)
                
            with st.chat_message("assistant"):
                # 1. 明确分离两个占位容器：思考状态框 + 正文框
                thinking_status = st.status("⚖️ AI 正在思考推演...", expanded=False)
                thinking_placeholder = thinking_status.empty()
                response_placeholder = st.empty()
                
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                
                # 🌟 2. 前端静默注入黑科技
                if st.session_state.ai_mode == "QUICK":
                    injected_prompt = f"【系统前置绝对指令：当前处于快速普法模式。请直接以专业律师口吻回答该问题，绝对不要试图收集案卷要素，也不要提示用户看右侧表单。】\n用户：{prompt}"
                    backend_msg = HumanMessage(content=injected_prompt)
                else:
                    backend_msg = user_msg_ui
                
                full_response = ""
                thinking_text = ""
                
                try:
                    # 使用流式调用
                    for event in app.stream({"messages": [backend_msg]}, config, stream_mode="messages"):
                        if isinstance(event, tuple):
                            msg, _ = event
                        else:
                            msg = event
                        
                        if hasattr(msg, 'content') and msg.content:
                            full_response += msg.content
                            thinking_text, clean_response = parse_ai_message(full_response)
                            
                            # 动态更新思考框
                            if thinking_text:
                                thinking_placeholder.markdown(thinking_text)
                            
                            # 动态更新外部的正文框
                            if clean_response:
                                response_placeholder.markdown(clean_response)
                    
                    # 流式结束后，关闭思考状态框
                    thinking_status.update(label="✅ 已完成思考", state="complete", expanded=False)
                    
                    # 保存完整消息（包含thinking）
                    if full_response.strip():
                        ai_msg = AIMessage(content=full_response.strip())
                        st.session_state.messages.append(ai_msg)
                
                except Exception as e:
                    thinking_status.update(label="⚠️ 出错了", state="error", expanded=False)
                    err_msg = "抱歉，服务暂时不可用，请稍后再试。"
                    response_placeholder.markdown(err_msg)
                    st.session_state.messages.append(AIMessage(content=err_msg))
                
                # 3. 只有 PRO 模式才会触发右侧卷宗系统的联动
                if st.session_state.ai_mode == "PRO":
                    # Context 工程：对话轮数计数
                    st.session_state.context_round_count += 1
                    round_count = st.session_state.context_round_count
                    
                    # 获取最终状态
                    final_state = app.get_state(config)
                    action = final_state.values.get("triage_result", {}).get("action", "chat")
                    
                    # 触发静默提取，自动填写右侧表格
                    updated_data = extract_info_silently(st.session_state.messages, st.session_state.form_data)
                    st.session_state.form_data = updated_data
                    
                    # Context 工程：评估信息完善度
                    completeness_pct, missing_fields, suggestion = evaluate_form_completeness(st.session_state.form_data)
                    
                    # 判断是否收集完毕（三重判断：AI 判断 + 完善度 + 轮数）
                    ai_says_ready = action == "form"
                    data_says_ready = completeness_pct >= 60
                    rounds_exceeded = round_count >= 10  # 超过10轮强制提示
                    
                    if ai_says_ready or data_says_ready:
                        st.session_state.ready_for_analysis = True
                        st.toast("🎯 核心信息已收集完毕！请查看右侧面板并生成报告。", icon="✅")
                    elif rounds_exceeded and completeness_pct >= 40:
                        st.session_state.ready_for_analysis = True
                        st.toast(f"⏰ 已对话 {round_count} 轮，信息完善度 {completeness_pct}%。建议生成报告。", icon="📋")
                    else:
                        st.session_state.ready_for_analysis = False
                        # 每5轮对话提示一下完善度
                        if round_count % 5 == 0 and round_count > 0:
                            st.toast(f"📊 已对话 {round_count} 轮，信息完善度 {completeness_pct}%。{suggestion}", icon="💡")
                else:
                    # 快速模式下，永远不提示收集完毕
                    st.session_state.ready_for_analysis = False
                    
            st.rerun()

# ------------------------------------------
# 右侧：卷宗面板 & 沉浸式报告预览
# ------------------------------------------
if col_panel is not None:
    with col_panel:
        panel_class = "panel-container highlight-border" if st.session_state.ready_for_analysis else "panel-container"
        st.markdown(f'<div class="{panel_class}">', unsafe_allow_html=True)
        
        if st.session_state.report_generated and st.session_state.analysis_result:
            # --- 纸质感报告预览区 --- 
            st.markdown('<div class="panel-header">📄 分析报告预览</div>', unsafe_allow_html=True)
            
            # PDF 缓存：只在首次生成，后续直接读取
            if 'cached_pdf_bytes' not in st.session_state or st.session_state.get('pdf_cache_key') != str(st.session_state.analysis_result):
                pdf_bytes = create_pdf_report(st.session_state.form_data, st.session_state.analysis_result)
                st.session_state.cached_pdf_bytes = pdf_bytes
                st.session_state.pdf_cache_key = str(st.session_state.analysis_result)
            else:
                pdf_bytes = st.session_state.cached_pdf_bytes
            
            st.download_button(
                label="📥 下载 PDF 格式正式报告",
                data=pdf_bytes,
                file_name=f"劳动法分析报告_{datetime.now().strftime('%Y%m%d%H%M')}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )
            
            facts = strip_markdown(st.session_state.analysis_result.get('legal_facts_summary', '无数据'))
            laws = strip_markdown(st.session_state.analysis_result.get('relevant_laws', '无数据'))
            advice = strip_markdown(st.session_state.analysis_result.get('final_review', '无数据'))
            
            preview_html = f"""
            <div class="report-preview-box">
                <h3 style="text-align:center; color:#1e3a8a; margin-bottom: 20px;">劳动法律深度分析报告</h3>
                <h4>一、事实梳理</h4>
                <p>{facts.replace(chr(10), '<br>')}</p>
                <h4>二、法条适用分析</h4>
                <p>{laws.replace(chr(10), '<br>')}</p>
                <h4>三、最终合规建议</h4>
                <p>{advice.replace(chr(10), '<br>')}</p>
            </div>
            """
            st.markdown(preview_html, unsafe_allow_html=True)
            
        else:
            # --- 卷宗收集区 ---
            st.markdown('<div class="panel-header">📑 智能案件卷宗</div>', unsafe_allow_html=True)
            
            if st.session_state.ai_mode == "QUICK":
                st.info("⚡ 当前为**普法模式**，AI 只负责快速解答法律疑问，不收集案件卷宗。如需出具正式案件报告，请在聊天框上方切换至「案件模式」。")
                
            elif st.session_state.ai_mode == "PRO":
                # Context 工程：显示信息完善度指标
                is_empty = all(v == "" for v in st.session_state.form_data.values())
                completeness_pct, missing_fields, suggestion = evaluate_form_completeness(st.session_state.form_data)
                
                if st.session_state.ready_for_analysis:
                    st.success("✅ AI 认为信息已充足，请核对下方数据并生成报告。")
                elif is_empty:
                    st.info("👋 **卷宗目前为空。**\n\n请在左侧向我描述您的案情，我会自动为您提取并填写此处的关键信息。")
                else:
                    # 显示完善度进度条
                    st.markdown(f"""
                    <div style="margin-bottom: 8px;">
                        <span style="font-size: 0.85rem; color: #475569;">信息完善度</span>
                        <span style="float: right; font-size: 0.85rem; font-weight: 600; color: {'#16a34a' if completeness_pct >= 60 else '#ea580c' if completeness_pct >= 30 else '#dc2626'};">{completeness_pct}%</span>
                    </div>
                    <div style="background: #e2e8f0; border-radius: 4px; height: 6px; overflow: hidden;">
                        <div style="background: {'#16a34a' if completeness_pct >= 60 else '#ea580c' if completeness_pct >= 30 else '#dc2626'}; height: 100%; width: {completeness_pct}%; transition: width 0.3s;"></div>
                    </div>
                    <div style="font-size: 0.8rem; color: #64748b; margin-top: 6px;">💡 {suggestion}</div>
                    """, unsafe_allow_html=True)
                    st.caption("左侧沟通时，AI 会自动为您更新下方信息。您也可随时手动修正。")
                
            with st.form("case_confirmation_form", border=False):
                # 快速模式下锁定所有输入框
                disabled_status = st.session_state.ai_mode == "QUICK"
                
                f_region = st.text_input("案件发生地", value=st.session_state.form_data.get("案件发生地", ""), disabled=disabled_status)
                f_company = st.text_input("涉事单位名称", value=st.session_state.form_data.get("单位名称", ""), disabled=disabled_status)
                f_salary = st.text_input("平均月薪", value=st.session_state.form_data.get("平均月薪", ""), disabled=disabled_status)
                f_date = st.text_input("时间节点", value=st.session_state.form_data.get("时间节点", ""), disabled=disabled_status)
                f_demand = st.text_input("核心诉求", value=st.session_state.form_data.get("核心诉求", ""), disabled=disabled_status)
                f_details = st.text_area("详细经过与证据", value=st.session_state.form_data.get("详细经过", ""), height=150, disabled=disabled_status)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                if st.session_state.ai_mode == "PRO":
                    btn_type = "primary" if st.session_state.ready_for_analysis else "secondary"
                    btn_text = "✅ 卷宗确认无误，生成法律分析报告" if st.session_state.ready_for_analysis else "强制跳过收集，直接生成报告"
                    
                    if st.form_submit_button(btn_text, type=btn_type, use_container_width=True):
                        final_form = {
                            "案件发生地": f_region, "单位名称": f_company, "平均月薪": f_salary, 
                            "时间节点": f_date, "核心诉求": f_demand, "详细经过": f_details
                        }
                        st.session_state.form_data = final_form
                        
                        with st.spinner("⚖️ 多智能体正在后台进行法条检索与深度推演，请稍候..."):
                            config = {"configurable": {"thread_id": st.session_state.thread_id}}
                            
                            # 检查当前 LangGraph 状态，判断是否已经过 triage 并中断在 fact_summarizer 之前
                            try:
                                current_state = app.get_state(config)
                                next_steps = current_state.next if hasattr(current_state, 'next') else []
                            except:
                                next_steps = []
                            
                            if next_steps and 'fact_summarizer' in next_steps:
                                # 正常流程：已通过 triage，从中断点恢复
                                app.update_state(config, {"form_data": final_form})
                                final_result = app.invoke(None, config)
                            else:
                                # 强制生成：未经过 triage，直接启动完整分析流程
                                from langchain_core.messages import HumanMessage, SystemMessage
                                case_msg = HumanMessage(content=f"请分析以下劳动法案件：\n" + "\n".join([f"{k}：{v}" for k, v in final_form.items() if v]))
                                # 使用新的 thread_id 避免旧状态干扰
                                new_thread = st.session_state.thread_id + "_direct"
                                config_direct = {"configurable": {"thread_id": new_thread}}
                                # 先触发 triage 让它走到 interrupt_before
                                init_result = app.invoke({"messages": [case_msg], "form_data": final_form}, config_direct)
                                # 检查 triage 结果，如果是 chat 模式则强制改为 form
                                triage_res = init_result.get("triage_result", {})
                                if triage_res.get("action") != "form":
                                    app.update_state(config_direct, {"triage_result": {"action": "form", "category": "强制案件分析", "reply": "开始分析"}})
                                app.update_state(config_direct, {"form_data": final_form})
                                final_result = app.invoke(None, config_direct)
                            
                            # 稳健提取：兼容不同 LangGraph 版本的返回格式
                            if isinstance(final_result, dict):
                                analysis = {
                                    "legal_facts_summary": final_result.get("legal_facts_summary", ""),
                                    "relevant_laws": final_result.get("relevant_laws", ""),
                                    "final_review": final_result.get("final_review", ""),
                                }
                            else:
                                try:
                                    cfg = config if next_steps and 'fact_summarizer' in next_steps else config_direct
                                    state = app.get_state(cfg)
                                    state_values = state.values if hasattr(state, 'values') else {}
                                    analysis = {
                                        "legal_facts_summary": state_values.get("legal_facts_summary", ""),
                                        "relevant_laws": state_values.get("relevant_laws", ""),
                                        "final_review": state_values.get("final_review", ""),
                                    }
                                except:
                                    analysis = {"legal_facts_summary": str(final_result), "relevant_laws": "", "final_review": ""}
                            
                            st.session_state.analysis_result = analysis
                            st.session_state.messages = [] 
                            st.session_state.report_generated = True 
                        st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)