#!/usr/bin/env python3
"""
劳动法智能助理 
开发者：罗志远 广东财经大学人工智能法研究中心研究人员
联系方式：1452723426@qq.com
"""

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
        
        .legal-watermark { left: 50%; width: 80vw; } /* 手机端天平居中 */
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
        from labor_law_complete_fixed import app, llm 
        return app, llm
    except ImportError as e:
        return None, None

app, llm = load_backend()

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
    st.session_state.form_data = {"单位名称": "", "平均月薪": "", "时间节点": "", "核心诉求": "", "详细经过": ""}
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'ready_for_analysis' not in st.session_state:
    st.session_state.ready_for_analysis = False
if 'report_generated' not in st.session_state:
    st.session_state.report_generated = False 
if 'ai_mode' not in st.session_state:
    st.session_state.ai_mode = "PRO"

# ==========================================
# 3. 辅助函数
# ==========================================
def extract_info_silently(chat_history, current_data):
    if not chat_history: return current_data
    prompt = f"""你是一个后台数据提取器。请阅读最新的【聊天记录】，提取关键信息并更新【当前数据】。
    没提到的保持空字符串 ""。
    【当前数据】：{json.dumps(current_data, ensure_ascii=False)}
    【最近对话】：{[{'role': 'user' if isinstance(m, HumanMessage) else 'ai', 'content': m.content} for m in chat_history[-4:]]}
    请严格返回包含这5个键的JSON："单位名称", "平均月薪", "时间节点", "核心诉求", "详细经过"。"""
    try:
        response = llm.invoke([SystemMessage(content="只输出合法JSON"), HumanMessage(content=prompt)])
        clean_json = response.content.replace('```json', '').replace('```', '').strip()
        new_data = json.loads(clean_json)
        return {k: new_data.get(k) if new_data.get(k) else current_data.get(k, "") for k in current_data.keys()}
    except Exception:
        return current_data

def strip_markdown(text):
    if not text: return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[*#`>]', '', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def create_pure_pdf_report(form_data, result_dict):
    from fpdf import FPDF
    import os
    pdf = FPDF()
    pdf.add_page()
    
    font_loaded = False
    # 优先加载项目目录下的 msyh.ttf（云端部署用），其次尝试本地系统字体
    font_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "msyh.ttf"),
        "msyh.ttf",
        "C:/Windows/Fonts/msyh.ttf",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                pdf.add_font('msyh', '', path, uni=True)
                font_loaded = True
                break
            except: continue

    def safe_print(title, text, is_title=False):
        if font_loaded:
            pdf.set_font('msyh', '', 16 if is_title else 11)
        else:
            pdf.set_font('Helvetica', '', 16 if is_title else 11)
            
        if is_title:
            pdf.cell(0, 10, title, ln=1, align='C')
            pdf.ln(5)
        else:
            if font_loaded:
                pdf.set_font('msyh', '', 12)
            else:
                pdf.set_font('Helvetica', 'B', 12)
            pdf.cell(0, 8, title, ln=1)
            if font_loaded:
                pdf.set_font('msyh', '', 10)
            else:
                pdf.set_font('Helvetica', '', 10)
            cleaned_text = strip_markdown(text)
            pdf.multi_cell(0, 6, cleaned_text)
            pdf.ln(6)

    safe_print("劳动法律深度分析报告", "", is_title=True)
    info_text = "\n".join([f"• {k}: {v}" for k, v in form_data.items() if v])
    safe_print("案件基本信息", info_text)
    safe_print("一、事实梳理", result_dict.get('legal_facts_summary', '无数据'))
    safe_print("二、法条适用分析", result_dict.get('relevant_laws', '无数据'))
    safe_print("三、合规审查与最终建议", result_dict.get('final_review', '无数据'))
    
    if not font_loaded:
        pdf.set_text_color(255, 0, 0)
        pdf.set_font('Helvetica', '', 10)
        pdf.multi_cell(0, 6, "WARNING: Chinese font not found.")
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
        st.session_state.form_data = {"单位名称": "", "平均月薪": "", "时间节点": "", "核心诉求": "", "详细经过": ""}
        st.session_state.analysis_result = None
        st.session_state.ready_for_analysis = False
        st.session_state.report_generated = False
        st.rerun()

# ==========================================
# 5. 主页面布局
# ==========================================
st.markdown('<h1 class="chat-title">AI 劳动法</h1>', unsafe_allow_html=True)
st.markdown('<p class="chat-subtitle">左侧沟通案情，右侧智能建档。生成报告后对话将自动销毁。</p>', unsafe_allow_html=True)

col_chat, col_panel = st.columns([6, 4], gap="large")

# ------------------------------------------
# 左侧：聊天面板
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

        # 渲染历史聊天
        for msg in st.session_state.messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            with st.chat_message(role):
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
                with st.spinner("AI 正在思考..."):
                    config = {"configurable": {"thread_id": st.session_state.thread_id}}
                    
                    # 🌟 2. 前端静默注入黑科技
                    if st.session_state.ai_mode == "QUICK":
                        injected_prompt = f"【系统前置绝对指令：当前处于快速普法模式。请直接以专业律师口吻回答该问题，绝对不要试图收集案卷要素，也不要提示用户看右侧表单。】\n用户：{prompt}"
                        backend_msg = HumanMessage(content=injected_prompt)
                    else:
                        backend_msg = user_msg_ui
                    
                    # 仅发送构造好的单条消息给后端
                    result = app.invoke({"messages": [backend_msg]}, config)
                    
                    # 提取后端回复上屏
                    latest_messages = result.get("messages", [])
                    if latest_messages:
                        st.session_state.messages.append(latest_messages[-1])
                    
                    # 3. 只有 PRO 模式才会触发右侧卷宗系统的联动
                    if st.session_state.ai_mode == "PRO":
                        action = result.get("triage_result", {}).get("action", "chat")
                        if action == "form":
                            st.session_state.ready_for_analysis = True
                            st.toast("🎯 核心信息已收集完毕！请查看右侧面板。", icon="✅")
                        else:
                            st.session_state.ready_for_analysis = False
                        
                        # 触发静默提取，自动填写右侧表格
                        updated_data = extract_info_silently(st.session_state.messages, st.session_state.form_data)
                        st.session_state.form_data = updated_data
                    else:
                        # 快速模式下，永远不提示收集完毕
                        st.session_state.ready_for_analysis = False
                    
            st.rerun()

# ------------------------------------------
# 右侧：卷宗面板 & 沉浸式报告预览
# ------------------------------------------
with col_panel:
    panel_class = "panel-container highlight-border" if st.session_state.ready_for_analysis else "panel-container"
    st.markdown(f'<div class="{panel_class}">', unsafe_allow_html=True)
    
    if st.session_state.report_generated and st.session_state.analysis_result:
        # --- 纸质感报告预览区 ---
        st.markdown('<div class="panel-header">📄 分析报告预览</div>', unsafe_allow_html=True)
        
        pdf_bytes = create_pure_pdf_report(st.session_state.form_data, st.session_state.analysis_result)
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
            # 🎯 空状态 (Empty State) 优化逻辑
            is_empty = all(v == "" for v in st.session_state.form_data.values())
            
            if st.session_state.ready_for_analysis:
                st.success("✅ AI 认为信息已充足，请核对下方数据并生成报告。")
            elif is_empty:
                st.info("👋 **卷宗目前为空。**\n\n请在左侧向我描述您的案情，我会自动为您提取并填写此处的关键信息。")
            else:
                st.caption("左侧沟通时，AI 会自动为您更新下方信息。您也可随时手动修正。")
            
        with st.form("case_confirmation_form", border=False):
            # 快速模式下锁定所有输入框
            disabled_status = st.session_state.ai_mode == "QUICK"
            
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
                        "单位名称": f_company, "平均月薪": f_salary, "时间节点": f_date, 
                        "核心诉求": f_demand, "详细经过": f_details
                    }
                    st.session_state.form_data = final_form
                    
                    with st.spinner("⚖️ 多智能体正在后台进行法条检索与深度推演，请稍候..."):
                        config = {"configurable": {"thread_id": st.session_state.thread_id}}
                        app.update_state(config, {"form_data": final_form})
                        final_result = app.invoke(None, config)
                        
                        st.session_state.analysis_result = final_result
                        st.session_state.messages = [] 
                        st.session_state.report_generated = True 
                    st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)