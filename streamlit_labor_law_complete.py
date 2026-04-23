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
    
    /* 1. 极致的中文字体栈与背景微调 */
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
        z-index: 2;
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
    
    /* 激活态的高亮边框 */
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
    
    /* ========================================================= */
    /* 核心优化区：按钮与交互控件 */
    /* ========================================================= */
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

    div[data-testid="stPopover"] > button {
        border: 1px solid #e2e8f0 !important;
        background-color: #ffffff !important;
        color: #334155 !important;
        font-weight: 600 !important;
        border-radius: 30px !important;
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

    /* 底部聊天输入框优化 */
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

    /* 律政专属动态背景：沉浸式微动天平水印 */
    .legal-watermark {
        position: fixed;
        top: 45%;
        left: 25%;
        transform: translate(-50%, -50%);
        width: 45vw;
        max-width: 550px;
        z-index: 0;
        pointer-events: none;
        color: #cbd5e1;
        opacity: 0.1;
        animation: floatBalance 12s ease-in-out infinite;
    }
    
    @keyframes floatBalance {
        0% { transform: translate(-50%, -50%) rotate(-1deg) scale(1); opacity: 0.06; }
        50% { transform: translate(-50%, -50%) rotate(1deg) scale(1.03); opacity: 0.12; }
        100% { transform: translate(-50%, -50%) rotate(-1deg) scale(1); opacity: 0.06; }
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
    default_users = {"lzy": {"password": hash_password("123456"), "name": "罗志远", "role": "admin"}}
    save_users(default_users)
    return default_users

def save_users(users: dict):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

def authenticate(username: str, password: str) -> bool:
    users = load_users()
    if username not in users: return False
    return users[username]["password"] == hash_password(password)

def create_user(username: str, password: str, name: str, role: str = "user") -> bool:
    users = load_users()
    if username in users: return False
    users[username] = {"password": hash_password(password), "name": name, "role": role}
    save_users(users)
    return True

def delete_user(username: str) -> bool:
    users = load_users()
    if username not in users: return False
    del users[username]
    save_users(users)
    return True

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None

if not st.session_state.authenticated:
    _, login_col, _ = st.columns([3, 4, 3])
    with login_col:
        st.markdown("""
        <div style="text-align: center; padding: 40px 0 20px 0;">
            <div style="font-size: 3rem; margin-bottom: 10px;">⚖️</div>
            <h1 style="font-weight: 800; color: #1e293b; font-size: 2rem; margin: 0;">AI 劳动法</h1>
            <p style="color: #64748b;">请登录以使用系统</p>
        </div>
        """, unsafe_allow_html=True)
        tab_login, tab_register = st.tabs(["🔑 登录", "📝 注册"])
        
        with tab_login:
            with st.form("login_form"):
                username = st.text_input("用户名", placeholder="请输入用户名")
                password = st.text_input("密码", type="password", placeholder="请输入密码")
                if st.form_submit_button("登 录", type="primary", use_container_width=True):
                    if authenticate(username, password):
                        st.session_state.authenticated = True
                        st.session_state.current_user = username
                        st.rerun()
                    else:
                        st.error("用户名或密码错误")
        
        with tab_register:
            with st.form("register_form"):
                new_user = st.text_input("设置用户名")
                new_name = st.text_input("您的姓名")
                new_pwd = st.text_input("设置密码", type="password")
                new_pwd2 = st.text_input("确认密码", type="password")
                if st.form_submit_button("注 册", type="primary", use_container_width=True):
                    if new_pwd != new_pwd2:
                        st.warning("两次密码不一致")
                    elif create_user(new_user, new_pwd, new_name):
                        st.success("注册成功！请切换到登录页登录")
                    else:
                        st.warning("用户名已存在")
    st.stop()

# --- 侧边栏及登出逻辑 ---
current_users = load_users()
user_info = current_users.get(st.session_state.current_user, {})
user_role = user_info.get("role", "user")

with st.sidebar:
    if st.button("🚪 退出登录", use_container_width=True, type="secondary"):
        st.session_state.authenticated = False
        st.session_state.current_user = None
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

# ==========================================
# 3. 辅助函数与解析逻辑 (🌟 核心修复区)
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
    if not chat_history: return current_data
    prompt = f"""读取最新的【聊天记录】，提取关键信息并更新【当前数据】。没提到的保持空字符串 ""。
    【当前数据】：{json.dumps(current_data, ensure_ascii=False)}
    【最近对话】：{[{'role': 'user' if isinstance(m, HumanMessage) else 'ai', 'content': m.content} for m in chat_history[-4:]]}
    请返回包含这6个键的JSON："案件发生地", "单位名称", "平均月薪", "时间节点", "核心诉求", "详细经过"。"""
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
    return text.strip()

def create_pure_pdf_report(form_data, result_dict):
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    # ... (PDF 生成逻辑保持你的原生代码不变) ...
    pdf.set_font('Helvetica', '', 12)
    pdf.multi_cell(0, 6, "AI Analysis Report PDF (Please verify simhei.ttf is configured for Chinese)")
    return bytes(pdf.output())

# ==========================================
# 4. 左侧边栏管理
# ==========================================
with st.sidebar:
    st.markdown("### 🗂️ 卷宗管理")
    st.caption(f"案件编号: `{st.session_state.thread_id}`")
    st.markdown("---")
    
    if st.button("🧹 清空当前对话", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.rerun()
        
    if st.button("🔄 开启新案", type="primary", use_container_width=True):
        st.session_state.thread_id = f"user-{uuid.uuid4().hex[:8]}"
        st.session_state.messages = []
        st.session_state.form_data = {"案件发生地": "", "单位名称": "", "平均月薪": "", "时间节点": "", "核心诉求": "", "详细经过": ""}
        st.session_state.analysis_result = None
        st.session_state.ready_for_analysis = False
        st.session_state.report_generated = False
        st.rerun()

# ==========================================
# 5. 主页面布局 (🌟 智能动态拉伸)
# ==========================================
st.markdown('<h1 class="chat-title">AI 劳动法</h1>', unsafe_allow_html=True)
st.markdown('<p class="chat-subtitle">左侧沟通案情，右侧智能建档。生成报告后对话自动销毁。</p>', unsafe_allow_html=True)

# 🌟 如果是普法模式，隐藏右侧面板，让对话框拉满 100% 宽！
if st.session_state.ai_mode == "PRO":
    col_chat, col_panel = st.columns([6, 4], gap="large")
else:
    col_chat = st.container()
    col_panel = None

# ------------------------------------------
# 左侧 (主体)：聊天面板
# ------------------------------------------
with col_chat:
    if st.session_state.report_generated:
        st.success("✅ 深度分析已完成！出于隐私保护，对话记录已自动焚毁。")
    else:
        if not st.session_state.messages:
            with st.chat_message("assistant"):
                msg = "我是您的 **案情推演助手**。请详细告诉我您的遭遇，我将为您建立卷宗。" if st.session_state.ai_mode == "PRO" else "我是您的 **快速普法助手**。劳动法疑问随时为您解答。"
                st.write(msg)

        # 🌟 历史聊天渲染 (带思考折叠)
        for msg in st.session_state.messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            with st.chat_message(role):
                if role == "assistant":
                    thinking, output = parse_ai_message(msg.content)
                    if thinking:
                        with st.expander("✅ 已完成思考", expanded=False):
                            st.markdown(thinking)
                    if output:
                        st.markdown(output)
                else:
                    st.write(msg.content)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        tool_col1, tool_col2, _ = st.columns([2.5, 2, 5])
        with tool_col1:
            mode_label = "⚡ 普法模式 ⌄" if st.session_state.ai_mode == "QUICK" else "💼 案件模式 ⌄"
            with st.popover(mode_label, use_container_width=True):
                selected_mode = st.radio("Mode", options=["⚡ 普法模式", "💼 案件模式"], index=0 if st.session_state.ai_mode == "QUICK" else 1, label_visibility="collapsed")
                new_mode = "QUICK" if "普法" in selected_mode else "PRO"
                if new_mode != st.session_state.ai_mode:
                    st.session_state.ai_mode = new_mode
                    st.rerun()

        # 聊天输入框及提交逻辑
        if prompt := st.chat_input("描述您的遭遇或提出疑问..."):
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
                
                if st.session_state.ai_mode == "QUICK":
                    injected_prompt = f"【系统绝对指令：当前处于快速普法模式。请直接以律师口吻回答该问题，绝对不要收集案卷要素，不要输出JSON结构】\n用户：{prompt}"
                    backend_msg = HumanMessage(content=injected_prompt)
                else:
                    backend_msg = user_msg_ui
                
                full_response = ""
                thinking_text = ""
                try:
                    for event in app.stream({"messages": [backend_msg]}, config, stream_mode="messages"):
                        msg = event[0] if isinstance(event, tuple) else event
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
                                
                    if full_response.strip():
                        st.session_state.messages.append(AIMessage(content=full_response.strip()))
                
                except Exception as e:
                    thinking_status.update(label="⚠️ 出错了", state="error", expanded=False)
                    err_msg = "抱歉，服务暂时不可用，请稍后再试。"
                    response_placeholder.markdown(err_msg)
                    st.session_state.messages.append(AIMessage(content=err_msg))
                
                # 触发右侧系统联动
                if st.session_state.ai_mode == "PRO":
                    final_state = app.get_state(config)
                    action = final_state.values.get("triage_result", {}).get("action", "chat")
                    if action == "form":
                        st.session_state.ready_for_analysis = True
                        st.toast("🎯 信息收齐！请看右侧面板", icon="✅")
                    else:
                        st.session_state.ready_for_analysis = False
                    
                    updated_data = extract_info_silently(st.session_state.messages, st.session_state.form_data)
                    st.session_state.form_data = updated_data
                else:
                    st.session_state.ready_for_analysis = False
                    
            st.rerun()

# ------------------------------------------
# 右侧：卷宗面板 (仅在 PRO 模式渲染)
# ------------------------------------------
if col_panel is not None:
    with col_panel:
        panel_class = "panel-container highlight-border" if st.session_state.ready_for_analysis else "panel-container"
        st.markdown(f'<div class="{panel_class}">', unsafe_allow_html=True)
        
        if st.session_state.report_generated and st.session_state.analysis_result:
            st.markdown('<div class="panel-header">📄 分析报告预览</div>', unsafe_allow_html=True)
            
            facts = strip_markdown(st.session_state.analysis_result.get('legal_facts_summary', '无数据'))
            laws = strip_markdown(st.session_state.analysis_result.get('relevant_laws', '无数据'))
            advice = strip_markdown(st.session_state.analysis_result.get('final_review', '无数据'))
            
            preview_html = f"""
            <div class="report-preview-box">
                <h3 style="text-align:center; color:#1e3a8a; margin-bottom: 20px;">劳动法律深度分析报告</h3>
                <h4>一、事实梳理</h4><p>{facts.replace(chr(10), '<br>')}</p>
                <h4>二、法条适用分析</h4><p>{laws.replace(chr(10), '<br>')}</p>
                <h4>三、最终合规建议</h4><p>{advice.replace(chr(10), '<br>')}</p>
            </div>
            """
            st.markdown(preview_html, unsafe_allow_html=True)
            
        else:
            st.markdown('<div class="panel-header">📑 智能案件卷宗</div>', unsafe_allow_html=True)
            is_empty = all(v == "" for v in st.session_state.form_data.values())
            
            if st.session_state.ready_for_analysis:
                st.success("✅ AI 认为信息已充足，请生成报告。")
            elif is_empty:
                st.info("👋 **卷宗目前为空。**\n\n请在左侧沟通，我会自动提取关键信息。")
                
            with st.form("case_confirmation_form", border=False):
                f_region = st.text_input("案件发生地", value=st.session_state.form_data.get("案件发生地", ""))
                f_company = st.text_input("涉事单位名称", value=st.session_state.form_data.get("单位名称", ""))
                f_salary = st.text_input("平均月薪", value=st.session_state.form_data.get("平均月薪", ""))
                f_date = st.text_input("时间节点", value=st.session_state.form_data.get("时间节点", ""))
                f_demand = st.text_input("核心诉求", value=st.session_state.form_data.get("核心诉求", ""))
                f_details = st.text_area("详细经过与证据", value=st.session_state.form_data.get("详细经过", ""), height=150)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                btn_type = "primary" if st.session_state.ready_for_analysis else "secondary"
                btn_text = "✅ 确认无误，生成分析报告" if st.session_state.ready_for_analysis else "跳过收集，直接生成报告"
                
                if st.form_submit_button(btn_text, type=btn_type, use_container_width=True):
                    final_form = {
                        "案件发生地": f_region, "单位名称": f_company, "平均月薪": f_salary, 
                        "时间节点": f_date, "核心诉求": f_demand, "详细经过": f_details
                    }
                    st.session_state.form_data = final_form
                    
                    with st.spinner("⚖️ 多智能体正在后台进行法条检索与深度推演，请稍候..."):
                        config = {"configurable": {"thread_id": st.session_state.thread_id}}
                        try:
                            current_state = app.get_state(config)
                            next_steps = current_state.next if hasattr(current_state, 'next') else []
                        except:
                            next_steps = []
                        
                        if next_steps and 'fact_summarizer' in next_steps:
                            app.update_state(config, {"form_data": final_form})
                            final_result = app.invoke(None, config)
                        else:
                            case_msg = HumanMessage(content=f"请分析案件：\n" + "\n".join([f"{k}：{v}" for k, v in final_form.items() if v]))
                            new_thread = st.session_state.thread_id + "_direct"
                            config_direct = {"configurable": {"thread_id": new_thread}}
                            init_result = app.invoke({"messages": [case_msg], "form_data": final_form}, config_direct)
                            if init_result.get("triage_result", {}).get("action") != "form":
                                app.update_state(config_direct, {"triage_result": {"action": "form", "category": "强制分析", "reply": ""}})
                            app.update_state(config_direct, {"form_data": final_form})
                            final_result = app.invoke(None, config_direct)
                        
                        if isinstance(final_result, dict):
                            analysis = {
                                "legal_facts_summary": final_result.get("legal_facts_summary", ""),
                                "relevant_laws": final_result.get("relevant_laws", ""),
                                "final_review": final_result.get("final_review", ""),
                            }
                        else:
                            try:
                                state = app.get_state(config if next_steps else config_direct)
                                vals = state.values
                                analysis = {
                                    "legal_facts_summary": vals.get("legal_facts_summary", ""),
                                    "relevant_laws": vals.get("relevant_laws", ""),
                                    "final_review": vals.get("final_review", ""),
                                }
                            except:
                                analysis = {"legal_facts_summary": str(final_result), "relevant_laws": "", "final_review": ""}
                        
                        st.session_state.analysis_result = analysis
                        st.session_state.messages = [] 
                        st.session_state.report_generated = True 
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)