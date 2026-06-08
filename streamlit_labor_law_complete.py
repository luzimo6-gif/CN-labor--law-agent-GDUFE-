#!/usr/bin/env python3
"""
劳动法智能助理 - Streamlit 前端（纯 UI 层）
开发者：罗志远 广东财经大学人工智能法研究中心研究人员
联系方式：1452723426@qq.com

注意：本文件是纯前端 UI，所有 AI 推理逻辑已迁移至 FastAPI 后端 (api.py)。
      前端通过 HTTP 调用 POST /chat 和 POST /analyze 接口与后端通信。
"""

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import uuid
import re
import json
import hashlib
import urllib3
import warnings
import requests
from datetime import datetime

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", module="urllib3")

# ==========================================
# API 后端地址（部署到 Streamlit Cloud 时改为生产地址）
# ==========================================
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
_VERIFY_SSL = os.getenv("VERIFY_SSL", "false").lower() == "true"

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
    /* ========================================================= */
    /* 藏青 + 金黄 品牌撞色视觉系统 */
    /* 主色：#0F2C5C (深邃藏青) | 辅色：#E6B800 (高级金黄) */
    /* ========================================================= */
    
    /* 隐藏默认元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background-color: transparent !important;}
    
    /* 1. 中文字体栈 */
    body, [class*="css"], .stTextInput, .stTextArea, .stMarkdown {
        font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Helvetica Neue", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji" !important;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    /* ========================================================= */
    /* 2. 登录页面背景 */
    /* ========================================================= */
    [data-testid="stApp"] > div:first-child {
        position: relative;
    }
    
    /* 背景图片层 */
    .login-background {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        opacity: 0.12;
        z-index: 0;
        pointer-events: none;
    }
    
    /* 登录卡片容器 */
    .login-card-container {
        position: relative;
        z-index: 10;
        background: linear-gradient(145deg, rgba(255,255,255,0.98) 0%, rgba(250,252,255,0.96) 100%);
        backdrop-filter: blur(30px);
        -webkit-backdrop-filter: blur(30px);
        border-radius: 24px;
        border: 1px solid rgba(15, 44, 92, 0.1);
        box-shadow: 
            0 30px 100px rgba(15, 44, 92, 0.25),
            0 0 0 1px rgba(230, 184, 0, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.9);
        padding: 50px 40px;
        margin: 20px auto;
        max-width: 480px;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    }
    
    .login-card-container:hover {
        box-shadow: 
            0 40px 120px rgba(15, 44, 92, 0.3),
            0 0 100px rgba(230, 184, 0, 0.15),
            0 0 0 2px rgba(230, 184, 0, 0.2);
        transform: translateY(-5px);
    }
    
    /* 登录标题区域 */
    .login-header {
        text-align: center;
        padding: 0 0 30px 0;
        border-bottom: 2px solid transparent;
        border-image: linear-gradient(90deg, transparent, #E6B800, #0F2C5C, transparent) 1;
        margin-bottom: 35px;
    }
    
    .login-icon {
        font-size: 3.5rem;
        margin-bottom: 15px;
        filter: drop-shadow(0 4px 8px rgba(230, 184, 0, 0.3));
        animation: iconFloat 3s ease-in-out infinite;
    }
    
    @keyframes iconFloat {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-8px); }
    }
    
    .login-title {
        font-weight: 800;
        background: linear-gradient(135deg, #0F2C5C 0%, #1a3a6a 50%, #E6B800 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.2rem;
        letter-spacing: -0.03em;
        margin: 0 0 10px 0;
    }
    
    .login-subtitle {
        color: #5a6a7a;
        font-size: 1rem;
        margin: 0;
        font-weight: 400;
    }
    
    /* 登录表单区域 */
    .login-form {
        padding: 10px 0;
    }
    
    /* 登录按钮撞色效果 */
    .login-form button[kind="primary"] {
        background: linear-gradient(135deg, #0F2C5C 0%, #1a3a6a 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        padding: 14px 24px !important;
        box-shadow: 
            0 6px 25px rgba(15, 44, 92, 0.3),
            0 0 40px rgba(230, 184, 0, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        position: relative;
        overflow: hidden;
        letter-spacing: 0.1em;
    }
    
    .login-form button[kind="primary"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(230, 184, 0, 0.3), transparent);
        transition: left 0.5s ease;
    }
    
    .login-form button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1a3a6a 0%, #0F2C5C 100%) !important;
        box-shadow: 
            0 10px 40px rgba(15, 44, 92, 0.4),
            0 0 60px rgba(230, 184, 0, 0.25),
            inset 0 1px 0 rgba(255, 255, 255, 0.15) !important;
        transform: translateY(-3px);
    }
    
    .login-form button[kind="primary"]:hover::before {
        left: 100%;
    }
    
    .login-form button[kind="primary"]:active {
        transform: translateY(-1px);
    }
    
    /* 登录输入框 */
    .login-form .stTextInput > div > div > input,
    .login-form .stTextArea > div > div > textarea {
        background: linear-gradient(145deg, #ffffff, #f8f9fc);
        border: 2px solid rgba(15, 44, 92, 0.1);
        border-radius: 12px;
        padding: 14px 16px;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .login-form .stTextInput > div > div > input:hover,
    .login-form .stTextArea > div > div > textarea:hover {
        border-color: #E6B800;
        box-shadow: 0 0 0 4px rgba(230, 184, 0, 0.1);
    }
    
    .login-form .stTextInput > div > div > input:focus,
    .login-form .stTextArea > div > div > textarea:focus {
        border-color: #0F2C5C;
        box-shadow: 
            0 0 0 4px rgba(15, 44, 92, 0.1),
            0 0 25px rgba(230, 184, 0, 0.1);
    }
    
    /* 标签页样式 */
    .login-tabs [data-baseweb="tab-list"] {
        background: rgba(15, 44, 92, 0.03);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
    }
    
    .login-tabs [data-baseweb="tab"] {
        border-radius: 10px;
        font-weight: 500;
        color: #5a6a7a;
        transition: all 0.25s ease;
    }
    
    .login-tabs [data-baseweb="tab"]:hover {
        background: rgba(230, 184, 0, 0.1);
        color: #0F2C5C;
    }
    
    .login-tabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0F2C5C 0%, #1a3a6a 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 15px rgba(15, 44, 92, 0.25);
    }
    
    /* ========================================================= */
    /* 3. 标题区域 - 藏青金黄撞色 */
    /* ========================================================= */
    .chat-title { 
        font-weight: 700; 
        color: #0F2C5C; 
        margin-bottom: 0.3rem; 
        font-size: 1.6rem; 
        letter-spacing: -0.01em; 
        background: linear-gradient(135deg, #0F2C5C 0%, #1a3a6a 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .chat-subtitle { 
        color: #5a6a7a; 
        font-size: 0.85rem; 
        margin-bottom: 1rem; 
        font-weight: 400; 
        border-left: 3px solid #E6B800;
        padding-left: 12px;
    }
    
    /* ========================================================= */
    /* 4. 消息动画特效 */
    /* ========================================================= */
    @keyframes slideUpFade {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes messageSlideIn {
        from { opacity: 0; transform: translateX(-10px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes avatarPop {
        0% { transform: scale(0.8); opacity: 0; }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); opacity: 1; }
    }
    
    .stChatMessage {
        animation: slideUpFade 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        position: relative;
        z-index: 2;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        border-radius: 12px;
    }
    
    .stChatMessage:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(15, 44, 92, 0.12), 
                    0 0 0 1px rgba(230, 184, 0, 0.1);
    }
    
    .stChatMessage[data-testid="stChatMessage"] > div:first-child {
        animation: messageSlideIn 0.3s ease-out;
    }
    
    .stChatMessage > div > div:has(img[alt*="assistant"]) {
        animation: avatarPop 0.4s ease-out;
    }
    
    /* ========================================================= */
    /* 4. 右侧面板 - 双层撞色呼吸阴影 */
    /* ========================================================= */
    .panel-container { 
        background-color: #ffffff; 
        border-radius: 12px; 
        padding: 20px; 
        border: 1px solid rgba(15, 44, 92, 0.08); 
        height: 85vh; 
        overflow-y: auto; 
        position: relative;
        z-index: 2;
        /* 双层阴影 - 藏青+金黄撞色 */
        box-shadow: 
            0 4px 20px rgba(15, 44, 92, 0.06),
            0 0 0 1px rgba(230, 184, 0, 0.05),
            inset 0 1px 0 rgba(255, 255, 255, 0.8);
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        animation: panelBreathingShadow 5s ease-in-out infinite;
    }
    
    @keyframes panelBreathingShadow {
        0%, 100% {
            box-shadow: 
                0 4px 20px rgba(15, 44, 92, 0.06),
                0 0 30px rgba(15, 44, 92, 0.03),
                0 0 0 1px rgba(230, 184, 0, 0.05);
        }
        50% {
            box-shadow: 
                0 8px 35px rgba(15, 44, 92, 0.1),
                0 0 50px rgba(230, 184, 0, 0.08),
                0 0 0 1px rgba(230, 184, 0, 0.1);
        }
    }
    
    .panel-container:hover {
        box-shadow: 
            0 15px 50px rgba(15, 44, 92, 0.15),
            0 0 60px rgba(230, 184, 0, 0.12),
            0 0 0 2px rgba(230, 184, 0, 0.15);
        border-color: rgba(230, 184, 0, 0.2);
        transform: translateY(-3px);
    }
    
    .panel-header { 
        color: #0F2C5C; 
        font-weight: 600; 
        font-size: 0.95rem; 
        margin-bottom: 14px; 
        border-bottom: 2px solid transparent;
        border-image: linear-gradient(90deg, #E6B800, #0F2C5C) 1;
        padding-bottom: 10px;
        transition: color 0.3s ease;
    }
    
    .highlight-border { 
        border: 2px solid #E6B800 !important; 
        box-shadow: 
            0 0 0 4px rgba(230, 184, 0, 0.15),
            0 4px 20px rgba(15, 44, 92, 0.1) !important; 
        transition: all 0.3s ease;
    }
    
    /* ========================================================= */
    /* 5. 报告预览卡片 - 撞色悬浮质感 */
    /* ========================================================= */
    .report-preview-box {
        background: linear-gradient(145deg, #ffffff 0%, #fafbfc 100%);
        border: 1px solid rgba(15, 44, 92, 0.1);
        border-radius: 12px;
        padding: 40px 30px;
        margin-top: 15px;
        font-size: 0.95rem;
        line-height: 1.8;
        color: #1a2b48;
        /* 三层撞色阴影 */
        box-shadow: 
            0 10px 40px rgba(15, 44, 92, 0.08),
            0 0 0 1px rgba(230, 184, 0, 0.08),
            inset 0 1px 0 rgba(255, 255, 255, 0.9);
        transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
        cursor: default;
        position: relative;
        overflow: hidden;
    }
    
    .report-preview-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #0F2C5C, #E6B800, #0F2C5C);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .report-preview-box:hover {
        transform: translateY(-6px) scale(1.005);
        box-shadow: 
            0 20px 60px rgba(15, 44, 92, 0.15),
            0 0 80px rgba(230, 184, 0, 0.12),
            0 0 0 2px rgba(230, 184, 0, 0.2);
        border-color: rgba(230, 184, 0, 0.25);
    }
    
    .report-preview-box:hover::before {
        opacity: 1;
    }
    
    .report-preview-box h4 { 
        color: #0F2C5C; 
        border-bottom: 2px solid #E6B800; 
        padding-bottom: 10px; 
        margin-top: 25px;
        transition: all 0.25s ease;
        position: relative;
    }
    
    .report-preview-box h4::before {
        content: '◆';
        color: #E6B800;
        margin-right: 8px;
        font-size: 0.7em;
    }
    
    .report-preview-box:hover h4 {
        color: #1a3a6a;
        border-bottom-color: #0F2C5C;
    }
    
    .report-preview-box p { margin-bottom: 15px; }
    
    /* ========================================================= */
    /* 6. 输入框 - 藏青金黄撞色边框 */
    /* ========================================================= */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea { 
        border-radius: 10px; 
        border: 1px solid rgba(15, 44, 92, 0.15); 
        background: linear-gradient(145deg, #ffffff, #f8f9fc);
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
    }
    
    .stTextInput>div>div>input::placeholder, 
    .stTextArea>div>div>textarea::placeholder {
        color: #8a9aaa;
        transition: color 0.2s ease;
    }
    
    .stTextInput>div>div>input:hover, 
    .stTextArea>div>div>textarea:hover {
        border-color: #E6B800;
        box-shadow: 
            0 0 0 3px rgba(230, 184, 0, 0.1),
            0 4px 15px rgba(15, 44, 92, 0.08);
    }
    
    .stTextInput>div>div>input:focus, 
    .stTextArea>div>div>textarea:focus {
        border-color: #0F2C5C;
        box-shadow: 
            0 0 0 3px rgba(15, 44, 92, 0.12),
            0 0 20px rgba(230, 184, 0, 0.1),
            0 4px 20px rgba(15, 44, 92, 0.1);
        outline: none;
    }
    
    .stTextInput>div>div>input:focus::placeholder, 
    .stTextArea>div>div>textarea:focus::placeholder {
        color: #c0c8d0;
    }
    
    /* ========================================================= */
    /* 7. 按钮 - 藏青金黄撞色深化 */
    /* ========================================================= */
    
    /* 主按钮 */
    button[kind="primary"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fc 100%) !important;
        color: #0F2C5C !important;
        border: 1px solid rgba(15, 44, 92, 0.2) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.45rem 1.1rem !important;
        box-shadow: 
            0 2px 8px rgba(15, 44, 92, 0.08),
            0 0 0 1px rgba(230, 184, 0, 0.1) !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        position: relative;
        overflow: hidden;
    }
    
    button[kind="primary"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(230, 184, 0, 0.15), transparent);
        transition: left 0.5s ease;
    }
    
    button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0F2C5C 0%, #1a3a6a 100%) !important;
        color: #ffffff !important;
        border-color: #0F2C5C !important;
        box-shadow: 
            0 6px 25px rgba(15, 44, 92, 0.25),
            0 0 40px rgba(230, 184, 0, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        transform: translateY(-3px);
    }
    
    button[kind="primary"]:hover::before {
        left: 100%;
    }
    
    button[kind="primary"]:active {
        transform: translateY(-1px);
        box-shadow: 0 2px 10px rgba(15, 44, 92, 0.15) !important;
    }

    /* 次级按钮 */
    button[kind="secondary"] {
        background: linear-gradient(135deg, #ffffff 0%, #fafbfc 100%) !important;
        border: 1px solid rgba(15, 44, 92, 0.15) !important;
        color: #0F2C5C !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        box-shadow: 0 2px 6px rgba(15, 44, 92, 0.05) !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
        position: relative;
        overflow: hidden;
    }
    
    button[kind="secondary"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(15, 44, 92, 0.06), transparent);
        transition: left 0.5s ease;
    }
    
    button[kind="secondary"]:hover {
        background: linear-gradient(135deg, #E6B800 0%, #d4a600 100%) !important;
        color: #0F2C5C !important;
        border-color: #E6B800 !important;
        box-shadow: 
            0 6px 20px rgba(230, 184, 0, 0.25),
            0 0 30px rgba(230, 184, 0, 0.15) !important;
        transform: translateY(-2px);
    }
    
    button[kind="secondary"]:hover::before {
        left: 100%;
    }
    
    /* 危险按钮 */
    button.danger-btn {
        color: #991b1b !important;
        border-color: rgba(153, 27, 27, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    button.danger-btn:hover {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%) !important;
        border-color: #dc2626 !important;
        color: #7f1d1d !important;
        box-shadow: 0 4px 15px rgba(153, 27, 27, 0.15) !important;
    }

    /* Popover胶囊按钮 */
    div[data-testid="stPopover"] > button {
        border: 2px solid rgba(15, 44, 92, 0.15) !important;
        background: linear-gradient(135deg, #ffffff, #f8f9fc) !important;
        color: #0F2C5C !important;
        font-weight: 600 !important;
        border-radius: 30px !important;
        padding: 8px 24px !important;
        box-shadow: 
            0 2px 8px rgba(15, 44, 92, 0.06),
            0 0 0 1px rgba(230, 184, 0, 0.08) !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        position: relative;
        overflow: hidden;
    }
    
    div[data-testid="stPopover"] > button::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 0;
        height: 3px;
        background: linear-gradient(90deg, #0F2C5C, #E6B800);
        border-radius: 3px 3px 0 0;
        transition: width 0.35s ease;
    }
    
    div[data-testid="stPopover"] > button:hover {
        border-color: #E6B800 !important;
        background: linear-gradient(135deg, #0F2C5C 0%, #1a3a6a 100%) !important;
        color: #ffffff !important;
        box-shadow: 
            0 6px 25px rgba(15, 44, 92, 0.2),
            0 0 40px rgba(230, 184, 0, 0.15) !important;
        transform: translateY(-3px) !important;
    }
    
    div[data-testid="stPopover"] > button:hover::after {
        width: 70%;
        background: linear-gradient(90deg, #E6B800, #ffffff);
    }

    /* 聊天输入框 */
    .stChatInputContainer {
        border-radius: 28px !important;
        border: 2px solid rgba(15, 44, 92, 0.12) !important;
        box-shadow: 
            0 4px 20px rgba(15, 44, 92, 0.06),
            0 0 0 1px rgba(230, 184, 0, 0.08) !important;
        background: linear-gradient(145deg, #ffffff, #fafbfc) !important;
        padding-left: 12px;
        z-index: 3 !important;
        transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    .stChatInputContainer:focus-within {
        border-color: #E6B800 !important;
        box-shadow: 
            0 0 0 4px rgba(230, 184, 0, 0.12),
            0 8px 30px rgba(15, 44, 92, 0.12),
            0 0 50px rgba(230, 184, 0, 0.08) !important;
    }

    /* ========================================================= */
    /* 8. 品牌记忆系统 - 藏青金黄贯穿 */
    /* ========================================================= */
    
    /* 顶部撞色呼吸渐变线 */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, #0F2C5C 0%, #1a3a6a 25%, #E6B800 50%, #1a3a6a 75%, #0F2C5C 100%);
        background-size: 200% 100%;
        z-index: 99999;
        animation: brandGradientFlow 8s ease-in-out infinite, topBarPulse 3s ease-in-out infinite alternate;
    }
    
    @keyframes brandGradientFlow {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    @keyframes topBarPulse {
        0% { opacity: 0.9; box-shadow: 0 0 10px rgba(230, 184, 0, 0.3); }
        100% { opacity: 1; box-shadow: 0 0 25px rgba(230, 184, 0, 0.5), 0 0 50px rgba(15, 44, 92, 0.2); }
    }

    /* 文本选中 - 金黄藏青 */
    ::selection {
        background-color: rgba(230, 184, 0, 0.25) !important; 
        color: #0F2C5C !important; 
    }

    /* 滚动条 - 撞色渐变 */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: linear-gradient(180deg, rgba(15, 44, 92, 0.03), rgba(230, 184, 0, 0.03));
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #0F2C5C 0%, #E6B800 100%);
        border-radius: 10px;
        transition: all 0.3s ease;
        box-shadow: 0 0 8px rgba(230, 184, 0, 0.3);
    }
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #E6B800 0%, #0F2C5C 100%);
        box-shadow: 0 0 15px rgba(230, 184, 0, 0.5);
    }

    /* 左侧边栏 */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f7f9fc 0%, #f0f2f5 100%) !important;
        border-right: 2px solid transparent;
        border-image: linear-gradient(180deg, #0F2C5C, #E6B800) 1;
        z-index: 3;
        transition: all 0.3s ease;
    }
    
    [data-testid="stSidebar"]:hover {
        background: linear-gradient(180deg, #ffffff 0%, #f7f9fc 100%) !important;
    }

    /* 律政水印 - 撞色脉冲 */
    .legal-watermark {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 30vw;
        max-width: 380px;
        z-index: 0;
        pointer-events: none;
        color: #c5cdd8;
        opacity: 0.4;
        animation: watermarkPulse 12s ease-in-out infinite;
        filter: drop-shadow(0 0 20px rgba(230, 184, 0, 0.1));
    }
    
    @keyframes watermarkPulse {
        0%, 100% { 
            opacity: 0.3; 
            transform: translate(-50%, -50%) scale(1);
            filter: drop-shadow(0 0 15px rgba(15, 44, 92, 0.1));
        }
        50% { 
            opacity: 0.5; 
            transform: translate(-50%, -50%) scale(1.03);
            filter: drop-shadow(0 0 30px rgba(230, 184, 0, 0.15));
        }
    }

    /* 开发者标签 */
    .developer-info {
        position: fixed;
        bottom: 15px;
        right: 20px;
        font-size: 0.7rem;
        color: #5a6a7a;
        z-index: 998;
        text-align: right;
        line-height: 1.6;
        background: linear-gradient(145deg, rgba(255,255,255,0.98), rgba(247,249,252,0.98)) !important;
        backdrop-filter: blur(10px);
        padding: 10px 16px;
        border-radius: 12px;
        border: 1px solid rgba(15, 44, 92, 0.08) !important;
        box-shadow: 
            0 4px 20px rgba(15, 44, 92, 0.08),
            0 0 0 1px rgba(230, 184, 0, 0.1);
        transition: all 0.3s ease;
    }
    .developer-info:hover {
        background: linear-gradient(145deg, #ffffff, #f8f9fc) !important;
        box-shadow: 
            0 8px 30px rgba(15, 44, 92, 0.12),
            0 0 30px rgba(230, 184, 0, 0.1),
            0 0 0 1px rgba(230, 184, 0, 0.2) !important;
        border-color: rgba(230, 184, 0, 0.3) !important;
    }
    .developer-info a { 
        color: #0F2C5C; 
        text-decoration: none; 
        transition: color 0.2s ease;
        font-weight: 500;
    }
    .developer-info a:hover { 
        color: #E6B800; 
    }

    /* ========================================================= */
    /* 9. 额外特效 */
    /* ========================================================= */
    
    /* 加载动画 */
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    .loading-shimmer {
        background: linear-gradient(90deg, #f0f2f5 25%, rgba(230, 184, 0, 0.1) 50%, #f0f2f5 75%);
        background-size: 200% 100%;
        animation: shimmer 1.5s infinite;
    }
    
    /* 下拉选择器 */
    .stSelectbox > div > div {
        border: 1px solid rgba(15, 44, 92, 0.12) !important;
        border-radius: 10px;
        transition: all 0.25s ease;
        background: linear-gradient(145deg, #ffffff, #f8f9fc);
    }
    
    .stSelectbox > div > div:hover {
        border-color: #E6B800 !important;
        box-shadow: 0 0 0 3px rgba(230, 184, 0, 0.1);
    }
    
    /* 标签页 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        background: linear-gradient(145deg, #ffffff, #f8f9fc);
        border: 1px solid rgba(15, 44, 92, 0.08);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(145deg, #f8f9fc, #f0f2f5);
        border-color: rgba(230, 184, 0, 0.3);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0F2C5C 0%, #1a3a6a 100%) !important;
        color: #ffffff !important;
        border-color: #0F2C5C !important;
        box-shadow: 0 4px 15px rgba(15, 44, 92, 0.2);
    }
    
    .stTabs [aria-selected="true"]:hover {
        background: linear-gradient(135deg, #1a3a6a 0%, #0F2C5C 100%) !important;
    }

    /* 分隔线撞色 */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(15, 44, 92, 0.1), rgba(230, 184, 0, 0.3), rgba(15, 44, 92, 0.1), transparent);
        transition: all 0.3s ease;
    }

    /* ========================================================= */
    /* 10. 移动端适配 - 弱化动画保留撞色 */
    /* ========================================================= */
    @media (max-width: 768px) {
        .chat-title { 
            font-size: 1.3rem; 
            background: linear-gradient(135deg, #0F2C5C 0%, #1a3a6a 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .panel-container { 
            height: auto !important; 
            max-height: 45vh !important;
            border-radius: 12px !important;
            animation: panelBreathingShadowMobile 6s ease-in-out infinite;
        }
        
        @keyframes panelBreathingShadowMobile {
            0%, 100% { 
                box-shadow: 
                    0 4px 15px rgba(15, 44, 92, 0.08),
                    0 0 0 1px rgba(230, 184, 0, 0.08);
            }
            50% { 
                box-shadow: 
                    0 6px 25px rgba(15, 44, 92, 0.1),
                    0 0 30px rgba(230, 184, 0, 0.06);
            }
        }
        
        .panel-container:hover {
            transform: none;
        }
        
        .report-preview-box:hover {
            transform: translateY(-3px);
        }
        
        div[data-testid="stPopover"] > button {
            padding: 6px 14px !important; 
            font-size: 0.85rem !important;
        }
        
        div[data-testid="stPopover"] > button:hover {
            transform: translateY(-1px) !important;
        }
        
        .legal-watermark { display: none !important; }
        
        /* 强制移动端纵向堆叠 */
        div[data-testid="stColumn"] {
            width: 100% !important;
            flex: 0 0 100% !important;
        }
        div[data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
        }
        
        .developer-info { display: none !important; }
        
        .report-preview-box { padding: 20px 16px; }
        
        button[kind="primary"], button[kind="secondary"] {
            min-height: 42px !important;
        }
        
        button[kind="primary"]:hover,
        button[kind="secondary"]:hover {
            transform: translateY(-1px);
        }
        
        /* 移动端顶部渐变线简化 */
        .stApp::before {
            animation: topBarPulseMobile 4s ease-in-out infinite alternate;
            height: 3px;
        }
        
        @keyframes topBarPulseMobile {
            0% { opacity: 0.85; box-shadow: 0 0 8px rgba(230, 184, 0, 0.2); }
            100% { opacity: 1; box-shadow: 0 0 15px rgba(230, 184, 0, 0.35); }
        }
        
        /* 移动端登录页面优化 */
        .login-card-container {
            padding: 30px 20px !important;
            margin: 15px 10px !important;
            border-radius: 16px !important;
            background: rgba(255, 255, 255, 0.99) !important;
        }
        
        .login-title {
            font-size: 1.6rem !important;
        }
        
        .login-icon {
            font-size: 2.5rem !important;
        }
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
# 0.4 背景图片 - Base64编码
# ==========================================
import base64
LOGIN_BG_IMAGE = ""
MAIN_BG_IMAGE = ""

# 登录背景图
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGIN_BG_PATH = os.path.join(_SCRIPT_DIR, "UI 图片", "UI桌面设计.png")
if os.path.exists(LOGIN_BG_PATH):
    with open(LOGIN_BG_PATH, "rb") as img_file:
        LOGIN_BG_IMAGE = f"data:image/png;base64,{base64.b64encode(img_file.read()).decode()}"

# 主页背景图
MAIN_BG_PATH = os.path.join(_SCRIPT_DIR, "UI 图片", "UI主页背景.png")
if os.path.exists(MAIN_BG_PATH):
    with open(MAIN_BG_PATH, "rb") as img_file:
        MAIN_BG_IMAGE = f"data:image/png;base64,{base64.b64encode(img_file.read()).decode()}"

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
# HTML表单值存储
if 'html_username' not in st.session_state:
    st.session_state.html_username = ""
if 'html_password' not in st.session_state:
    st.session_state.html_password = ""
if 'show_register' not in st.session_state:
    st.session_state.show_register = False

if not st.session_state.authenticated:
    # 登录页居中布局
    col_left, col_center, col_right = st.columns([1, 3, 1])
    
    with col_center:
        # 登录卡片 - 简洁风格
        st.markdown(f"""
        <style>
            /* 登录页面背景 */
            .stApp {{
                background-image: url("{LOGIN_BG_IMAGE}") !important;
                background-size: cover !important;
                background-position: center !important;
                background-repeat: no-repeat !important;
            }}
            
            /* 登录卡片 - 无背景透明 */
            .login-card {{
                background: transparent !important;
                border-radius: 0;
                padding: 0;
                box-shadow: none;
            }}
            
            /* Logo */
            .login-logo {{
                font-size: 4rem;
                text-align: center;
                margin-bottom: 10px;
                filter: drop-shadow(0 4px 8px rgba(230, 184, 0, 0.5));
            }}
            
            /* 标题 */
            .login-title {{
                text-align: center;
                font-size: 2.2rem;
                font-weight: 800;
                color: #0F2C5C;
                margin-bottom: 8px;
                text-shadow: 0 1px 2px rgba(255, 255, 255, 0.8);
            }}
            
            /* 副标题 */
            .login-subtitle {{
                text-align: center;
                color: #0F2C5C;
                font-size: 1rem;
                margin-bottom: 35px;
                letter-spacing: 0.15em;
                opacity: 0.8;
            }}
            
            /* 表单 */
            [data-testid="stForm"] {{
                background: transparent !important;
            }}
            
            /* 输入框 */
            .stTextInput > div {{
                background: rgba(255, 255, 255, 0.95) !important;
                border-radius: 10px !important;
                border: 2px solid rgba(255, 255, 255, 0.3) !important;
            }}
            
            .stTextInput > div:focus-within {{
                border-color: #E6B800 !important;
            }}
            
            /* 切换按钮样式 - 默认透明，黑色边框和文字 */
            [data-testid="stHorizontalBlock"] button {{
                width: 100% !important;
                background: rgba(255, 255, 255, 0.15) !important;
                color: #0F2C5C !important;
                border: 2px solid #0F2C5C !important;
                border-radius: 10px !important;
                padding: 12px 20px !important;
                font-size: 1rem !important;
                font-weight: 600 !important;
                margin: 0 5px !important;
                transition: all 0.3s ease !important;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15) !important;
            }}
            
            [data-testid="stHorizontalBlock"] button:hover {{
                background: #E6B800 !important;
                border-color: #E6B800 !important;
                color: #0F2C5C !important;
                transform: translateY(-2px) !important;
                box-shadow: 0 6px 20px rgba(230, 184, 0, 0.4) !important;
            }}
            
            /* 表单内按钮样式 - 默认透明，黑色边框和文字 */
            [data-testid="stForm"] button[kind="primary"], [data-testid="stForm"] button[kind="secondary"] {{
                width: 100% !important;
                background: rgba(255, 255, 255, 0.15) !important;
                color: #0F2C5C !important;
                border: 2px solid #0F2C5C !important;
                border-radius: 10px !important;
                padding: 14px 24px !important;
                font-size: 1rem !important;
                font-weight: 600 !important;
                margin-top: 15px !important;
                transition: all 0.3s ease !important;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15) !important;
            }}
            
            [data-testid="stForm"] button[kind="primary"]:hover, [data-testid="stForm"] button[kind="secondary"]:hover {{
                background: #E6B800 !important;
                border-color: #E6B800 !important;
                color: #0F2C5C !important;
                transform: translateY(-2px) !important;
                box-shadow: 0 8px 25px rgba(230, 184, 0, 0.4) !important;
            }}
            
            /* 密码显示按钮 */
            [data-testid="stTextInput"] button {{
                border: 1px solid rgba(15, 44, 92, 0.3) !important;
                background: transparent !important;
                box-shadow: none !important;
                border-radius: 6px !important;
                padding: 4px 8px !important;
                min-width: 32px !important;
                width: 32px !important;
                height: 32px !important;
            }}
            
            [data-testid="stTextInput"] button:hover {{
                border-color: #E6B800 !important;
                background: rgba(230, 184, 0, 0.1) !important;
            }}
        </style>
        
        <div class="login-card">
            <div class="login-logo">⚖️</div>
            <h1 class="login-title">AI 劳动法</h1>
            <p class="login-subtitle">智能法律咨询系统</p>
        """, unsafe_allow_html=True)
        
        # 登录表单
        # 切换按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔐 登录", use_container_width=True, type="primary"):
                st.session_state.show_register = False
                st.rerun()
        with col2:
            if st.button("📝 注册", use_container_width=True):
                st.session_state.show_register = True
                st.rerun()
        
        st.markdown("<hr style='margin: 20px 0; border-color: rgba(255,255,255,0.2);'>", unsafe_allow_html=True)
        
        # 根据状态显示对应表单
        if st.session_state.show_register:
            # 注册表单
            with st.form("register_form", clear_on_submit=True):
                new_user = st.text_input("用户名", placeholder="请设置用户名")
                new_name = st.text_input("姓名", placeholder="请输入真实姓名")
                new_pwd = st.text_input("密码", type="password", placeholder="请设置密码（至少6位）")
                new_pwd2 = st.text_input("确认密码", type="password", placeholder="请再次输入密码")
                reg_submitted = st.form_submit_button("完成注册", type="primary", use_container_width=True)
                
                if reg_submitted:
                    if not new_user or not new_pwd or not new_name:
                        st.warning("请填写所有字段")
                    elif len(new_pwd) < 6:
                        st.warning("密码至少6位")
                    elif new_pwd != new_pwd2:
                        st.warning("两次密码不一致")
                    elif create_user(new_user, new_pwd, new_name):
                        st.success("注册成功！请登录")
                        st.session_state.show_register = False
                        st.rerun()
                    else:
                        st.warning("该用户名已存在")
        else:
            # 登录表单
            with st.form("login_form", clear_on_submit=True):
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
        
        # 关闭卡片
        st.markdown("""
        </div>
        """, unsafe_allow_html=True)
    
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
# 1. 后端 API 连通性检查
# ==========================================
@st.cache_resource(ttl=30)
def check_backend():
    """检查后端 API 是否可达"""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return True, data.get("components", {})
        return False, {}
    except Exception as e:
        return False, {"error": str(e)}

backend_ok, backend_info = check_backend()
if not backend_ok:
    st.warning(f"⚠️ 后端服务未连接 ({API_BASE_URL})。请先启动 `python api.py`。")
else:
    st.sidebar.success(f"🟢 后端已连接 ({API_BASE_URL})")

# ==========================================
# 2. 状态初始化（消息存储为纯字典，无 langchain 对象）
# ==========================================
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = f"user-{uuid.uuid4().hex[:8]}"
if 'messages' not in st.session_state:
    st.session_state.messages = []  # 每个元素: {"role": "user"/"assistant", "content": "..."}
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
    """专为流式输出优化的极简解析：稳定提取思考与正文，无过度清洗"""
    thinking = ""
    output = ""

    if not text or not text.strip():
        return "", ""

    # 1. 提取 <thinking>
    think_match = re.search(r'<thinking>(.*?)(?:</thinking>|$)', text, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()

    # 2. 提取 <output>
    out_match = re.search(r'<output>(.*?)(?:</output>|$)', text, re.DOTALL)

    if out_match:
        json_str = out_match.group(1).strip()
        # 清理代码块外壳
        json_str = re.sub(r'^```(?:json)?\s*', '', json_str)
        json_str = re.sub(r'\s*```$', '', json_str)

        # 3. 尝试 JSON 解析
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict) and "reply" in parsed:
                output = parsed["reply"]
            else:
                for v in parsed.values():
                    if isinstance(v, str) and len(v) > 10:
                        output = v
                        break
                if not output:
                    output = json_str
        except (json.JSONDecodeError, ValueError):
            # 4. JSON 未闭合（流式中常见），正则兜底提取 reply
            reply_match = re.search(r'"reply"\s*:\s*"((?:[^"\\]|\\.)*)"', json_str)
            if reply_match:
                output = reply_match.group(1).replace('\\n', '\n')
            else:
                # 5. 完全无法解析，剥掉标签返回原文
                output = re.sub(r'</?output>', '', text)
                output = re.sub(r'</?thinking>.*?(?:</thinking>|$)', '', output, flags=re.DOTALL)
                output = re.sub(r'^```(?:json)?\s*', '', output)
                output = re.sub(r'```$', '', output).strip()
    else:
        # 没有 <output> 标签
        output = text
        if think_match:
            output = re.sub(r'<thinking>.*?(?:</thinking>|$)', '', output, flags=re.DOTALL).strip()

        output = re.sub(r'^```(?:json)?\s*', '', output)
        output = re.sub(r'```$', '', output).strip()

        # 如果正文看起来像 JSON，尝试提取 reply
        if output.strip().startswith('{'):
            try:
                parsed = json.loads(output.strip())
                if isinstance(parsed, dict) and "reply" in parsed:
                    output = parsed["reply"]
            except (json.JSONDecodeError, ValueError):
                pass

    # 清理残留 XML 标签
    output = re.sub(r'</?(?:thinking|output|system|instruction)\b[^>]*>', '', output)

    return thinking, output

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

def _find_font_path():
    """查找中文字体（仅相对路径，兼容 Streamlit Cloud Linux 环境）"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "simhei.ttf"),
        os.path.join(base_dir, "fonts", "simhei.ttf"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p).replace('\\', '/')
    return None

_FONT_PATH = _find_font_path()

def _clean_text(text):
    """清理 AI 输出中的特殊字符，保留结构标记（## 等）"""
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', text)          # 去HTML标签
    text = re.sub(r'```[\s\S]*?```', '', text)   # 去代码块
    text = re.sub(r'`([^`]+)`', r'\1', text)     # 去行内代码
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text) # 去加粗
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # 去斜体
    # 只保留 CJK + ASCII 可打印字符 + 常见中文标点
    cleaned = []
    for ch in text:
        cp = ord(ch)
        if (0x20 <= cp <= 0x7E or          # ASCII 可打印（含 # - * 等）
            0x4E00 <= cp <= 0x9FFF or       # CJK 统一汉字
            0x3400 <= cp <= 0x4DBF or       # CJK 扩展A
            0x3000 <= cp <= 0x303F or       # 中文标点
            0xFF00 <= cp <= 0xFFEF or       # 全角字符
            cp in (0x000A, 0x000D) or       # 换行
            0x2000 <= cp <= 0x206F or       # 通用标点
            0x2E80 <= cp <= 0x2EFF or       # CJK 部首补充
            0x2F00 <= cp <= 0x2FDF):        # 康熙部首
            cleaned.append(ch)
    text = ''.join(cleaned)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def _parse_ai_sections(text):
    """将 AI 输出按 ## 标题拆分为 [(标题, 正文), ...]"""
    sections = []
    current_title = ""
    current_lines = []
    for line in text.split('\n'):
        stripped = line.strip()
        # 匹配 ## 开头的章节标题
        if re.match(r'^##\s+', stripped):
            # 保存上一节
            if current_title or current_lines:
                sections.append((current_title, '\n'.join(current_lines).strip()))
            current_title = re.sub(r'^##\s+', '', stripped).strip()
            current_lines = []
        else:
            current_lines.append(stripped)
    # 最后一节
    if current_title or current_lines:
        sections.append((current_title, '\n'.join(current_lines).strip()))
    return sections

@st.cache_data(show_spinner=False)
def create_pdf_report_v3(form_data_json: str, result_json: str):
    """直接用 reportlab 生成 PDF（纯 Python，无系统依赖）"""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.colors import HexColor
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from io import BytesIO

    form_data = json.loads(form_data_json)
    result_dict = json.loads(result_json)

    # ── 注册字体 ──
    ff = "SimHei" if _FONT_PATH else "Helvetica"
    if _FONT_PATH:
        try:
            pdfmetrics.registerFont(TTFont('SimHei', _FONT_PATH))
        except Exception as e:
            print(f"[PDF] 字体注册失败: {e}，使用 Helvetica")
            ff = "Helvetica"

    # ── 样式定义 ──
    BLUE = HexColor('#1e3a8a')
    DARK = HexColor('#1e293b')
    GRAY = HexColor('#64748b')

    style_title = ParagraphStyle('Title', fontName=ff, fontSize=18, textColor=BLUE,
                                  alignment=1, spaceAfter=6, leading=24)
    style_date = ParagraphStyle('Date', fontName=ff, fontSize=10, textColor=GRAY,
                                 alignment=1, spaceAfter=4)
    style_h2 = ParagraphStyle('H2', fontName=ff, fontSize=14, textColor=BLUE,
                               spaceBefore=14, spaceAfter=6, leading=20)
    style_label = ParagraphStyle('Label', fontName=ff, fontSize=10, textColor=DARK,
                                  leftIndent=10, spaceAfter=2, leading=16)
    style_body = ParagraphStyle('Body', fontName=ff, fontSize=10, textColor=DARK,
                                 spaceBefore=4, spaceAfter=4, leading=17,
                                 firstLineIndent=20)
    style_footer = ParagraphStyle('Footer', fontName=ff, fontSize=8, textColor=GRAY,
                                   alignment=1, spaceBefore=12)

    # ── 构建 PDF ──
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=20*mm, rightMargin=20*mm,
                            topMargin=20*mm, bottomMargin=20*mm)

    story = []

    # 标题
    story.append(Paragraph("劳动法律合规建议报告", style_title))
    story.append(Paragraph(f"生成日期：{datetime.now().strftime('%Y年%m月%d日')}", style_date))
    story.append(Spacer(1, 4*mm))
    story.append(HRFlowable(width="100%", thickness=1.5, color=BLUE))
    story.append(Spacer(1, 4*mm))

    # 案件基本信息
    story.append(Paragraph("案件基本信息", style_h2))
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor('#e2e8f0')))
    story.append(Spacer(1, 2*mm))
    for k, v in form_data.items():
        if v:
            safe_k = _clean_text(k).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            safe_v = _clean_text(str(v)).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(Paragraph(f"<b>{safe_k}</b>：{safe_v}", style_label))
    story.append(Spacer(1, 4*mm))
    story.append(HRFlowable(width="100%", thickness=1.5, color=BLUE))
    story.append(Spacer(1, 4*mm))

    # 正文：汇总所有 AI 分析内容
    # 1) 事实梳理
    facts_raw = result_dict.get('legal_facts_summary', '')
    if facts_raw:
        story.append(Paragraph("案件事实梳理", style_h2))
        story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor('#e2e8f0')))
        story.append(Spacer(1, 2*mm))
        facts_text = _clean_text(facts_raw)
        for line in facts_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            safe = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(Paragraph(safe, style_body))
        story.append(Spacer(1, 3*mm))

    # 2) 法条适用分析
    laws_raw = result_dict.get('relevant_laws', '')
    if laws_raw:
        story.append(Paragraph("法条适用分析", style_h2))
        story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor('#e2e8f0')))
        story.append(Spacer(1, 2*mm))
        laws_text = _clean_text(laws_raw)
        for line in laws_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            safe = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(Paragraph(safe, style_body))
        story.append(Spacer(1, 3*mm))

    # 3) 合规建议报告（按 ## 章节拆分，结构化排版）
    review_raw = result_dict.get('final_review', '')
    if review_raw:
        review_text = _clean_text(review_raw)
        print(f"[PDF] cleaned review text:\n{review_text[:500]}...")

        sections = _parse_ai_sections(review_text)

        for sec_title, sec_body in sections:
            # 章节标题
            if sec_title:
                story.append(Paragraph(sec_title, style_h2))
                story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor('#e2e8f0')))
                story.append(Spacer(1, 2*mm))

            # 章节正文：按行渲染，识别列表项和普通段落
            if sec_body:
                for line in sec_body.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    safe = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    # 列表项（- 开头或 数字. 开头）
                    if re.match(r'^[-*]\s+', line):
                        item_text = re.sub(r'^[-*]\s+', '', safe)
                        story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;{item_text}", style_label))
                    elif re.match(r'^\d+[.、．]\s*', line):
                        item_text = re.sub(r'^\d+[.、．]\s*', '', safe)
                        story.append(Paragraph(f"&nbsp;&nbsp;&nbsp;&nbsp;{item_text}", style_label))
                    else:
                        story.append(Paragraph(safe, style_body))
            story.append(Spacer(1, 3*mm))

    story.append(Spacer(1, 6*mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor('#e2e8f0')))
    story.append(Paragraph("本报告由 AI 劳动法智能助理自动生成，仅供参考，不构成法律意见。", style_footer))

    doc.build(story)
    return buf.getvalue()

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
        st.rerun()

# ==========================================
# 5. 主页面布局：根据模式动态切换右侧栏显示
# ==========================================

# 主页背景图
if MAIN_BG_IMAGE:
    st.markdown(f"""
    <style>
        .stApp {{
            background-image: url("{MAIN_BG_IMAGE}") !important;
            background-size: cover !important;
            background-position: center !important;
            background-repeat: no-repeat !important;
        }}
        /* 左边收缩边框透明，显示背景图 */
        [data-testid="stSidebar"] {{
            background: transparent !important;
            border: none !important;
        }}
    </style>
    """, unsafe_allow_html=True)

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
            role = msg["role"]
            with st.chat_message(role):
                if role == "assistant":
                    thinking, output = parse_ai_message(msg["content"])
                    if thinking:
                        # 历史消息中的思考默认折叠
                        with st.expander("✅ 已完成思考", expanded=False):
                            st.markdown(thinking)
                    if output:
                        st.markdown(output)
                else:
                    st.write(msg["content"])
        
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

        # 聊天输入框及提交逻辑 (通过 API 调用后端)
        if prompt := st.chat_input("描述您的遭遇或提出疑问...", key="main_chat_input"):
            
            # 1. 展示用户消息
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                thinking_status = st.status("⚖️ AI 正在思考推演...", expanded=False)
                thinking_placeholder = thinking_status.empty()
                response_placeholder = st.empty()

                # 🌟 普法模式下，前端注入系统指令
                if st.session_state.ai_mode == "QUICK":
                    final_query = f"【系统前置绝对指令：当前处于快速普法模式。请直接以专业律师口吻回答该问题，绝对不要试图收集案卷要素，也不要提示用户看右侧表单。】\n用户：{prompt}"
                else:
                    final_query = prompt

                # 调用后端 POST /chat
                reply_text = ""
                action = "chat"
                thinking_text = ""
                try:
                    resp = requests.post(
                        f"{API_BASE_URL}/chat",
                        json={
                            "query": final_query,
                            "thread_id": st.session_state.thread_id,
                        },
                        timeout=90,
                        verify=_VERIFY_SSL,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        reply_text = data.get("reply", "")
                        action = data.get("action", "chat")
                        thinking_text = data.get("thinking", "")
                        
                        # 展示思考过程
                        if thinking_text:
                            thinking_placeholder.markdown(thinking_text)
                        # 展示回复
                        if reply_text:
                            response_placeholder.markdown(reply_text)
                        
                        thinking_status.update(label="✅ 已完成思考", state="complete", expanded=False)
                    else:
                        raise Exception(f"HTTP {resp.status_code}: {resp.text[:200]}")
                except Exception as e:
                    import traceback
                    print(f"[ERROR] API 调用失败: {type(e).__name__}: {e}")
                    traceback.print_exc()
                    thinking_status.update(label="⚠️ 出错了", state="error", expanded=False)
                    reply_text = f"抱歉，后端服务暂时不可用。请确保已启动 `python api.py`。\n> 错误: {type(e).__name__}"
                    response_placeholder.markdown(reply_text)
                    action = "chat"

                # 保存 AI 消息
                full_content = reply_text
                if thinking_text and thinking_text not in full_content:
                    full_content = f"<thinking>{thinking_text}</thinking>\n{reply_text}"
                st.session_state.messages.append({"role": "assistant", "content": full_content.strip()})

                # PRO 模式下判断是否触发分析
                if st.session_state.ai_mode == "PRO":
                    st.session_state.context_round_count += 1
                    round_count = st.session_state.context_round_count

                    # 评估信息完善度
                    completeness_pct, missing_fields, suggestion = evaluate_form_completeness(
                        st.session_state.form_data
                    )

                    ai_says_ready = action == "form"
                    data_says_ready = completeness_pct >= 60
                    rounds_exceeded = round_count >= 10

                    if ai_says_ready or data_says_ready:
                        st.session_state.ready_for_analysis = True
                        st.toast("🎯 核心信息已收集完毕！请查看右侧面板并生成报告。", icon="✅")
                    elif rounds_exceeded and completeness_pct >= 40:
                        st.session_state.ready_for_analysis = True
                        st.toast(f"⏰ 已对话 {round_count} 轮，信息完善度 {completeness_pct}%。建议生成报告。", icon="📋")
                    else:
                        st.session_state.ready_for_analysis = False
                        if round_count % 5 == 0 and round_count > 0:
                            st.toast(f"📊 已对话 {round_count} 轮，信息完善度 {completeness_pct}%。{suggestion}", icon="💡")
                else:
                    st.session_state.ready_for_analysis = False

            st.rerun()

# ------------------------------------------
# 右侧：卷宗收集区(上) + 报告预览区(下)
# ------------------------------------------
if col_panel is not None:
    with col_panel:
        # 上部：卷宗收集区
        st.markdown('<div class="panel-header">📑 智能案件卷宗</div>', unsafe_allow_html=True)
        
        if st.session_state.ai_mode == "QUICK":
            st.info("⚡ 当前为**普法模式**，AI 只负责快速解答法律疑问，不收集案件卷宗。如需出具正式案件报告，请在聊天框上方切换至「案件模式」。")
            
        elif st.session_state.ai_mode == "PRO":
            is_empty = all(v == "" for v in st.session_state.form_data.values())
            completeness_pct, missing_fields, suggestion = evaluate_form_completeness(st.session_state.form_data)
            
            if st.session_state.ready_for_analysis:
                st.success("✅ AI 认为信息已充足，请核对下方数据并生成报告。")
            elif is_empty:
                st.info("👋 **卷宗目前为空。**\n\n请在左侧向我描述您的案情，我会自动为您提取并填写此处的关键信息。")
            else:
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
                st.caption("左侧沟通时，AI 会自动为您更新下方信息。")
            
        with st.form("case_confirmation_form", border=False):
            disabled_status = st.session_state.ai_mode == "QUICK"
            
            f_region = st.text_input("案件发生地", value=st.session_state.form_data.get("案件发生地", ""), disabled=disabled_status)
            f_company = st.text_input("涉事单位名称", value=st.session_state.form_data.get("单位名称", ""), disabled=disabled_status)
            f_salary = st.text_input("平均月薪", value=st.session_state.form_data.get("平均月薪", ""), disabled=disabled_status)
            f_date = st.text_input("时间节点", value=st.session_state.form_data.get("时间节点", ""), disabled=disabled_status)
            f_demand = st.text_input("核心诉求", value=st.session_state.form_data.get("核心诉求", ""), disabled=disabled_status)
            f_details = st.text_area("详细经过与证据", value=st.session_state.form_data.get("详细经过", ""), height=120, disabled=disabled_status)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.session_state.ai_mode == "PRO":
                btn_type = "primary" if st.session_state.ready_for_analysis else "secondary"
                btn_text = "✅ 生成法律分析报告" if st.session_state.ready_for_analysis else "强制生成报告"
                
                if st.form_submit_button(btn_text, type=btn_type, use_container_width=True):
                        final_form = {
                            "案件发生地": f_region, "单位名称": f_company, "平均月薪": f_salary, 
                            "时间节点": f_date, "核心诉求": f_demand, "详细经过": f_details
                        }
                        st.session_state.form_data = final_form
                        
                        is_force = not st.session_state.ready_for_analysis
                        
                        with st.status("⚖️ 正在调用 AI 多智能体分析案情...", expanded=True) as status_ctx:
                            st.write("🚀 分析请求已发送至后端...")

                            try:
                                resp = requests.post(
                                    f"{API_BASE_URL}/analyze",
                                    json={
                                        "thread_id": st.session_state.thread_id,
                                        "form_data": final_form,
                                        "force": is_force,
                                    },
                                    timeout=180,  # 完整分析可能需要 1-3 分钟
                                    verify=_VERIFY_SSL,
                                )
                                if resp.status_code == 200:
                                    data = resp.json()
                                    analysis = {
                                        "legal_facts_summary": data.get("legal_facts_summary", ""),
                                        "relevant_laws": data.get("relevant_laws", ""),
                                        "final_review": data.get("final_review", ""),
                                    }
                                    status_ctx.update(
                                        label="✅ 深度分析完成！",
                                        state="complete",
                                        expanded=False
                                    )
                                else:
                                    raise Exception(f"HTTP {resp.status_code}: {resp.text[:200]}")
                            except Exception as e:
                                import traceback
                                traceback.print_exc()
                                status_ctx.update(label="⚠️ 分析失败", state="error", expanded=True)
                                st.error(f"后端分析失败: {e}")
                                analysis = {
                                    "legal_facts_summary": f"分析出错: {e}",
                                    "relevant_laws": "",
                                    "final_review": "",
                                }
                        
                        st.session_state.analysis_result = analysis
                        st.session_state.messages = [] 
                        st.session_state.report_generated = True 
                        st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
        
        # 下部：报告预览区
        if st.session_state.report_generated and st.session_state.analysis_result:
            st.markdown('<div class="panel-header">📄 分析报告预览</div>', unsafe_allow_html=True)
            
            form_json = json.dumps(st.session_state.form_data, ensure_ascii=False, sort_keys=True)
            result_json = json.dumps(st.session_state.analysis_result, ensure_ascii=False, sort_keys=True)
            pdf_bytes = create_pdf_report_v3(form_json, result_json)
            
            st.download_button(
                label="📥 下载 PDF 格式正式报告",
                data=pdf_bytes,
                file_name=f"劳动法分析报告_{datetime.now().strftime('%Y%m%d%H%M')}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )
            
            advice = strip_markdown(st.session_state.analysis_result.get('final_review', '无数据'))
            
            preview_html = f"""
            <div class="report-preview-box">
                <h3 style="text-align:center; color:#1e40af; margin-bottom: 15px;">劳动法律合规建议报告</h3>
                <p>{advice.replace(chr(10), '<br>')}</p>
            </div>
            """
            st.markdown(preview_html, unsafe_allow_html=True)