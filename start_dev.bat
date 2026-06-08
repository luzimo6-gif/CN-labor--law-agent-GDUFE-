@echo off
chcp 65001 > NUL
title 劳动法 AI 助手 - 前后端双开

echo.
echo ╔══════════════════════════════════════════════╗
echo ║     劳动法 AI 智能助理 - 本地开发模式       ║
echo ╚══════════════════════════════════════════════╝
echo.

echo [1/2] 正在启动 FastAPI 后端服务 (端口 8000)...
start "FastAPI Backend" cmd /k "chcp 65001 > NUL && "d:\A DEMO\.venv\Scripts\python.exe" "d:\A DEMO\api.py""

timeout /t 3 /nobreak > NUL

echo [2/2] 正在启动 Streamlit 前端服务 (端口 8501)...
start "Streamlit Frontend" cmd /k "chcp 65001 > NUL && "d:\A DEMO\.venv\Scripts\python.exe" -m streamlit run "d:\A DEMO\streamlit_labor_law_complete.py""

echo.
echo ═══════════════════════════════════════════════
echo   ✅ 两个服务正在独立窗口中启动...
echo.
echo   🔧 后端 API : http://localhost:8000
echo   📖 API 文档 : http://localhost:8000/docs
echo   🌐 前端界面 : http://localhost:8501
echo.
echo   请等待两个窗口加载完毕后打开浏览器访问前端。
echo   关闭窗口即可停止对应服务。
echo ═══════════════════════════════════════════════
echo.
pause
