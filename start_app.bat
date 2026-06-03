@echo off
chcp 65001 >nul
title AI 劳动法 - 律政视觉旗舰版 v3.0
color 0A

echo ============================================
echo    AI 劳动法
echo    律政视觉旗舰版 v3.0（前后端分离架构）
echo    开发者：罗志远 广东财经大学人工智能法研究中心
echo ============================================
echo.
echo 版本特性：
echo    • 前后端分离 - backend.py（引擎）+ streamlit（UI）
echo    • 离线建库 - build_db.py 处理 2500+ 法律文档
echo    • 多线程 Embedding - 8 线程并发调用 DashScope API
echo    • 律政视觉界面 - 品牌呼吸线、渐变按钮、毛玻璃特效
echo    • 入场动画 - 消息滑入、面板渐显
echo    • 悬浮模式切换 - 普法模式/案件模式胶囊按钮一键切换
echo    • 静默提示词注入 - 普法模式自动屏蔽卷宗收集
echo    • 附件上传交互 - 支持上传合同、截图等文件
echo    • A4纸沉浸式报告预览 - 下载前在线预览完整报告
echo    • 阅后即焚 - 生成报告后自动清空对话保护隐私
echo    • 纯净PDF - 生成带中文字体的专业PDF报告
echo    • 登录系统 - 账号密码验证，未登录无法使用
echo    • 用户注册 - 支持自主注册新账号
echo    • 管理员面板 - 管理员可管理用户账号
echo.

echo [1/7] 检查 Python 环境...
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 Python，请先安装 Python 3.10+
    pause
    exit /b 1
)

echo [2/7] 检查虚拟环境...
if not exist ".venv\Scripts\python.exe" (
    echo [信息] 未找到虚拟环境，正在创建...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [错误] 创建虚拟环境失败
        pause
        exit /b 1
    )
)

echo [3/7] 激活虚拟环境...
call .venv\Scripts\activate.bat

echo [4/7] 配置 LibreOffice 环境变量...
set "PATH=D:\1 下载\LibreOffice\program;%PATH%"
set PYTHONPATH=D:\1 下载\LibreOffice\program;%PYTHONPATH%

echo [5/7] 检查并安装依赖...
echo   正在安装依赖包，请稍候...
python -m pip install --upgrade pip -q 2>nul
pip install -r requirements.txt -q 2>nul

echo [6/7] 检查中文字体...
if exist "simhei.ttf" (
    echo [OK] 找到 simhei.ttf 字体文件
) else (
    if exist "C:/Windows/Fonts/simhei.ttf" (
        echo [OK] 找到系统黑体字体
    ) else (
        if exist "C:/Windows/Fonts/msyh.ttc" (
            echo [OK] 找到系统微软雅黑字体
        ) else (
            echo [警告] 未找到中文字体文件！
            echo [建议] 请下载 simhei.ttf 字体文件放到当前目录
            echo [下载] https://github.com/StellarCN/scp_zh/raw/master/fonts/simhei.ttf
            echo [备用] 使用系统自带的微软雅黑字体也可以正常工作
        )
    )
)

echo [7/7] 检查向量知识库...
if exist "vectorstore.pkl" (
    echo [OK] 找到向量知识库 vectorstore.pkl
) else (
    echo.
    echo ============================================
    echo [WARNING] 未找到向量知识库！
    echo.
    echo   请先在终端运行离线建库脚本：
    echo     python build_db.py
    echo.
    echo   预计耗时 15~30 分钟（2500+ 文档）
    echo   中断后可用 python build_db.py --from-cache 续传
    echo ============================================
    echo.
    choice /c YN /m "是否现在运行离线建库？"
    if errorlevel 2 goto skip_build
    if errorlevel 1 (
        echo.
        echo [建库] 开始离线建库，请耐心等待...
        python build_db.py
        if %errorlevel% neq 0 (
            echo [警告] 建库过程中出现错误，可稍后手动重试
        )
    )
    :skip_build
)

echo.
echo [启动] AI 劳动法 - 律政视觉旗舰版 v3.0...
echo [地址] http://localhost:8501
echo [提示] 启动后请在浏览器中访问上述地址
echo.

echo 使用说明：
echo    1. 首次使用请注册账号，或使用默认管理员 lzy / 123456 登录
echo    2. 登录后即可使用全部功能
echo    3. 聊天框上方点击「普法模式/案件模式」悬浮按钮切换
echo    4. 普法模式：直接回答法律疑问，不收集卷宗
echo    5. 案件模式：AI 自动提取案件信息到右侧表单
echo    6. 点击「📎附件」按钮可上传合同、截图等文件
echo    7. 当 AI 判断信息充足时，右侧面板会高亮提示
echo    8. 核对右侧卷宗信息，点击"生成法律分析报告"
echo    9. 右侧沉浸式 A4 纸预览报告，确认后下载 PDF
echo   10. 如需新案件，点击左侧边栏"彻底重置并开启新案"
echo.

set port=8501
set max_port=8510

::try_port
echo [尝试] 在端口 %port% 启动服务...
streamlit run streamlit_labor_law_complete.py --server.port %port% --server.headless false --theme.base light

if %errorlevel% neq 0 (
    set /a port+=1
    if %port% gtr %max_port% (
        echo.
        echo [错误] 尝试了 %max_port% 个端口均失败
        echo [建议] 请检查：
        echo        1. 端口是否被其他应用占用
        echo        2. 防火墙设置
        echo        3. 网络连接
        echo.
        pause
        exit /b 1
    )
    echo [信息] 端口 %port% 被占用，尝试下一个端口...
    goto try_port
)

pause
