# AI 劳动法律师助理 - 左右分栏 Copilot 旗舰版

基于 LangGraph 多智能体工作流和 Copilot 交互模式的劳动法咨询平台。

## ✨ 版本特性

### 📱 左右分栏 Copilot 模式
- **左侧对话区（60%）** - 像微信一样自然聊天
- **右侧卷宗区（40%）** - 实时自动提取并展示案件信息
- **宽屏布局** - 充分利用屏幕空间，信息一目了然

### 🤖 后台信息实时静默提取
- **AI 自动提取** - 在聊天过程中自动识别并提取关键信息
- **实时更新** - 每轮对话后自动更新右侧表单
- **智能合并** - 新信息自动合并，不丢失历史数据

### 🎯 智能确认机制
- **自动判断** - AI 判断信息是否充足
- **高亮提示** - 信息充足时右侧面板高亮显示
- **灵活生成** - 信息不全也可手动生成报告

### 📄 HTML 报告导出
- **完美中文支持** - 使用 HTML + CSS，彻底解决字体问题
- **浏览器打开** - 可直接查看和打印为 PDF
- **专业排版** - 三段式报告：事实梳理 → 法条分析 → 合规建议

## 🚀 快速启动

### 一键启动
```bash
双击运行: start_labor_law_app.bat
```

### 手动启动
```bash
# 激活虚拟环境
.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 启动应用
streamlit run streamlit_labor_law_complete.py --server.port 8501
```

### 访问地址
```
http://localhost:8501
```

## 📁 项目结构

```
d:\A DEMO\
├── start_labor_law_app.bat          # 启动脚本
├── streamlit_labor_law_complete.py  # Streamlit 前端（左右分栏 Copilot 版）
├── labor_law_complete_fixed.py      # LangGraph 工作流核心
├── Labor_law.py                     # 原始工作流（参考）
├── .env                             # API 密钥配置
├── requirements.txt                 # 依赖文件
├── README.md                        # 说明文档
├── data/                            # 法律文档目录
└── chroma_db_labor_law_complete/    # 向量数据库
```

## 📋 使用流程

### 1. 开始对话
- 在左侧底部输入框像微信一样和 AI 聊天
- 描述您的案件情况，回答 AI 的问题

### 2. 实时建档
- AI 自动从对话中提取信息
- 右侧卷宗区实时更新显示
- 您可以直接在右侧修改或补充

### 3. 信息确认
- 当 AI 判断信息充足时，右侧面板会高亮提示
- 显示"AI 认为信息已充足，请核对"

### 4. 生成报告
- 点击"卷宗确认无误，生成法律分析报告"
- AI 多智能体进行深度分析
- 报告生成后在右侧显示下载按钮

### 5. 下载报告
- 点击下载 HTML 格式报告
- 用浏览器打开，按 Ctrl+P 可保存为 PDF

## 🔧 配置要求

### 环境变量
在 `.env` 文件中配置：
```bash
DASHSCOPE_API_KEY=sk-your-api-key-here
```

### Python 依赖
```bash
streamlit>=1.28.0
langchain>=0.1.0
langgraph>=0.0.47
langchain-chroma>=0.1.0
langchain-openai>=0.0.8
langchain-community>=0.0.10
langchain-text-splitters>=0.0.1
pypdf>=3.17.0
python-dotenv>=1.0.0
requests>=2.31.0
```

## 🛠️ 故障排除

### 端口占用
```bash
streamlit run streamlit_labor_law_complete.py --server.port 8502
```

### 依赖缺失
```bash
pip install -r requirements.txt
```

### API 密钥错误
检查 `.env` 文件中的 `DASHSCOPE_API_KEY` 是否正确

## 📞 技术支持

如遇到问题：
1. 检查虚拟环境是否激活
2. 检查依赖包是否完整安装
3. 检查 API 密钥是否正确配置
4. 检查端口是否可用

## 📄 许可证

本项目仅供学习和研究使用。

---

**启动应用：双击 `start_labor_law_app.bat`，访问 `http://localhost:8501`**

**左右分栏 Copilot 旗舰版：左侧聊天 + 右侧实时建档 + 智能确认**