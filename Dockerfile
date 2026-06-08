# ============================================
# 劳动法智能助理 - FastAPI 后端 Docker 镜像
# ============================================
# 构建:  docker build -t labor-law-api .
# 运行:  docker run -p 8000:8000 --env-file .env labor-law-api
# ============================================

FROM python:3.10-slim




# ---- 工作目录 ----
WORKDIR /app

# ---- 安装 Python 依赖（利用 Docker 缓存层）----
COPY backend_requirements.txt .
RUN pip install --no-cache-dir -r backend_requirements.txt

# ---- 拷贝应用代码（向量数据已存于 Pinecone 云端，无需 PDF）----
COPY api.py .

# ---- 创建非 root 用户（安全最佳实践）----
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# ---- 暴露端口 ----
EXPOSE 8000

# ---- 健康检查（每 60 秒检查一次）----
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# ---- 启动命令 ----
CMD ["uvicorn", "api:fastapi_app", "--host", "0.0.0.0", "--port", "8000"]
