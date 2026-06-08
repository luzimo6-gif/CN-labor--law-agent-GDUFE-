#!/usr/bin/env python3
"""
劳动法律师智能助理 - FastAPI 后端服务
=========================================
将 LangGraph 多智能体推理引擎独立为 API，供 Streamlit 前端或其他客户端调用。
使用 Pinecone 云端向量库 + DashScope Embeddings + DeepSeek LLM。

启动方式：
    uvicorn api:app --host 0.0.0.0 --port 8000
    或
    python api.py
"""

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import json
import re
import uuid
import traceback
import warnings
from typing import TypedDict, List, Dict, Any, Annotated, Optional, Literal
from contextlib import asynccontextmanager

# =========================
# 环境变量与基础库
# =========================
from dotenv import load_dotenv
load_dotenv(override=True)

# 抑制 SSL 警告（兼容内部代理环境）
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)

# FastAPI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field as PydanticField

# LangChain / LangGraph
from langchain_openai import ChatOpenAI
from langchain.embeddings.base import Embeddings
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.documents import Document as LangchainDocument

# Pinecone (云端向量库)
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# ==========================================
# 0. 全局配置（全部从环境变量读取）
# ==========================================
# DashScope Embeddings
DASHSCOPE_API_KEY      = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL     = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
DASHSCOPE_EMBED_MODEL  = os.getenv("DASHSCOPE_EMBED_MODEL", "text-embedding-v2")
EMBED_DIMENSION        = int(os.getenv("EMBED_DIMENSION", "1536"))

# DeepSeek LLM
DEEPSEEK_API_KEY       = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL      = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODELS        = ["deepseek-chat"]

# Pinecone
PINECONE_API_KEY       = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME    = os.getenv("PINECONE_INDEX_NAME", "labor-law")
PINECONE_CLOUD         = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION        = os.getenv("PINECONE_REGION", "us-east-1")

# 数据目录
DATA_DIR               = os.getenv("DATA_DIR", "./data/")

# SSL
VERIFY_SSL             = os.getenv("VERIFY_SSL", "false").lower() == "true"

# 启动时打印配置（脱敏）
_m = DEEPSEEK_API_KEY
_MASKED_DS = _m[:8] + "..." + _m[-4:] if len(_m) > 12 else "(KEY too short)"
_m2 = DASHSCOPE_API_KEY
_MASKED_DSCOPE = _m2[:8] + "..." + _m2[-4:] if len(_m2) > 12 else "(KEY too short)"
_m3 = PINECONE_API_KEY
_MASKED_PC = _m3[:8] + "..." + _m3[-4:] if len(_m3) > 12 else "(KEY too short)"

print(f">>> DeepSeek   Key : {_MASKED_DS}")
print(f">>> DeepSeek   URL : {DEEPSEEK_BASE_URL}")
print(f">>> DashScope   Key : {_MASKED_DSCOPE}")
print(f">>> DashScope   URL : {DASHSCOPE_BASE_URL}")
print(f">>> Pinecone    Key : {_MASKED_PC}")
print(f">>> Pinecone  Index: {PINECONE_INDEX_NAME}")

# ==========================================
# 1. 自定义 DashScope Embeddings
# ==========================================
class DashScopeEmbeddings(Embeddings):
    """DashScope 文本向量化（使用原生 SDK 批量调用，兼容 LangChain 接口）"""

    MAX_BATCH_SIZE = 25  # DashScope API 单次最大输入条数

    def __init__(self, api_key: str, model: str = "text-embedding-v2"):
        self.api_key = api_key
        self.model = model
        import dashscope
        dashscope.api_key = api_key

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        from dashscope import TextEmbedding
        embeddings = []
        for i in range(0, len(texts), self.MAX_BATCH_SIZE):
            batch = texts[i:i + self.MAX_BATCH_SIZE]
            try:
                resp = TextEmbedding.call(
                    model=self.model,
                    input=batch if len(batch) > 1 else batch[0],
                )
                if resp.status_code == 200 and resp.output and resp.output.get("embeddings"):
                    for emb in resp.output["embeddings"]:
                        embeddings.append(emb["embedding"])
                else:
                    err = resp.message or f"HTTP {resp.status_code}"
                    print(f"[EMBED] DashScope API 错误: {err}")
                    embeddings.extend([[0.0] * EMBED_DIMENSION] * len(batch))
            except Exception as e:
                print(f"[EMBED] Error: {e}")
                embeddings.extend([[0.0] * EMBED_DIMENSION] * len(batch))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


# ==========================================
# 2. LLM 自动回退包装器
# ==========================================
class AutoFallbackLLM:
    """余额不足时自动切换到下一个模型"""

    def __init__(self, models: List[str], api_key: str, base_url: str,
                 temperature: float = 0.3, max_tokens: int = 4000):
        self.models = models
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._llm = self._create_llm(models[0])
        self._current_model = models[0]

    def _create_llm(self, model: str):
        return ChatOpenAI(
            model=model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

    def _try_invoke(self, method_name: str, *args, **kwargs):
        last_error = None
        for i, model in enumerate(self.models):
            try:
                if i > 0:
                    self._llm = self._create_llm(model)
                    self._current_model = model
                    print(f">>> 切换到模型: {model}")
                return getattr(self._llm, method_name)(*args, **kwargs)
            except Exception as e:
                err_str = str(e)
                if any(kw in err_str for kw in [
                    "FreeTierOnly", "InsufficientBalance",
                    "余额", "balance", "quota"
                ]):
                    last_error = e
                    next_model = self.models[i+1] if i+1 < len(self.models) else "无可用模型"
                    print(f">>> 模型 {model} 余额不足，尝试 {next_model} ...")
                    continue
                raise e
        raise Exception(f"所有模型额度均已耗尽: {last_error}")

    def invoke(self, *args, **kwargs):
        return self._try_invoke("invoke", *args, **kwargs)

    def stream(self, *args, **kwargs):
        return self._try_invoke("stream", *args, **kwargs)

    def with_structured_output(self, schema, **kwargs):
        """委派 structured_output 到底层 LLM"""
        return self._llm.with_structured_output(schema, **kwargs)

    @property
    def model_name(self):
        return getattr(self._llm, 'model_name', self._current_model)


# ==========================================
# 3. 法条级正则切分器
# ==========================================
class LegalRegexSplitter:
    """专为中国法律文档定制的「第X条」正则切分器，支持中文/阿拉伯数字。无条款文档回退到段落切分。"""

    ARTICLE_PATTERN = re.compile(
        r"(?:第[一二三四五六七八九十百千万零]+条)"   # 第一条、第十二条...
        r"|(?:第\d+条)",                            # 第1条、第12条...
    )
    MIN_CHUNK_CHARS = 80
    MAX_CHUNK_CHARS = 1200

    def split_documents(self, documents: list) -> list:
        # 1. 跨页缝合
        source_text_map: Dict[str, str] = {}
        for doc in documents:
            source = doc.metadata.get("source", "未知法律文件")
            if source not in source_text_map:
                source_text_map[source] = ""
            source_text_map[source] += doc.page_content + "\n"

        final_chunks: list = []
        for source, text in source_text_map.items():
            # 2. 清理硬回车断行
            text = re.sub(r'(?<=[^。；：？！\n])\n(?=[^\n])', '', text)

            # 3. 按「第X条」切分（不再要求条后有特定字符）
            raw_chunks = re.split(f"(?={self.ARTICLE_PATTERN.pattern})", text)

            # 4. 提取法律名并打标签
            law_name = os.path.basename(source).replace(".pdf", "")
            chunk_count = 0
            for chunk in raw_chunks:
                chunk = chunk.strip()
                if len(chunk) >= 15 and chunk[0] == "第":
                    enhanced = f"《{law_name}》 {chunk}"
                    final_chunks.append(LangchainDocument(
                        page_content=enhanced,
                        metadata={"source": source, "law_name": law_name}
                    ))
                    chunk_count += 1

            # 兜底：无条款文档 → 段落切分
            if chunk_count == 0 and len(text.strip()) > 50:
                self._split_by_paragraph(text, law_name, source, final_chunks)

        return final_chunks

    def _split_by_paragraph(self, text: str, law_name: str, source: str, output: list) -> int:
        paragraphs = re.split(r'\n\s*\n|(?:\f)', text.strip())
        count = 0
        buffer = ""
        for para in paragraphs:
            para = para.strip()
            if len(para) < 20:
                continue
            if len(buffer) + len(para) < self.MAX_CHUNK_CHARS:
                buffer += para + "\n"
            else:
                if len(buffer) >= self.MIN_CHUNK_CHARS:
                    output.append(LangchainDocument(
                        page_content=f"《{law_name}》\n{buffer.strip()}",
                        metadata={"source": source, "law_name": law_name}
                    ))
                    count += 1
                buffer = para + "\n"
        if len(buffer.strip()) >= self.MIN_CHUNK_CHARS:
            output.append(LangchainDocument(
                page_content=f"《{law_name}》\n{buffer.strip()}",
                metadata={"source": source, "law_name": law_name}
            ))
            count += 1
        return count


# ==========================================
# 4. 全局组件（在 lifespan 中初始化）
# ==========================================
embeddings: Optional[DashScopeEmbeddings] = None
llm: Optional[AutoFallbackLLM] = None
llm_fast: Optional[AutoFallbackLLM] = None
vectorstore: Optional[PineconeVectorStore] = None
retriever = None
app: Optional[StateGraph] = None
memory: Optional[MemorySaver] = None


def init_llm_clients():
    """初始化 LLM + Embeddings 客户端"""
    global embeddings, llm, llm_fast

    print(">>> 正在初始化 LLM 与 Embeddings 客户端...")

    # Embeddings
    embeddings = DashScopeEmbeddings(
        api_key=DASHSCOPE_API_KEY,
        model=DASHSCOPE_EMBED_MODEL,
    )

    # 对话 LLM
    llm = AutoFallbackLLM(
        models=DEEPSEEK_MODELS,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.3,
        max_tokens=4000
    )

    # 轻量 LLM（triage / query rewrite / QA）
    llm_fast = AutoFallbackLLM(
        models=DEEPSEEK_MODELS,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.1,
        max_tokens=3000
    )

    # 快速连通性测试
    try:
        import requests as _req
        _test = _req.get(DEEPSEEK_BASE_URL,
                         headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
                         timeout=10, verify=VERIFY_SSL)
        print(f">>> DeepSeek 连通性: HTTP {_test.status_code}")
    except Exception as e:
        print(f">>> ⚠️ DeepSeek 连接测试失败: {type(e).__name__}: {e}")

    print(">>> LLM 客户端初始化完成 ✅")


def init_vectorstore():
    """初始化 Pinecone 云端向量库（首次运行时自动构建）"""
    global vectorstore, retriever

    print(">>> 正在初始化 Pinecone 向量库...")

    pc = PineconeClient(api_key=PINECONE_API_KEY)

    # 兼容不同版本的 list_indexes 返回值
    existing_raw = pc.list_indexes()
    try:
        existing_names = [idx.name for idx in existing_raw]
    except (AttributeError, TypeError):
        existing_names = [idx["name"] for idx in existing_raw]

    if PINECONE_INDEX_NAME not in existing_names:
        # ---- 创建索引 ----
        print(f">>> 创建 Pinecone 索引 '{PINECONE_INDEX_NAME}' (dim={EMBED_DIMENSION})...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBED_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
        )

        # ---- 从 PDF 构建向量库 ----
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f">>> 扫描 PDF 目录: {DATA_DIR}")
        loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyMuPDFLoader)
        documents = loader.load()
        print(f">>> 加载了 {len(documents)} 页 PDF")

        if documents:
            splitter = LegalRegexSplitter()
            splits = splitter.split_documents(documents)
            print(f">>> 切分出 {len(splits)} 条法律条款，正在上传到 Pinecone ...")

            vectorstore = PineconeVectorStore.from_documents(
                documents=splits,
                embedding=embeddings,
                index_name=PINECONE_INDEX_NAME,
                pinecone_api_key=PINECONE_API_KEY
            )
            print(f">>> Pinecone 向量库构建完成 ✅ ({len(splits)} 条)")
        else:
            print("[WARN] data/ 目录下没有 PDF，创建空向量库")
            vectorstore = PineconeVectorStore(
                index_name=PINECONE_INDEX_NAME,
                embedding=embeddings,
                pinecone_api_key=PINECONE_API_KEY
            )
    else:
        # ---- 连接已有索引 ----
        print(f">>> 连接已有 Pinecone 索引 '{PINECONE_INDEX_NAME}' ...")
        vectorstore = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings,
            pinecone_api_key=PINECONE_API_KEY
        )

        # 快速检查索引状态
        try:
            index = pc.Index(PINECONE_INDEX_NAME)
            stats = index.describe_index_stats()
            dim = stats.get("dimension", "?")
            count = stats.get("total_vector_count", 0)
            print(f">>> Pinecone 索引状态: {count} 条向量, 维度={dim}")
        except Exception as e:
            print(f">>> ⚠️ 无法获取索引状态: {e}")

    # 配置检索器（返回 Top-6 去重后）
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    print(">>> Pinecone 向量库就绪 ✅")


# ==========================================
# 5. LangGraph 状态与 Pydantic 模型
# ==========================================
class LaborLawState(TypedDict):
    messages: Annotated[list, add_messages]
    chat_summary: str
    triage_result: dict
    form_data: dict
    legal_facts_summary: str
    relevant_laws: str
    similar_cases: str
    final_review: str
    reviewer_feedback: str
    retry_count: int
    is_pass_flag: bool


class TriageOutput(BaseModel):
    model_config = {"use_enum_values": True}
    action: Literal["chat", "form"] = PydanticField(description="动作类型：chat 或 form")
    category: str = PydanticField(description="意图分类")
    reply: str = PydanticField(description="回复给用户的话术")


class SearchQueries(BaseModel):
    queries: List[str] = PydanticField(description="3个核心法律检索短语")


class QualityOutput(BaseModel):
    is_pass: bool = PydanticField(description="审查是否合格")
    feedback: str = PydanticField(description="修改意见，合格填'无'")


# ==========================================
# 6. 智能体节点函数
# ==========================================
def extract_output(text: str) -> str:
    """从 LLM 输出中提取 <output> 标签内容"""
    match = re.search(r'<output>(.*?)</output>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'</thinking>(.*)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def summarize_conversation_node(state: LaborLawState):
    """节点0：Context 工程记忆压缩机（每10轮对话压缩一次）"""
    messages = state.get("messages", [])
    old_summary = state.get("chat_summary", "")

    if len(messages) <= 10:
        return {}

    print(f"\n[CONTEXT] 检测到对话达到 {len(messages)} 条，触发压缩...")
    messages_to_summarize = messages[:-4]

    form_data = state.get("form_data", {})
    if not form_data:
        form_data = {"案件发生地": "", "单位名称": "", "平均月薪": "",
                      "时间节点": "", "核心诉求": "", "详细经过": ""}

    missing_fields = [k for k, v in form_data.items() if not v]
    filled_fields = [f"{k}: {v}" for k, v in form_data.items() if v]
    missing_hint = f"缺失字段：{missing_fields}" if missing_fields else "所有字段已收集完毕"
    filled_hint = f"已收集：{' | '.join(filled_fields)}" if filled_fields else "暂无"

    prompt = f"""你是一名法庭书记员，进行 Context 压缩。

【卷宗状态】：{filled_hint} / {missing_hint}

⚠️ 绝对保留：数字、时间、地点、公司名、诉求、情绪
⚠️ 可以省略：问候、寒暄、重复解释

【已有档案】：{old_summary if old_summary else '暂无'}

【待压缩对话】：
{messages_to_summarize}

请输出更新后的完整档案（严格控制在600字以内）。"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        new_summary = response.content
        delete_messages = [RemoveMessage(id=m.id) for m in messages_to_summarize]
        return {"chat_summary": new_summary, "messages": delete_messages}
    except Exception as e:
        print(f"[CONTEXT] 压缩失败: {e}")
        return {}


def triage_node(state: LaborLawState) -> LaborLawState:
    """节点1：前台分诊 + Context 工程信息完善度审视"""
    print("\n[AI] [分诊台] 正在分析意图...")
    chat_history = state.get("messages", [])
    summary = state.get("chat_summary", "")
    summary_text = f"\n【历史档案】：\n{summary}" if summary else ""

    form_data = state.get("form_data", {})
    if not form_data:
        form_data = {"案件发生地": "", "单位名称": "", "平均月薪": "",
                      "时间节点": "", "核心诉求": "", "详细经过": ""}

    filled = {k: v for k, v in form_data.items() if v}
    missing = [k for k, v in form_data.items() if not v]
    total_fields = len(form_data)
    filled_count = len(filled)
    completeness_pct = int(filled_count / total_fields * 100) if total_fields > 0 else 0

    form_status = "\n".join([
        f"- {k}: ✅ {v}" if v else f"- {k}: ❌ [缺失]"
        for k, v in form_data.items()
    ])

    is_post_compression = bool(summary) and len(chat_history) <= 6
    compression_note = ""
    if is_post_compression:
        compression_note = """
【🔔 上下文刚被压缩，请审视卷宗状态并规划下一个问题】"""

    core_fields = ["核心诉求", "详细经过"]
    core_filled = sum(1 for f in core_fields if form_data.get(f, ""))
    has_basic_info = filled_count >= 3 and core_filled >= 1

    prompt = f"""你是一名劳动法律师助理。

【当前卷宗】：
{form_status}
{summary_text}
完善度: {completeness_pct}%。{'可转交报告' if has_basic_info else '需继续追问'}
{compression_note}

【核心指令】：
1. 绝不重复询问已填字段。每次只自然追问1个缺失字段。
2. 若完善度≥60%且含核心诉求/经过，或用户要求出报告，设action="form"，否则为"chat"。

先输出极简思考（50字内），再输出JSON：
<thinking>简述缺什么、该问什么</thinking>
<output>
{{"action": "chat" 或 "form", "category": "案件分类", "reply": "发给用户的自然语言回复"}}
</output>"""

    messages_for_llm = [SystemMessage(content=prompt)]
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            messages_for_llm.append(msg)
        elif isinstance(msg, AIMessage):
            clean = msg.content
            if "<thinking>" in clean and "</thinking>" in clean:
                clean = re.sub(r'<thinking>.*?</thinking>', '', clean, flags=re.DOTALL).strip()
            if clean:
                messages_for_llm.append(AIMessage(content=clean))

    # 调用 LLM
    raw_output = ""
    last_error = ""
    try:
        raw_output = llm_fast.invoke(messages_for_llm).content
        print(f"[TRIAGE] LLM 输出: {raw_output[:300]}...")
    except Exception as e:
        last_error = f"LLM调用失败: {type(e).__name__}: {e}"
        print(f"[ERROR] 分诊台 {last_error}")
        try:
            raw_output = llm.invoke(messages_for_llm).content
            last_error = ""
        except Exception as e2:
            last_error = f"LLM重试也失败: {type(e2).__name__}: {e2}"

    # 解析输出
    thinking_text = ""
    output_text = ""

    if raw_output:
        think_match = re.search(r'<thinking>(.*?)</thinking>', raw_output, re.DOTALL)
        if think_match:
            thinking_text = think_match.group(1).strip()

        out_match = re.search(r'<output>(.*?)</output>', raw_output, re.DOTALL)
        if out_match:
            output_text = out_match.group(1).strip()
        else:
            after_think = re.search(r'</thinking>(.*)', raw_output, re.DOTALL)
            if after_think:
                output_text = after_think.group(1).strip()
            else:
                output_text = raw_output.strip()

    # 解析 JSON
    triage_result = None
    if output_text:
        json_str = re.sub(r'^```(?:json)?\s*', '', output_text)
        json_str = re.sub(r'\s*```$', '', json_str)
        json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                triage_result = {
                    "action": parsed.get("action", "chat"),
                    "category": parsed.get("category", "通用咨询"),
                    "reply": parsed.get("reply", "")
                }
                print(f"[TRIAGE] JSON 解析成功: action={triage_result['action']}")
            except json.JSONDecodeError as e:
                print(f"[WARNING] JSON 解析失败: {e}")

    # 兜底
    if not triage_result:
        action = "chat"
        if output_text and ("form" in output_text.lower() or "转交" in output_text):
            action = "form"

        if raw_output and len(raw_output.strip()) > 10:
            reply_text = re.sub(r'</?thinking>', '', raw_output).strip()
            reply_text = re.sub(r'</?output>', '', reply_text).strip()
        elif last_error:
            reply_text = f"[错误] {last_error}"
        else:
            reply_text = "您好，我是劳动法智能助理，请问有什么可以帮您的？"

        triage_result = {"action": action, "category": "通用咨询", "reply": reply_text}
        print(f"[TRIAGE] 兜底模式: action={action}")

    ai_content = triage_result["reply"]
    if thinking_text:
        print(f"[TRIAGE] 思考: {thinking_text[:200]}...")

    print(f"[TRIAGE] 动作={triage_result['action']}, 分类={triage_result['category']}, 完善度={completeness_pct}%")
    return {"triage_result": triage_result, "messages": [AIMessage(content=ai_content)]}


def fact_summarizer_node(state: LaborLawState) -> LaborLawState:
    """节点2：事实梳理员"""
    form_data = state.get("form_data", {})
    form_text = "\n".join([f"- {k}: {v}" for k, v in form_data.items()])
    prompt = f"""请根据案件信息梳理关键事实：
【用户表单信息】：
{form_text}

请在 <thinking> 内进行逻辑自洽检查，在 <output> 内梳理：
1. 争议焦点 2. 关键时间节点 3. 证据情况分析 4. 法律适用预判"""
    messages = [SystemMessage(content="你是劳动法律师助理"), HumanMessage(content=prompt)]
    return {"legal_facts_summary": extract_output(llm_fast.invoke(messages).content)}


def legal_researcher_node(state: LaborLawState) -> LaborLawState:
    """节点3：法条检索 + 案例参考"""
    print("\n[法条+案例专员] 正在查询知识库...")
    summary = state.get("legal_facts_summary", "")
    if not summary:
        return {"relevant_laws": "暂无事实，无法检索", "similar_cases": ""}

    # 提取检索词
    rewrite_prompt = f"根据以下案件事实，提取3个法律检索关键词，用逗号分隔。只输出关键词。\n事实：{summary[:300]}"
    try:
        queries_text = llm_fast.invoke([HumanMessage(content=rewrite_prompt)]).content
        queries = [q.strip() for q in queries_text.split('，') if q.strip()][:3]
        if not queries:
            queries = [summary[:50]]
    except Exception:
        queries = [summary[:50]]

    # 向量检索
    all_docs = []
    seen = set()
    try:
        batch_results = retriever.batch(queries)
        for docs in batch_results:
            for doc in docs:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    all_docs.append(doc)
    except Exception:
        for q in queries:
            for doc in retriever.invoke(q):
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    all_docs.append(doc)

    all_docs = all_docs[:6]
    relevant_docs = "\n\n".join([d.page_content for d in all_docs]) if all_docs else "暂无法条"

    prompt = f"""你是一名精通中国劳动法的裁判者。
【案件事实】：{summary}
【相关法条】：{relevant_docs}

在 <thinking> 中排查法条冲突（注意特别法优于一般法）。
在 <output> 中输出：
## 法条适用分析
1. 适用法律条款 2. 适用说明 3. 赔偿计算依据 4. 程序建议

## 参考案例要旨
简述1个典型相似判例要旨及参考价值。"""
    messages = [SystemMessage(content="你是资深法律专家"), HumanMessage(content=prompt)]
    result = extract_output(llm.invoke(messages).content)
    return {"relevant_laws": result, "similar_cases": ""}


def compliance_reviewer_node(state: LaborLawState) -> LaborLawState:
    """节点4：合规审核员（汇合点）"""
    print("\n[合规审核员] 正在汇编最终报告...")
    facts = state.get("legal_facts_summary", "")
    laws = state.get("relevant_laws", "")
    feedback = state.get("reviewer_feedback", "")
    form_text = "\n".join([f"- {k}: {v}" for k, v in state.get("form_data", {}).items()])
    prompt = f"原始信息：\n{form_text}\n事实：\n{facts}\n法条分析：\n{laws}"

    if feedback:
        print(f"⚠️ 收到修改意见：{feedback}")
        prompt += f"\n\n【主编打回修改意见】：{feedback}\n请严格修正！"

    prompt += """
在 <thinking> 中审视前置分析。在 <output> 中输出完整合规法律建议报告，必须包含三部分：

## 适用法条
列出本案最核心的2-3条法律条款及适用要点。每条写明：法律名称、条款号、适用要点。

## 关键证据与注意事项
用户需准备的核心证据及取证注意事项，逐条列出。

## 操作建议
具体可执行的操作步骤和风险提示，按优先级排列。

⚠️ 禁止使用表格。三部分缺一不可。输出正文在 1500-2000 字之间。"""

    messages = [SystemMessage(content="你是资深劳动法律师"), HumanMessage(content=prompt)]
    return {"final_review": extract_output(llm.invoke(messages).content)}


def quality_inspector_node(state: LaborLawState) -> LaborLawState:
    """节点5：主编质检员"""
    print("\n[主编质检员] 审核中...")
    review = state.get("final_review", "")
    retry_count = state.get("retry_count", 0)

    if retry_count >= 1:
        print(f"[QA] ⏭️ 已重写过一次，直接通过 | 重试: {retry_count}")
        return {"reviewer_feedback": "", "retry_count": retry_count + 1, "is_pass_flag": True}

    form_text = "\n".join([f"- {k}: {v}" for k, v in state.get("form_data", {}).items()])
    prompt = f"""审查报告是否正面回答用户诉求？有无逻辑漏洞或法条引用不当？
【用户诉求】：{form_text}
【报告】：{review[:1500]}
只回答 PASS 或 FAIL + 简要意见。"""

    try:
        qa_out = llm_fast.with_structured_output(QualityOutput).invoke([HumanMessage(content=prompt)])
        is_pass, feedback = qa_out.is_pass, qa_out.feedback
    except Exception as e:
        print(f"[QA ERROR] {e}, 换 llm 重试")
        try:
            qa_out = llm.with_structured_output(QualityOutput).invoke([HumanMessage(content=prompt)])
            is_pass, feedback = qa_out.is_pass, qa_out.feedback
        except Exception as e2:
            print(f"[QA ERROR] 重试也失败，放行: {e2}")
            is_pass, feedback = True, "无"

    print(f"[QA] {'✅ 通过' if is_pass else '❌ 打回'} | 重试: {retry_count} | {feedback}")
    return {
        "reviewer_feedback": feedback if not is_pass else "",
        "retry_count": retry_count + 1,
        "is_pass_flag": is_pass
    }


# ==========================================
# 7. 构建 LangGraph 工作流
# ==========================================
def build_langgraph_app():
    """构建并编译 LangGraph 多智能体工作流"""
    global app, memory

    print(">>> 正在构建 LangGraph 多智能体工作流...")

    workflow = StateGraph(LaborLawState)

    # 注册节点
    workflow.add_node("summarizer", summarize_conversation_node)
    workflow.add_node("triage", triage_node)
    workflow.add_node("fact_summarizer", fact_summarizer_node)
    workflow.add_node("legal_researcher", legal_researcher_node)
    workflow.add_node("compliance_reviewer", compliance_reviewer_node)
    workflow.add_node("quality_inspector", quality_inspector_node)

    workflow.set_entry_point("summarizer")
    workflow.add_edge("summarizer", "triage")

    # 路由 1：分诊
    def route_after_triage(state: LaborLawState):
        return "process_case" if state.get("triage_result", {}).get("action") == "form" else "end"

    workflow.add_conditional_edges(
        "triage", route_after_triage,
        {"process_case": "fact_summarizer", "end": END}
    )

    # 串行流水线
    workflow.add_edge("fact_summarizer", "legal_researcher")
    workflow.add_edge("legal_researcher", "compliance_reviewer")
    workflow.add_edge("compliance_reviewer", "quality_inspector")

    # 质检纠错环
    def route_after_qa(state: LaborLawState):
        if state.get("is_pass_flag", True) or state.get("retry_count", 0) >= 2:
            return "end"
        return "retry"

    workflow.add_conditional_edges(
        "quality_inspector", route_after_qa,
        {"end": END, "retry": "compliance_reviewer"}
    )

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory, interrupt_before=["fact_summarizer"])
    print(">>> LangGraph 工作流构建完成 ✅")


# ==========================================
# 8. FastAPI 应用 & 生命周期
# ==========================================
@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    """应用生命周期：启动时加载所有客户端和模型"""
    print("=" * 50)
    print(">>> 🚀 FastAPI 后端服务启动中...")
    print("=" * 50)

    # 顺序初始化（LLM → 向量库 → LangGraph）
    init_llm_clients()
    init_vectorstore()
    build_langgraph_app()

    print("=" * 50)
    print(">>> ✅ 所有组件初始化完成，等待请求...")
    print("=" * 50)
    yield
    print(">>> 服务关闭")


fastapi_app = FastAPI(
    title="劳动法智能助理 API",
    description="LangGraph 多智能体推理引擎后端",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS 中间件
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 9. API 数据模型
# ==========================================
class ChatRequest(BaseModel):
    query: str = PydanticField(..., description="用户问题")
    thread_id: Optional[str] = PydanticField(None, description="会话线程ID（不传则自动创建）")

class ChatResponse(BaseModel):
    reply: str = PydanticField(..., description="AI 回复")
    action: str = PydanticField(..., description="动作类型：chat | form")
    category: str = PydanticField(..., description="意图分类")
    thread_id: str = PydanticField(..., description="会话线程ID")
    thinking: str = PydanticField("", description="AI 思考过程（可选）")


class AnalyzeRequest(BaseModel):
    thread_id: Optional[str] = PydanticField(None, description="会话线程ID")
    form_data: Dict[str, str] = PydanticField(..., description="案件信息表单")
    force: bool = PydanticField(False, description="是否强制生成报告")

class AnalyzeResponse(BaseModel):
    thread_id: str = PydanticField(..., description="会话线程ID")
    legal_facts_summary: str = PydanticField("", description="事实梳理")
    relevant_laws: str = PydanticField("", description="法条分析 + 案例参考")
    final_review: str = PydanticField("", description="最终合规报告")
    retry_count: int = PydanticField(0, description="质检重试次数")
    is_pass_flag: bool = PydanticField(True, description="是否通过质检")
    reviewer_feedback: str = PydanticField("", description="质检反馈")


# ==========================================
# 10. API 端点
# ==========================================

@fastapi_app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "service": "劳动法智能助理 API",
        "version": "2.0.0",
        "components": {
            "llm": llm is not None,
            "llm_fast": llm_fast is not None,
            "embeddings": embeddings is not None,
            "vectorstore": vectorstore is not None,
            "langgraph": app is not None,
        }
    }


@fastapi_app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    对话接口

    接收用户 query，通过 LangGraph 分诊台判断意图后返回回复。
    - 如果意图是闲聊/普法咨询：直接返回 AI 回复
    - 如果需要案件分析：返回 action="form"，前端提示用户填写卷宗
    """
    if not app:
        raise HTTPException(status_code=503, detail="后端引擎尚未初始化完成")

    thread_id = request.thread_id or f"api-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # 运行 LangGraph（summarizer → triage，停在 fact_summarizer 之前）
        result = app.invoke({"messages": [HumanMessage(content=request.query)]}, config)

        triage = result.get("triage_result", {})
        reply = triage.get("reply", "抱歉，我暂时无法处理您的请求。")
        action = triage.get("action", "chat")
        category = triage.get("category", "通用咨询")

        # 提取 thinking（不返回给前端，仅日志）
        thinking = ""
        msgs = result.get("messages", [])
        for m in reversed(msgs):
            if isinstance(m, AIMessage) and m.content:
                think_match = re.search(r'<thinking>(.*?)</thinking>', m.content, re.DOTALL)
                if think_match:
                    thinking = think_match.group(1).strip()
                break

        return ChatResponse(
            reply=reply,
            action=action,
            category=category,
            thread_id=thread_id,
            thinking=thinking,
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"处理请求时出错: {str(e)}")


@fastapi_app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    案件分析接口

    接收案件卷宗信息，运行完整的 LangGraph 多智能体流程：
    triage → fact_summarizer → legal_researcher → compliance_reviewer → quality_inspector

    返回完整分析报告。
    """
    if not app:
        raise HTTPException(status_code=503, detail="后端引擎尚未初始化完成")

    thread_id = request.thread_id or f"case-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    # 构建案件描述
    form_text = "\n".join([f"{k}：{v}" for k, v in request.form_data.items() if v])
    if not form_text.strip():
        raise HTTPException(status_code=400, detail="form_data 不能全部为空")

    case_msg = HumanMessage(content=f"请分析以下劳动法案件：\n{form_text}")

    try:
        # Step 1: 运行 triage（会自动停在 interrupt_before=["fact_summarizer"]）
        init_result = app.invoke({"messages": [case_msg]}, config)

        triage = init_result.get("triage_result", {})

        # Step 2: 准备继续（修复 triage 结果 + 注入 form_data）
        if triage.get("action") != "form" or request.force:
            app.update_state(config, {
                "triage_result": {
                    "action": "form",
                    "category": "案件分析",
                    "reply": "正在为您分析案件..."
                }
            })
        app.update_state(config, {"form_data": request.form_data})

        # Step 3: 从 interrupt 点继续执行完整流程
        final_result = app.invoke(None, config)

        return AnalyzeResponse(
            thread_id=thread_id,
            legal_facts_summary=final_result.get("legal_facts_summary", ""),
            relevant_laws=final_result.get("relevant_laws", ""),
            final_review=final_result.get("final_review", ""),
            retry_count=final_result.get("retry_count", 0),
            is_pass_flag=final_result.get("is_pass_flag", True),
            reviewer_feedback=final_result.get("reviewer_feedback", ""),
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")


@fastapi_app.get("/status/{thread_id}")
async def get_status(thread_id: str):
    """查询指定线程的状态"""
    if not app:
        raise HTTPException(status_code=503, detail="后端尚未就绪")

    config = {"configurable": {"thread_id": thread_id}}
    try:
        state = app.get_state(config)
        return {
            "thread_id": thread_id,
            "values": state.values if hasattr(state, 'values') else {},
            "next": state.next if hasattr(state, 'next') else [],
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"未找到线程: {str(e)}")


# ==========================================
# 11. 入口
# ==========================================
if __name__ == "__main__":
    uvicorn.run("api:fastapi_app", host="0.0.0.0", port=8000, reload=True)
